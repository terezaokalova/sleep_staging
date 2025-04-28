import os, sys, glob, itertools, json, hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# ==== Initialization ==== 
# -------------------------
# HOME = Path(os.environ.get("HOME", "."))
HOME = Path("/mnt/sauce/littlab/users/kimliang")
# PROJECT = HOME / 'Documents/STAT4830' / "STAT-4830-GOALZ-project"
PROJECT = HOME / 'sleep' / "STAT-4830-GOALZ-project"
BASE = PROJECT / "data"
print("Project base:", PROJECT)

PROCESSED_DATA_DIR = BASE / "processed_sleepedf"
CATCH22_DATA_DIR   = BASE / "c22_processed_sleepedf"
RESULTS_DIR        = BASE / "hybrid_psd_model_results"
for sub in ["plots","models","metrics","grid_search"]:
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

print("BASE:", BASE)
print("Processed exists?", PROCESSED_DATA_DIR.exists())
print("C22 exists?", CATCH22_DATA_DIR.exists())

# reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# ==== Dataset Class ======
# -------------------------
class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, recording_ids=None, seq_length=30, seq_stride=5):
        self.seq_length = seq_length
        self.seq_stride = seq_stride
        raw_list = sorted(glob.glob(str(Path(raw_dir)/"*_sequences.npz")))
        c22_list = sorted(glob.glob(str(Path(c22_dir)/"*_c22.csv")))
        if recording_ids:
            raw_list = [f for f in raw_list if any(r in f for r in recording_ids)]
            c22_list = [f for f in c22_list if any(r in f for r in recording_ids)]
        self.raw_map = {Path(f).stem.split("_")[0]:f for f in raw_list}
        self.c22_map = {Path(f).stem.split("_")[0]:f for f in c22_list}
        common = sorted(set(self.raw_map)&set(self.c22_map))
        if not common:
            raise ValueError("No overlapping recordings between raw & Catch22!")
        seqs, c22, labels = [], [], []
        for rid in common:
            d = np.load(self.raw_map[rid])
            s, l = d['sequences'].astype(np.float32), d['seq_labels'].astype(np.int64)
            df = pd.read_csv(self.c22_map[rid])
            f = df.drop(columns=['label']).values.astype(np.float32)
            n = s.shape[0]
            exp = n*self.seq_length
            if f.shape[0]!=exp:
                dim = f.shape[1]
                newf = np.zeros((n,self.seq_length,dim),np.float32)
                for i in range(n):
                    start=i*self.seq_stride; end=start+self.seq_length
                    if end<=f.shape[0]: newf[i]=f[start:end]
                    else:
                        avail=f.shape[0]-start
                        if avail>0:
                            newf[i,:avail]=f[start:]
                            newf[i,avail:]=f[-1]
                        else:
                            newf[i]=newf[i-1]
                f=newf
            else:
                f=f.reshape(n,self.seq_length,-1)
            seqs.append(s); c22.append(f); labels.append(l)
        self.sequences = torch.from_numpy(np.concatenate(seqs,0))
        self.c22_feats = torch.from_numpy(np.concatenate(c22,0))
        self.seq_labels= torch.from_numpy(np.concatenate(labels,0))
    def __len__(self): return len(self.sequences)
    def __getitem__(self,i): return self.sequences[i], self.c22_feats[i], self.seq_labels[i]

# -------------------------
# ==== Utilities ==========
# -------------------------
def get_true_subject_id(rid): return rid[:5]

def group_by_true_subjects(data_dir):
    files = glob.glob(str(Path(data_dir)/"*_sequences.npz"))
    m = {}
    for f in files:
        rid = Path(f).stem.split("_")[0]
        subj = get_true_subject_id(rid)
        m.setdefault(subj,[]).append(rid)
    return m

# -------------------------
# ==== Model ==============
# -------------------------
class EpochEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2,32,3,padding=1), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.adapt = nn.AdaptiveAvgPool1d(48)
        self.fc = nn.Linear(128*48,emb)
        self.ln = nn.LayerNorm(emb)
    def forward(self,x):
        B,S,C,T = x.shape
        x = x.view(B*S,C,T)
        x = self.conv(x)
        x = self.adapt(x).view(B*S,-1)
        x = F.relu(self.fc(x))
        x = self.ln(x)
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self,in_dim,emb=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim,256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256,emb), nn.LayerNorm(emb)
        )
    def forward(self,x):
        B,S,D = x.shape
        y = x.view(B*S,D)
        y = self.fc(y)
        return y.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self,c22_dim,raw_emb=128,c22_emb=64,cl=5,nl=2,nh=4,dp=0.1,sl=30):
        super().__init__()
        self.epoch = EpochEncoder(raw_emb)
        self.c22   = C22Encoder(c22_dim,c22_emb)
        D = raw_emb+c22_emb
        self.pos  = nn.Parameter(torch.randn(1,sl,D))
        enc = nn.TransformerEncoderLayer(d_model=D,nhead=nh,dim_feedforward=4*D,dropout=dp,batch_first=True)
        self.trans= nn.TransformerEncoder(enc,nl)
        self.out  = nn.Linear(D,cl)
    def forward(self,raw,c22):
        r = self.epoch(raw); c = self.c22(c22)
        B,SR,_=r.shape
        x = torch.cat([r,c],-1)+self.pos[:,:SR]
        x = self.trans(x)
        return self.out(x)

# -------------------------
# ==== Loss & Training ====
# -------------------------
def focal_loss(logits,labels,alpha=0.25,gamma=2.0):
    B,S,C=logits.shape
    logp = F.log_softmax(logits,-1).view(-1,C)
    pt   = torch.exp(logp)
    tgt  = labels.view(-1)
    ce   = F.nll_loss(logp,tgt,reduction='none')
    ptg  = pt.gather(1,tgt.unsqueeze(1)).squeeze(1)
    w    = alpha*((1-ptg)**gamma)
    return (w*ce).mean()

def train_epoch(m,loader,opt):
    m.train(); tot_loss=0
    for raw,c22,lab in loader:
        raw,c22,lab = raw.to(device),c22.to(device),lab.to(device)
        opt.zero_grad()
        out = m(raw,c22)
        loss= focal_loss(out,lab)
        loss.backward(); opt.step()
        tot_loss+=loss.item()*raw.size(0)
    return tot_loss/len(loader.dataset)

def eval_epoch(m,loader):
    m.eval(); preds, labs=[],[]
    with torch.no_grad():
        for raw,c22,lab in loader:
            out=m(raw.to(device),c22.to(device))
            p=out.argmax(-1).cpu().numpy()
            preds.append(p); labs.append(lab.numpy())
    p=np.concatenate(preds).ravel(); t=np.concatenate(labs).ravel()
    acc=(p==t).mean()
    rpt=classification_report(t,p,output_dict=True)
    return acc, rpt, confusion_matrix(t,p)

# -------------------------
# ==== Experiment Runner =
# -------------------------
def run_experiment(hp, rd):
    train_ids,test_ids = [],[] # implement split logic
    ds_tr = HybridSleepDataset(PROCESSED_DATA_DIR,CATCH22_DATA_DIR,train_ids,
                                seq_length=hp['seq_length'],seq_stride=hp['seq_length']//2)
    ds_te = HybridSleepDataset(PROCESSED_DATA_DIR,CATCH22_DATA_DIR,test_ids,
                                seq_length=hp['seq_length'],seq_stride=hp['seq_length']//2)
    loader_tr=DataLoader(ds_tr,batch_size=hp['batch_size'],shuffle=True)
    loader_te=DataLoader(ds_te,batch_size=hp['batch_size'],shuffle=False)
    model = HybridSleepTransformer(ds_tr.c22_feats.shape[-1],nl=hp['num_layers'],nh=hp['num_heads'],dp=hp['dropout'],sl=hp['seq_length']).to(device)
    opt   = optim.AdamW(model.parameters(),lr=hp['learning_rate'])
    for ep in range(10): train_epoch(model,loader_tr,opt)
    acc,rpt,cm=eval_epoch(model,loader_te)
    return {'accuracy':acc,'report':rpt}

# -------------------------
# ==== Grid Search ========
# -------------------------
PARAM_GRID={'learning_rate':[1e-4,3e-4],'batch_size':[32,64],'seq_length':[20,30],'num_layers':[2,3],'num_heads':[4,8],'dropout':[0.1,0.3]}
def main():
    best=None; best_hp=None
    for hp in (dict(zip(PARAM_GRID,comb)) for comb in itertools.product(*PARAM_GRID.values())):
        key=hashlib.md5(str(hp).encode()).hexdigest()[:8]
        rd=RESULTS_DIR/"grid_search"/key; rd.mkdir(exist_ok=True)
        res=run_experiment(hp,rd)
        with open(rd/"result.json","w") as f: json.dump(res,f)
        if best is None or res['accuracy']>best:
            best, best_hp = res['accuracy'], hp
    print("Best HP",best_hp,"Acc",best)

if __name__=="__main__": main()
