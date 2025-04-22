#!/usr/bin/env python3
import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

BASE               = Path("/users/okalova/sleep/STAT-4830-GOALZ-project/data")
PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
RESULTS_DIR        = BASE/"hybrid_ensemble_results"
for d in ("plots","models","metrics"):
    (RESULTS_DIR/d).mkdir(parents=True, exist_ok=True)

SEED        = 42
BATCH_SIZE  = 32
NUM_EPOCHS  = 50
LEARNING_RATE = 2e-4
SEQ_LENGTH  = 30
SEQ_STRIDE  = 5

# Stage 1 params
LR1         = 1e-4
DESIRED_N1  = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

class Stage1Dataset(Dataset):
    def __init__(self, raw_dir):
        files, seqs, labs = glob.glob(str(raw_dir/"*_sequences.npz")), [], []
        for f in files:
            d = np.load(f)
            s, lab = d["sequences"], d["seq_labels"]
            if s.ndim==4:
                N,S,C,T = s.shape
                s = s.reshape(N*S, C, T)
                lab = lab.reshape(N*S)
            seqs.append(s)
            labs.append((lab==1).astype(np.int64))
        X = torch.from_numpy(np.concatenate(seqs,0)).float()
        y = torch.from_numpy(np.concatenate(labs,0)).long()
        self.X, self.y = X, y
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

class Stage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2,16,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(32,2)
        )
    def forward(self,x): return self.net(x)

def get_true_subject_id(fn):
    b = Path(fn).stem
    return b[:5] if b.startswith(("SC4","ST7")) else b[:6]

def group_by_true_subjects(data_dir):
    files = glob.glob(str(data_dir/"*_sequences.npz"))
    m = {}
    for f in files:
        rid = Path(f).stem.split("_")[0]
        subj = get_true_subject_id(rid)
        m.setdefault(subj, []).append(rid)
    return m

class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, recording_ids=None):
        all_raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        all_c22 = glob.glob(str(c22_dir/"*_c22.csv"))
        if recording_ids:
            all_raw = [p for p in all_raw if any(r in p for r in recording_ids)]
            all_c22 = [p for p in all_c22 if any(r in p for r in recording_ids)]
        raw_map = {Path(p).stem.split("_")[0]:p for p in all_raw}
        c22_map = {Path(p).stem.split("_")[0]:p for p in all_c22}
        ids = sorted(raw_map.keys() & c22_map.keys())
        seqs, c22s, labels = [], [], []
        for rid in ids:
            d = np.load(raw_map[rid])
            s, lab = d["sequences"], d["seq_labels"]
            if s.ndim==4:
                N,S,C,T = s.shape
                s = s.reshape(N*S, C, T)
                lab = lab.reshape(N*S)
            df = pd.read_csv(c22_map[rid])
            f  = df.drop(columns="label").values
            nS,feat = s.shape[0], f.shape[1]
            exp = nS*SEQ_LENGTH
            if f.shape[0]!=exp:
                newf = np.zeros((nS,SEQ_LENGTH,feat),np.float32)
                for i in range(nS):
                    st,en = i*SEQ_STRIDE, i*SEQ_STRIDE+SEQ_LENGTH
                    if en<=f.shape[0]:
                        newf[i] = f[st:en]
                    else:
                        av = f.shape[0]-st
                        if av>0:
                            newf[i,:av] = f[st:]
                            newf[i,av:] = f[-1]
                        else:
                            newf[i] = newf[i-1]
                f = newf
            else:
                f = f.reshape(nS,SEQ_LENGTH,feat).astype(np.float32)
            seqs.append(s.astype(np.float32))
            c22s.append(f)
            labels.append(lab.astype(np.int64))
        self.raw    = torch.from_numpy(np.concatenate(seqs,0)).float()
        self.c22    = torch.from_numpy(np.concatenate(c22s,0)).float()
        self.labels = torch.from_numpy(np.concatenate(labels,0)).long()
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        return ( self.raw[i].unsqueeze(0),
                 self.c22[i].unsqueeze(0),
                 self.labels[i].unsqueeze(0) )

class EpochEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2,32,3,padding=1); self.bn1=nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,3,padding=1);self.bn2=nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3,padding=1);self.bn3=nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,128,3,padding=1);self.bn4=nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.attn  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128,32,1), nn.ReLU(),
            nn.Conv1d(32,128,1), nn.Sigmoid()
        )
        self.adp   = nn.AdaptiveAvgPool1d(48)
        self.fc    = nn.Linear(128*48, emb)
        self.ln    = nn.LayerNorm(emb)
        self.do    = nn.Dropout(0.2)
    def forward(self,x):
        B,S,C,T = x.shape
        x = x.view(B*S,C,T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x * self.attn(x)
        x = self.adp(x).view(B*S,-1)
        x = self.ln(self.do(F.relu(self.fc(x))))
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self, in_d, emb=64):
        super().__init__()
        self.ln0 = nn.LayerNorm(in_d)
        self.fc1 = nn.Linear(in_d,256); self.ln1=nn.LayerNorm(256)
        self.fc2 = nn.Linear(256,128); self.ln2=nn.LayerNorm(128)
        self.fc3 = nn.Linear(128,emb); self.ln3=nn.LayerNorm(emb)
        self.do  = nn.Dropout(0.2)
    def forward(self,x):
        # x: (B,S,SEQ_LENGTH,D)
        x = x.mean(dim=2)            # -> (B,S,D)
        B,S,D = x.shape
        h = x.view(B*S, D)
        h = self.do(F.relu(self.ln1(self.fc1(self.ln0(h)))))
        h = self.do(F.relu(self.ln2(self.fc2(h))))
        h = self.do(F.relu(self.ln3(self.fc3(h))))
        return h.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_emb=128, c22_emb=64,
                 num_classes=5, num_layers=3, num_heads=8,
                 dropout=0.2, seq_len=SEQ_LENGTH):
        super().__init__()
        self.eenc = EpochEncoder(raw_emb)
        self.cenc = C22Encoder(c22_dim, c22_emb)
        D = raw_emb + c22_emb
        self.fuse = nn.Sequential(
            nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(D,D)
        )
        self.lnf  = nn.LayerNorm(D)
        self.pos  = nn.Parameter(torch.randn(1,seq_len,D))
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads,
            dim_feedforward=8*D,
            dropout=dropout, batch_first=True
        )
        self.tfm  = nn.TransformerEncoder(layer, num_layers)
        self.aux1 = nn.Sequential(
            nn.Linear(raw_emb,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128,num_classes)
        )
        self.aux2 = nn.Sequential(
            nn.Linear(c22_emb,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128,num_classes)
        )
        self.fc   = nn.Linear(D,256); self.ln_fc=nn.LayerNorm(256)
        self.do   = nn.Dropout(dropout)
        self.cls  = nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])
        self.n1det = nn.Sequential(
            nn.Linear(D,128), nn.LayerNorm(128), nn.ReLU()
        )
        self.n1lstm= nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.n1attn= nn.MultiheadAttention(128,4,batch_first=True)
        self.n1out = nn.Linear(128,1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.zero_()

    def forward(self, raw, c22):
        r = self.eenc(raw)              # (B,S,raw_emb)
        c = self.cenc(c22)              # (B,S,c22_emb)
        B,Sr,_ = r.shape
        _,Sc,_ = c.shape
        if Sr!=Sc:
            m = min(Sr,Sc)
            r = r[:,:m]; c = c[:,:m]
        a1 = self.aux1(r)
        a2 = self.aux2(c)
        x  = torch.cat([r,c],dim=2).view(B*Sr,-1)
        x  = self.lnf(self.fuse(x)).view(B,Sr,-1)
        x  = x + self.pos[:,:x.size(1),:]
        cls_tok = torch.randn(B,1,x.size(2),device=x.device)
        x  = torch.cat([x,cls_tok],dim=1)
        out= self.tfm(x)
        seq, _ = out[:,:Sr,:], out[:,Sr:,:]
        nf = self.n1det(seq)
        h,_= self.n1lstm(nf); a,_ = self.n1attn(h,h,h)
        n1s = self.n1out(a)
        sh = self.do(F.relu(self.ln_fc(self.fc(seq))))
        outs=[]
        for i,fc in enumerate(self.cls):
            if i==1: 
                outs.append(fc(sh)+n1s)
            else:
                outs.append(fc(sh))
        return torch.cat(outs,dim=2), a1, a2

def focal_loss(inputs, targets,
               alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    B,S,C = inputs.shape
    L = inputs.view(-1, C)
    t = targets.view(-1)
    logp = F.log_softmax(L,1)
    p    = torch.exp(logp).clamp(min=1e-7)
    ce   = F.nll_loss(logp, t, reduction='none')
    pt   = p.gather(1,t.unsqueeze(1)).squeeze(1)
    m1   = (t==1).float()
    a    = alpha_general*(1-m1) + alpha_n1*m1
    return (a*((1-pt)**gamma)*ce).mean()

def mixup_batch(raw, c22, y, alpha=0.2):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1
    B   = raw.size(0)
    idx = torch.randperm(B).to(raw.device)
    return lam*raw + (1-lam)*raw[idx], lam*c22+(1-lam)*c22[idx], y, y[idx], lam

def train_epoch(model, loader, opt, sched, alpha, detector):
    model.train()
    loss_sum=0; corr=0; tot=0
    for raw,c22,labels in loader:
        raw    = raw.to(device)
        c22    = c22.to(device)
        labels = labels.to(device)
        if np.random.rand()<0.5:
            raw2,c22_2,la,lb,lam = mixup_batch(raw,c22,labels,alpha)
            use_m=True
        else:
            raw2,c22_2,la,lb,lam = raw,c22,labels,labels,1
            use_m=False
        opt.zero_grad()
        B,S,C,T = raw2.shape
        with torch.no_grad():
            dlog = detector(raw2.view(B*S,C,T))
            dsc  = dlog[:,1].view(B,S)
        out, a1, a2 = model(raw2, c22_2)
        out[:,:,1] += dsc
        mloss = focal_loss(out, la)
        l1    = focal_loss(a1, la)
        l2    = focal_loss(a2, la)
        loss  = mloss + 0.3*l1 + 0.3*l2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if isinstance(sched, (optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR)):
            sched.step()
        loss_sum += loss.item()*raw.size(0)
        if not use_m:
            preds = out.argmax(-1)
            corr += (preds==labels).sum().item()
            tot  += labels.numel()
    return loss_sum/len(loader.dataset), corr/tot

def smooth_predictions(preds, win=5):
    if win%2==0: win+=1
    sm = preds.copy()
    pad= np.pad(preds,(win//2,)*2,mode='edge')
    for i in range(len(preds)):
        w = pad[i:i+win]
        if 1 in w:
            c = (w==1).sum()
            if c>1 or w[win//2]==1:
                sm[i]=1
            else:
                bc = np.bincount(w, minlength=5)
                if bc[1]==1: bc[1]=0
                sm[i]=bc.argmax()
        else:
            sm[i]=np.bincount(w, minlength=5).argmax()
    for i in range(1,len(sm)-1):
        if sm[i] in (0,4) and sm[i-1]==sm[i+1]!=sm[i]:
            sm[i]=sm[i-1]
    return sm

def eval_epoch(model, loader, smooth=True, detector=None):
    model.eval()
    loss_sum=0
    all_p, all_l = [], []
    with torch.no_grad():
        for raw,c22,labels in loader:
            raw    = raw.to(device)
            c22    = c22.to(device)
            labels = labels.to(device)
            B,S,C,T = raw.shape
            dlog = detector(raw.view(B*S,C,T))
            dsc  = dlog[:,1].view(B,S)
            out, a1, a2 = model(raw, c22)
            out[:,:,1] += dsc
            loss_sum += focal_loss(out, labels).item()*raw.size(0)
            p = out.argmax(-1).cpu().numpy().ravel()
            l = labels.cpu().numpy().ravel()
            all_p.append(p); all_l.append(l)
    all_p = np.concatenate(all_p)
    all_l = np.concatenate(all_l)
    raw_acc = (all_p==all_l).mean()
    if smooth:
        sp = smooth_predictions(all_p)
        return loss_sum/len(loader.dataset), raw_acc, (sp==all_l).mean(), sp, all_l
    else:
        return loss_sum/len(loader.dataset), raw_acc, raw_acc, all_p, all_l

def main():
    # train stage1
    ds1 = Stage1Dataset(PROCESSED_DATA_DIR)
    w   = np.bincount(ds1.y.numpy(), minlength=2)
    w1  = (w[0]/w[1])*(DESIRED_N1/(1-DESIRED_N1))
    sw  = np.where(ds1.y.numpy()==1, w1, 1.0)
    l1  = DataLoader(ds1, BATCH_SIZE,
                     sampler=WeightedRandomSampler(sw,len(sw),True))
    det = Stage1Detector().to(device)
    o1  = optim.Adam(det.parameters(), lr=LR1)
    for ep in range(10):
        tot=ls=0
        det.train()
        for X,y in l1:
            X,y = X.to(device), y.to(device)
            o1.zero_grad()
            loss = F.cross_entropy(det(X), y)
            loss.backward(); o1.step()
            ls += loss.item()*X.size(0); tot+=X.size(0)
        print(f"[Stage1] Ep{ep+1}: loss={ls/tot:.4f}")
    det.eval()
    for p in det.parameters(): p.requires_grad=False

    subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
    subs     = list(subj_map.keys())
    np.random.seed(SEED); np.random.shuffle(subs)
    folds    = np.array_split(subs,5)
    cv_res   = {'acc':[],'f1_w':[],'f1_n1':[],'f1_n2':[],'f1_n3':[],'f1_r':[]}
    all_cm   = np.zeros((5,5))

    for k in range(5):
        test_subs  = folds[k]
        train_subs = [s for i,f in enumerate(folds) if i!=k for s in f]
        train_ids  = [rid for s in train_subs for rid in subj_map[s]]
        test_ids   = [rid for s in test_subs  for rid in subj_map[s]]

        tr_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, train_ids)
        te_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, test_ids)

        flat = tr_ds.labels.view(-1).numpy()
        cnts = np.bincount(flat, minlength=5)
        wts  = 1.0/np.sqrt(cnts+1e-6)
        wts[1]*=1.5
        seq_w = np.array([wts[tr_ds.labels[i].item()] for i in range(len(tr_ds))])
        samp  = WeightedRandomSampler(seq_w,len(seq_w),True)

        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                           sampler=samp, num_workers=4, pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4, pin_memory=True)

        model = HybridSleepTransformer(tr_ds.c22.size(-1)).to(device)
        o2    = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        s2    = optim.lr_scheduler.OneCycleLR(o2, max_lr=LEARNING_RATE,
                   epochs=NUM_EPOCHS, steps_per_epoch=len(tr_ld),
                   pct_start=0.3, div_factor=25, final_div_factor=1000)

        best_n1=0; pat=7; cnt=0
        for ep in range(NUM_EPOCHS):
            tl,ta = train_epoch(model, tr_ld, o2, s2, 0.2, det)
            vl,ra,sa,pred,lab = eval_epoch(model, te_ld, True, det)
            rpt = classification_report(lab, pred,
                                       target_names=["W","N1","N2","N3","REM"],
                                       output_dict=True)
            f1n1 = rpt["N1"]["f1-score"]
            print(f"Fold{k+1} Ep{ep+1}: train_loss={tl:.4f} N1 f1={f1n1:.4f}")
            if f1n1>best_n1:
                best_n1, cnt = f1n1, 0
                torch.save(model.state_dict(), RESULTS_DIR/f"models/best_fold{k+1}.pth")
            else:
                cnt+=1
                if cnt>=pat: break

        model.load_state_dict(torch.load(RESULTS_DIR/f"models/best_fold{k+1}.pth"))
        _,_,_,pred,lab = eval_epoch(model, te_ld, True, det)
        cm = confusion_matrix(lab,pred)
        rpt= classification_report(lab,pred,
                    target_names=["W","N1","N2","N3","REM"],output_dict=True)
        all_cm += cm
        cv_res['acc'].append((pred==lab).mean())
        cv_res['f1_w'].append(rpt["W"]["f1-score"])
        cv_res['f1_n1'].append(rpt["N1"]["f1-score"])
        cv_res['f1_n2'].append(rpt["N2"]["f1-score"])
        cv_res['f1_n3'].append(rpt["N3"]["f1-score"])
        cv_res['f1_r'].append(rpt["REM"]["f1-score"])
        np.savez(RESULTS_DIR/f"metrics/fold{k+1}.npz", cm=cm, report=rpt)

    print("CV acc:", cv_res['acc'])
    for key in ['f1_w','f1_n1','f1_n2','f1_n3','f1_r']:
        arr = np.array(cv_res[key])
        print(f"{key}: {arr.mean():.4f} ± {arr.std():.4f}")
    print("Overall CM:\n", all_cm)

if __name__=="__main__":
    main()
