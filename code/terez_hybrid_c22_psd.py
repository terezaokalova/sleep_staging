#!/usr/bin/env python3
import os
import sys
import glob
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ─── Argument parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data-root",    type=Path, required=True)
parser.add_argument("--results-root", type=Path, required=True)
parser.add_argument("--figures-root", type=Path, required=True)
parser.add_argument("--n-jobs",       type=int,   default=16)
parser.add_argument("--batch-size",   type=int,   default=32)
parser.add_argument("--epochs",       type=int,   default=50)
parser.add_argument("--lr",           type=float, default=2e-4)
parser.add_argument("--seq-length",   type=int,   default=30)
parser.add_argument("--seq-stride",   type=int,   default=5)
args = parser.parse_args()

# ─── Logging setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
LOGFILE    = SCRIPT_DIR/"terez_hybrid_c22_psd.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE               = args.data_root
PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
PSD_DATA_DIR       = BASE/"psd_features_sleepedf"

RESULTS_DIR = args.results_root
FIGURES_DIR = args.figures_root

for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
for sub in ("plots","models","metrics"):
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

# ─── Hyperparams & seed ─────────────────────────────────────────────────────────
SEED           = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

BATCH_SIZE    = args.batch_size
NUM_EPOCHS    = args.epochs
LEARNING_RATE = args.lr
SEQ_LENGTH    = args.seq_length
SEQ_STRIDE    = args.seq_stride
NUM_WORKERS   = args.n_jobs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"BASE: {BASE}")
for d in (PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR):
    logger.info(f"{d}: exists={d.exists()}")
if not (PROCESSED_DATA_DIR.exists() and CATCH22_DATA_DIR.exists() and PSD_DATA_DIR.exists()):
    sys.exit("Missing data directories")

# ─── Utility ────────────────────────────────────────────────────────────────────
def get_true_subject_id(fn):
    b = Path(fn).stem
    return b[:5] if b.startswith(("SC4","ST7")) else b[:6]

# ─── Dataset ───────────────────────────────────────────────────────────────────
class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, psd_dir, ids=None):
        raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        c22 = glob.glob(str(c22_dir/"*_c22.csv"))
        psd = glob.glob(str(psd_dir/"*_psd.npz"))
        if ids:
            raw = [p for p in raw if any(i in p for i in ids)]
            c22 = [p for p in c22 if any(i in p for i in ids)]
            psd = [p for p in psd if any(i in p for i in ids)]
        self.raw_map = {Path(p).stem.split("_")[0]: p for p in raw}
        self.c22_map = {Path(p).stem.split("_")[0]: p for p in c22}
        self.psd_map= {Path(p).stem.split("_")[0]: p for p in psd}
        common = sorted(set(self.raw_map)&set(self.c22_map)&set(self.psd_map))
        if not common:
            raise ValueError("No overlap")
        seqs, c22s, psds, lbls = [], [], [], []
        for rid in common:
            d       = np.load(self.raw_map[rid])
            s, l    = d["sequences"], d["seq_labels"]
            f22     = pd.read_csv(self.c22_map[rid]).drop("label",1).values
            fpsd    = np.load(self.psd_map[rid])["features"]
            for feats, store in ((f22, c22s),(fpsd, psds)):
                n, D     = s.shape[0], feats.shape[1]
                exp      = n*SEQ_LENGTH
                if feats.shape[0]!=exp:
                    newf = np.zeros((n,SEQ_LENGTH,D),np.float32)
                    for i in range(n):
                        st, ed = i*SEQ_STRIDE, i*SEQ_STRIDE+SEQ_LENGTH
                        if ed<=feats.shape[0]:
                            newf[i]=feats[st:ed]
                        else:
                            a = feats.shape[0]-st
                            newf[i,:a]=feats[st:]
                            newf[i,a:]=feats[-1]
                    feats=newf
                else:
                    feats=feats.reshape(n,SEQ_LENGTH,-1).astype(np.float32)
                store.append(feats)
            seqs.append(s.astype(np.float32))
            lbls.append(l.astype(np.int64))
        self.sequences  = torch.from_numpy(np.concatenate(seqs,0))
        self.c22_feats  = torch.from_numpy(np.concatenate(c22s,0))
        self.psd_feats  = torch.from_numpy(np.concatenate(psds,0))
        self.seq_labels = torch.from_numpy(np.concatenate(lbls,0))
    def __len__(self): return len(self.sequences)
    def __getitem__(self,i):
        return (self.sequences[i],
                self.c22_feats[i],
                self.psd_feats[i],
                self.seq_labels[i])

# ─── Model ──────────────────────────────────────────────────────────────────────
class EpochEncoder(nn.Module):
    def __init__(self, e=128):
        super().__init__()
        self.conv1=nn.Conv1d(2,32,3,padding=1); self.bn1=nn.BatchNorm1d(32)
        self.conv2=nn.Conv1d(32,64,3,padding=1);self.bn2=nn.BatchNorm1d(64)
        self.conv3=nn.Conv1d(64,128,3,padding=1);self.bn3=nn.BatchNorm1d(128)
        self.conv4=nn.Conv1d(128,128,3,padding=1);self.bn4=nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128,32,1), nn.ReLU(),
            nn.Conv1d(32,128,1), nn.Sigmoid()
        )
        self.apool=nn.AdaptiveAvgPool1d(48)
        self.fc   = nn.Linear(128*48,e)
        self.ln   = nn.LayerNorm(e)
        self.drop = nn.Dropout(0.2)
    def forward(self,x):
        B,S,C,T=x.shape
        x=x.view(B*S,C,T)
        for c,b in ((self.conv1,self.bn1),(self.conv2,self.bn2),(self.conv3,self.bn3)):
            x=F.relu(b(c(x))); x=self.pool(x)
        x=F.relu(self.bn4(self.conv4(x)))
        x=x*self.attn(x)
        x=self.apool(x).view(B*S,-1)
        x=self.drop(F.relu(self.fc(x))); x=self.ln(x)
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self,in_,e=64):
        super().__init__()
        self.ln0=nn.LayerNorm(in_)
        self.fc1=nn.Linear(in_,256);self.ln1=nn.LayerNorm(256)
        self.fc2=nn.Linear(256,128); self.ln2=nn.LayerNorm(128)
        self.fc3=nn.Linear(128,e);   self.ln3=nn.LayerNorm(e)
        self.drop=nn.Dropout(0.2)
    def forward(self,x):
        B,S,D=x.shape
        x=x.view(B*S,D)
        x=self.drop(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x=self.drop(F.relu(self.ln2(self.fc2(x))))
        x=self.drop(F.relu(self.ln3(self.fc3(x))))
        return x.view(B,S,-1)

class PSDEncoder(nn.Module):
    def __init__(self,in_,e=64):
        super().__init__()
        self.ln0=nn.LayerNorm(in_)
        self.fc1=nn.Linear(in_,128); self.ln1=nn.LayerNorm(128)
        self.fc2=nn.Linear(128,e);   self.ln2=nn.LayerNorm(e)
        self.drop=nn.Dropout(0.2)
    def forward(self,x):
        B,S,D=x.shape
        x=x.view(B*S,D)
        x=self.drop(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x=self.drop(F.relu(self.ln2(self.fc2(x))))
        return x.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self,c22_d,psd_d,raw_e=128,c22_e=64,psd_e=64,
                 nc=5,nl=3,nh=8,dp=0.2,sl=30):
        super().__init__()
        self.eenc=EpochEncoder(raw_e)
        self.cenc=C22Encoder(c22_d,c22_e)
        self.penc=PSDEncoder(psd_d,psd_e)
        cd=raw_e+c22_e+psd_e
        self.fuse=nn.Sequential(
            nn.Linear(cd,cd),
            nn.LayerNorm(cd),
            nn.ReLU(), nn.Dropout(dp),
            nn.Linear(cd,cd)
        )
        self.lnf = nn.LayerNorm(cd)
        self.pos = nn.Parameter(torch.randn(1,sl,cd))
        self.cls = nn.Parameter(torch.randn(1,nc,cd))
        layer=nn.TransformerEncoderLayer(
            d_model=cd,nhead=nh,
            dim_feedforward=8*cd,dropout=dp,
            batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer,num_layers=nl)
        self.ar = nn.Sequential(
            nn.Linear(raw_e,128),nn.LayerNorm(128),
            nn.ReLU(),nn.Dropout(dp),
            nn.Linear(128,nc)
        )
        self.ac = nn.Sequential(
            nn.Linear(c22_e,128),nn.LayerNorm(128),
            nn.ReLU(),nn.Dropout(dp),
            nn.Linear(128,nc)
        )
        self.ap = nn.Sequential(
            nn.Linear(psd_e,128),nn.LayerNorm(128),
            nn.ReLU(),nn.Dropout(dp),
            nn.Linear(128,nc)
        )
        self.fc = nn.Linear(cd,256)
        self.ln = nn.LayerNorm(256)
        self.dp2=nn.Dropout(dp)
        self.outs=nn.ModuleList([nn.Linear(256,1) for _ in range(nc)])
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self,raw,c22,psd):
        r=self.eenc(raw)
        c=self.cenc(c22)
        p=self.penc(psd)
        B,S,_=r.shape
        if c.size(1)!=S or p.size(1)!=S:
            m=min(r.size(1),c.size(1),p.size(1))
            r,c,p=r[:,:m],c[:,:m],p[:,:m]; S=m
        ar=self.ar(r); ac=self.ac(c); ap=self.ap(p)
        x=torch.cat([r,c,p],2).view(B*S,-1)
        x=self.fuse(x); x=F.relu(self.lnf(x))
        x=x.view(B,S,-1)+self.pos[:,:S]
        cls=self.cls.expand(B,-1,-1)
        x=torch.cat([x,cls],1)
        x=self.tr(x)
        main=x[:,:S]
        return main,ar,ac,ap

# ─── Loss & Helpers ────────────────────────────────────────────────────────────
def focal_loss(inputs, targets, alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    if inputs.ndim==3:
        B,S,C=inputs.shape
        l=inputs.reshape(-1,C)
    else:
        l=inputs.reshape(-1,inputs.shape[-1])
    t=targets.reshape(-1)
    logp=F.log_softmax(l,1)
    p=torch.exp(logp).clamp(min=1e-7)
    ce=F.nll_loss(logp,t,reduction='none')
    pt=p.gather(1,t.unsqueeze(1)).squeeze(1)
    n1=(t==1).float(); n3=(t==3).float(); rem=(t==4).float()
    a=alpha_general*(1-n1-n3-rem)+alpha_n1*n1+0.5*n3+0.45*rem
    return (a*((1-pt)**gamma)*ce).mean()

def mixup_batch(raw,c22,psd,labels,alpha=0.2):
    lam=np.random.beta(alpha,alpha) if alpha>0 else 1
    idx=torch.randperm(raw.size(0)).to(raw.device)
    return (lam*raw+(1-lam)*raw[idx],
            lam*c22+(1-lam)*c22[idx],
            lam*psd+(1-lam)*psd[idx],
            labels, labels[idx], lam)

def train_epoch(model, loader, optimizer, scheduler=None, mixup_alpha=0.2):
    model.train()
    run_loss, corr, tot = 0.0, 0, 0
    for raw,c22,psd,labels in loader:
        raw=torch.nan_to_num(raw).to(device,non_blocking=True)
        c22=torch.nan_to_num(c22).to(device,non_blocking=True)
        psd=torch.nan_to_num(psd).to(device,non_blocking=True)
        labels=labels.to(device,non_blocking=True)
        if np.random.rand()<0.5:
            r,c_,p,la,lb,lam=mixup_batch(raw,c22,psd,labels,mixup_alpha)
            use_m=True
        else:
            r,c_,p=raw,c22,psd; la=labels; use_m=False
        optimizer.zero_grad()
        main,ar,ac,ap=model(r,c_,p)
        ml=focal_loss(main,la)
        if use_m:
            lr=lam*focal_loss(ar,la)+(1-lam)*focal_loss(ar,lb)
            lc=lam*focal_loss(ac,la)+(1-lam)*focal_loss(ac,lb)
            lp=lam*focal_loss(ap,la)+(1-lam)*focal_loss(ap,lb)
        else:
            lr=focal_loss(ar,la); lc=focal_loss(ac,la); lp=focal_loss(ap,la)
        loss=ml+0.3*(lr+lc+lp)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if scheduler and isinstance(scheduler,(optim.lr_scheduler.OneCycleLR,optim.lr_scheduler.CyclicLR)):
            scheduler.step()
        run_loss+=loss.item()*raw.size(0)
        preds=main.argmax(-1)
        if not use_m:
            corr+=(preds==labels).sum().item()
            tot+=labels.numel()
    return run_loss/len(loader.dataset), corr/tot if tot>0 else 0.0

def eval_epoch(model, loader, apply_smoothing=True):
    model.eval()
    run_loss=0.0; all_raw=[]; all_lbl=[]
    with torch.no_grad():
        for raw,c22,psd,labels in loader:
            raw=torch.nan_to_num(raw).to(device)
            c22=torch.nan_to_num(c22).to(device)
            psd=torch.nan_to_num(psd).to(device)
            labels=labels.to(device)
            out=model(raw,c22,psd)[0]
            if not torch.isfinite(out).all(): continue
            loss=focal_loss(out,labels)
            run_loss+=loss.item()*raw.size(0)
            preds=out.argmax(-1).cpu().numpy().ravel()
            all_raw.append(preds); all_lbl.append(labels.cpu().numpy().ravel())
    if not all_raw:
        return float('nan'),0,0,[],[]
    raw_ar=np.concatenate(all_raw)
    lbl_ar=np.concatenate(all_lbl)
    raw_acc=(raw_ar==lbl_ar).mean()
    if apply_smoothing:
        from scipy.signal import medfilt # optional
        smoothed=raw_ar.copy() # or your smoothing fn
        sm_acc=(smoothed==lbl_ar).mean()
        preds=smoothed
    else:
        sm_acc=raw_acc; preds=raw_ar
    return run_loss/len(loader.dataset), raw_acc, sm_acc, preds, lbl_ar

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    # build subject->recordings
    smap={}
    for f in glob.glob(str(PROCESSED_DATA_DIR/"*_sequences.npz")):
        rid=Path(f).stem.split("_")[0]
        smap.setdefault(get_true_subject_id(rid),[]).append(rid)

    subs=list(smap); np.random.seed(SEED); np.random.shuffle(subs)
    folds=np.array_split(subs,5)

    all_conf=np.zeros((5,5),int)
    all_probs, all_lbls=[],[]

    for k in range(5):
        train_sub=[s for i,f in enumerate(folds) if i!=k for s in f]
        test_sub =folds[k]
        train_ids=[rid for s in train_sub for rid in smap[s]]
        test_ids =[rid for s in test_sub  for rid in smap[s]]

        train_ds=HybridSleepDataset(PROCESSED_DATA_DIR,CATCH22_DATA_DIR,PSD_DATA_DIR,train_ids)
        test_ds =HybridSleepDataset(PROCESSED_DATA_DIR,CATCH22_DATA_DIR,PSD_DATA_DIR,test_ids)

        # ----- sampler exactly as Stefan -----
        flat = train_ds.seq_labels.view(-1).numpy()
        cnts = np.bincount(flat, minlength=5)
        w    = 1.0/np.sqrt(cnts+1e-6); w[1]*=1.5
        seq_w=np.zeros(len(train_ds))
        for i in range(len(train_ds)):
            lab=train_ds.seq_labels[i].numpy()
            seq_w[i]=w[lab].mean()
        sampler=WeightedRandomSampler(seq_w,len(seq_w),True)

        train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,
                                sampler=sampler,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)
        test_loader =DataLoader(test_ds, batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

        c22_d=train_ds.c22_feats.shape[-1]
        psd_d=train_ds.psd_feats.shape[-1]
        model = HybridSleepTransformer(c22_d,psd_d).to(device)

        opt   = optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=1e-4)
        sched = optim.lr_scheduler.OneCycleLR(opt,
                    max_lr=LEARNING_RATE,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=len(train_loader),
                    pct_start=0.3,
                    div_factor=25,
                    final_div_factor=1000)

        train_losses,val_losses=[],[]
        for ep in range(NUM_EPOCHS):
            tr_loss,_=train_epoch(model,train_loader,opt,sched)
            vl_loss,raw_acc,sm_acc,_,_ = eval_epoch(model,test_loader,True)
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            logger.info(f"Fold{k+1} Ep{ep+1}/{NUM_EPOCHS} "
                        f"tr_loss={tr_loss:.4f} vl_loss={vl_loss:.4f} "
                        f"raw_acc={raw_acc:.4f} sm_acc={sm_acc:.4f}")

        # convergence
        plt.plot(train_losses,label="train")
        plt.plot(val_losses,  label="val")
        plt.legend(); plt.savefig(FIGURES_DIR/f"conv_fold{k+1}.png"); plt.clf()

        # final eval
        probs,lbls=eval_epoch_probs(model,test_loader)
        preds     = probs.argmax(1)
        cm        = confusion_matrix(lbls,preds)
        all_conf += cm
        all_probs.append(probs); all_lbls.append(lbls)
        rep=classification_report(lbls,preds,target_names=["W","N1","N2","N3","REM"])
        logger.info(f"Fold{k+1} report:\n{rep}")

    # overall CM
    plt.matshow(all_conf); plt.colorbar()
    plt.xticks(range(5),["W","N1","N2","N3","REM"])
    plt.yticks(range(5),["W","N1","N2","N3","REM"])
    plt.savefig(FIGURES_DIR/"confusion_matrix.png"); plt.clf()

    # ROC
    all_p=np.vstack(all_probs)
    all_l=np.hstack(all_lbls)
    bin_l=label_binarize(all_l,classes=[0,1,2,3,4])
    for i,cls in enumerate(["W","N1","N2","N3","REM"]):
        fpr,tpr,_=roc_curve(bin_l[:,i],all_p[:,i])
        rocA=auc(fpr,tpr)
        plt.plot(fpr,tpr,label=f"{cls}({rocA:.2f})")
    plt.plot([0,1],[0,1],"--")
    plt.legend(loc="lower right")
    plt.savefig(FIGURES_DIR/"roc_auc.png"); plt.clf()

    logger.info("Done.")

if __name__=="__main__":
    main()
