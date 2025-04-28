#!/usr/bin/env python3
import os
import sys
import glob
import json
import hashlib
import itertools
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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ─── Argument parsing ────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument("--data-root",    type=Path, required=True)
# parser.add_argument("--results-root", type=Path, required=True)
# parser.add_argument("--figures-root", type=Path, required=True)
# parser.add_argument("--n-jobs",       type=int,   default=16)
# parser.add_argument("--batch-size",   type=int,   default=32)
# parser.add_argument("--epochs",       type=int,   default=50)
# parser.add_argument("--lr",           type=float, default=2e-4)
# parser.add_argument("--seq-length",   type=int,   default=30)
# parser.add_argument("--seq-stride",   type=int,   default=5)
# parser.add_argument("--grid-search",  action="store_true",
#     help="Run full grid-search over hyperparameters")
# args = parser.parse_args()


# ─── Manual Args ────────────────────────────────────────────────────────────────
class Args:
    pass

args = Args()
args.data_root    = Path("/mnt/sauce/littlab/users/kimliang/sleep/STAT-4830-GOALZ-project/data")
args.results_root = Path("/mnt/sauce/littlab/users/kimliang/sleep/STAT-4830-GOALZ-project/results")
args.figures_root = Path("/mnt/sauce/littlab/users/kimliang/sleep/STAT-4830-GOALZ-project/figures")
args.n_jobs       = 16
args.batch_size   = 32
args.epochs       = 50
args.lr           = 2e-4
args.seq_length   = 30
args.seq_stride   = 5
args.grid_search  = True

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
# BASE               = args.data_root
# PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
# CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
# PSD_DATA_DIR       = BASE/"psd_features_sleepedf"
# RESULTS_DIR        = args.results_root
# FIGURES_DIR        = args.figures_root

HOME = Path("/mnt/sauce/littlab/users/kimliang")
# HOME = os.environ.get("HOME", ".")
# PROJECT = HOME / 'Documents/STAT4830' / "STAT-4830-GOALZ-project"
PROJECT = HOME / 'sleep' / "STAT-4830-GOALZ-project"
BASE = PROJECT / "data"
print("Project base:", PROJECT)

PROCESSED_DATA_DIR = BASE / "processed_sleepedf"
CATCH22_DATA_DIR   = BASE / "c22_processed_sleepedf"
RESULTS_DIR        = BASE / "hybrid_psd_model_results"
FIGURES_DIR        = BASE / "hybrid_psd_model_figures"
PSD_DATA_DIR       = BASE / "features_psd_sleep_edf"

# make sure subdirs exist
(RESULTS_DIR/"grid_search").mkdir(parents=True, exist_ok=True)
for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
for sub in ("plots","models","metrics"):
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

# ─── Global seed ────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

for d in (PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR):
    logger.info(f"{d}: exists = {d.exists()}")
if not (PROCESSED_DATA_DIR.exists() and CATCH22_DATA_DIR.exists() and PSD_DATA_DIR.exists()):
    sys.exit("ERROR: Missing data directories")

# ─── Utility ────────────────────────────────────────────────────────────────────
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

# ─── Dataset ───────────────────────────────────────────────────────────────────
class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, psd_dir,
                 recording_ids=None, seq_length=30, seq_stride=5):
        self.seq_length = seq_length
        self.seq_stride = seq_stride

        raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        c22 = glob.glob(str(c22_dir/"*_c22.csv"))
        psd = glob.glob(str(psd_dir/"*_psd.npz"))
        if recording_ids:
            raw = [p for p in raw if any(r in p for r in recording_ids)]
            c22 = [p for p in c22 if any(r in p for r in recording_ids)]
            psd = [p for p in psd if any(r in p for r in recording_ids)]

        self.raw_map = {Path(p).stem.split("_")[0]: p for p in raw}
        self.c22_map = {Path(p).stem.split("_")[0]: p for p in c22}
        self.psd_map= {Path(p).stem.split("_")[0]: p for p in psd}

        common = sorted(set(self.raw_map)&set(self.c22_map)&set(self.psd_map))
        if not common:
            raise ValueError("No overlapping recordings!")
        seqs, c22s, psds, lbls = [], [], [], []

        for rid in common:
            dat     = np.load(self.raw_map[rid])
            s, labs = dat["sequences"], dat["seq_labels"]
            # drop label column by name:
            feats_c22 = pd.read_csv(self.c22_map[rid]) \
                          .drop("label", axis=1).values
            feats_psd = np.load(self.psd_map[rid])["features"]

            for feats, store in ((feats_c22, c22s), (feats_psd, psds)):
                n, D  = s.shape[0], feats.shape[1]
                exp   = n * self.seq_length
                if feats.shape[0] != exp:
                    newf = np.zeros((n, self.seq_length, D), np.float32)
                    for i in range(n):
                        st = i*self.seq_stride
                        ed = st+self.seq_length
                        if ed <= feats.shape[0]:
                            newf[i] = feats[st:ed]
                        else:
                            a = feats.shape[0]-st
                            newf[i,:a] = feats[st:]
                            newf[i,a:] = feats[-1]
                    feats = newf
                else:
                    feats = feats.reshape(n, self.seq_length, -1).astype(np.float32)
                store.append(feats)

            seqs.append(s.astype(np.float32))
            lbls.append(labs.astype(np.int64))

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

# ─── Model components ──────────────────────────────────────────────────────────
class EpochEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2,32,3,padding=1); self.bn1=nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,3,padding=1); self.bn2=nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3,padding=1);self.bn3=nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,128,3,padding=1);self.bn4=nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128,32,1), nn.ReLU(),
            nn.Conv1d(32,128,1), nn.Sigmoid()
        )
        self.apool = nn.AdaptiveAvgPool1d(48)
        self.fc    = nn.Linear(128*48,emb)
        self.ln    = nn.LayerNorm(emb)
        self.drop  = nn.Dropout(0.2)
    def forward(self,x):
        B,S,C,T = x.shape
        x = x.view(B*S,C,T)
        for conv,bn in ((self.conv1,self.bn1),
                        (self.conv2,self.bn2),
                        (self.conv3,self.bn3)):
            x = F.relu(bn(conv(x))); x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = x * self.attn(x)
        x = self.apool(x).view(B*S,-1)
        x = self.drop(F.relu(self.fc(x))); x = self.ln(x)
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self,in_dim, emb=64):
        super().__init__()
        self.ln0=nn.LayerNorm(in_dim)
        self.fc1=nn.Linear(in_dim,256);  self.ln1=nn.LayerNorm(256)
        self.fc2=nn.Linear(256,128);     self.ln2=nn.LayerNorm(128)
        self.fc3=nn.Linear(128,emb);     self.ln3=nn.LayerNorm(emb)
        self.drop=nn.Dropout(0.2)
    def forward(self,x):
        B,S,D = x.shape
        x = x.view(B*S,D)
        x = self.drop(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x = self.drop(F.relu(self.ln2(self.fc2(x))))
        x = self.drop(F.relu(self.ln3(self.fc3(x))))
        return x.view(B,S,-1)

class PSDEncoder(nn.Module):
    def __init__(self,in_dim, emb=64):
        super().__init__()
        self.ln0=nn.LayerNorm(in_dim)
        self.fc1=nn.Linear(in_dim,128); self.ln1=nn.LayerNorm(128)
        self.fc2=nn.Linear(128,emb);    self.ln2=nn.LayerNorm(emb)
        self.drop=nn.Dropout(0.2)
    def forward(self,x):
        B,S,D = x.shape
        x = x.view(B*S,D)
        x = self.drop(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x = self.drop(F.relu(self.ln2(self.fc2(x))))
        return x.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self,
                 c22_dim, psd_dim,
                 raw_emb=128, c22_emb=64, psd_emb=64,
                 num_classes=5, num_layers=3, num_heads=8,
                 dropout=0.2, seq_length=30):
        super().__init__()
        self.eenc = EpochEncoder(raw_emb)
        self.cenc = C22Encoder(c22_dim, c22_emb)
        self.penc = PSDEncoder(psd_dim, psd_emb)
        D = raw_emb + c22_emb + psd_emb

        self.fuse = nn.Sequential(
            nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(D,D)
        )
        self.lnfuse = nn.LayerNorm(D)
        self.pos    = nn.Parameter(torch.randn(1, seq_length, D))
        self.cls    = nn.Parameter(torch.randn(1, num_classes, D))

        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads,
            dim_feedforward=8*D, dropout=dropout, batch_first=True
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=num_layers)

        # auxiliary heads
        self.ar = nn.Sequential(nn.Linear(raw_emb,128), nn.LayerNorm(128),
                                 nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128,num_classes))
        self.ac = nn.Sequential(nn.Linear(c22_emb,128), nn.LayerNorm(128),
                                 nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128,num_classes))
        self.ap = nn.Sequential(nn.Linear(psd_emb,128), nn.LayerNorm(128),
                                 nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128,num_classes))

        # final per-class heads
        self.fc_shared = nn.Linear(D,256)
        self.ln_shared = nn.LayerNorm(256)
        self.drop2     = nn.Dropout(dropout)
        self.outs      = nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self, raw, c22, psd):
        r = self.eenc(raw)
        c = self.cenc(c22)
        p = self.penc(psd)
        B,S,_ = r.shape
        # align seq lengths
        if c.size(1)!=S or p.size(1)!=S:
            m = min(r.size(1), c.size(1), p.size(1))
            r,c,p = r[:,:m], c[:,:m], p[:,:m]
            S = m

        ar = self.ar(r); ac = self.ac(c); ap = self.ap(p)

        x = torch.cat([r,c,p], dim=2).view(B*S, -1)
        x = self.fuse(x); x = F.relu(self.lnfuse(x))
        x = x.view(B,S,-1) + self.pos[:,:S,:]

        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([x, cls], dim=1)
        x   = self.tr(x)

        main = x[:, :S]  # (B,S,D)
        # per-class logits
        outs=[]
        shared = self.drop2(F.relu(self.ln_shared(self.fc_shared(main))))
        for i,head in enumerate(self.outs):
            outs.append(head(shared))
        out = torch.cat(outs, dim=2)  # (B,S,num_classes)

        return out, ar, ac, ap

# ─── Loss and helpers ───────────────────────────────────────────────────────────
def focal_loss(logits, labels,
               alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    B,S,C = logits.shape
    flat_logits = logits.reshape(-1, C)
    flat_tgt    = labels.reshape(-1)
    logp = F.log_softmax(flat_logits, dim=1)
    p    = torch.exp(logp).clamp(min=1e-7)
    ce   = F.nll_loss(logp, flat_tgt, reduction='none')
    pt   = p.gather(1, flat_tgt.unsqueeze(1)).squeeze(1)

    n1_mask  = (flat_tgt==1).float()
    n3_mask  = (flat_tgt==3).float()
    rem_mask = (flat_tgt==4).float()

    alpha = alpha_general*(1 - n1_mask - n3_mask - rem_mask) \
          + alpha_n1*n1_mask \
          + 0.5*n3_mask \
          + 0.45*rem_mask

    return (alpha * ((1-pt)**gamma) * ce).mean()

def mixup_batch(raw,c22,psd,labels,alpha=0.2):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1
    idx = torch.randperm(raw.size(0)).to(raw.device)
    return (lam*raw + (1-lam)*raw[idx],
            lam*c22 + (1-lam)*c22[idx],
            lam*psd + (1-lam)*psd[idx],
            labels, labels[idx], lam)

def train_epoch(model, loader, optimizer, scheduler=None, mixup_alpha=0.2):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for raw,c22,psd,labels in loader:
        raw   = torch.nan_to_num(raw).to(device)
        c22   = torch.nan_to_num(c22).to(device)
        psd   = torch.nan_to_num(psd).to(device)
        labels= labels.to(device)

        if np.random.rand() < 0.5:
            r_,c_,p_,la,lb,lam = mixup_batch(raw,c22,psd,labels,mixup_alpha)
            use_m = True
        else:
            r_,c_,p_ = raw,c22,psd; la=labels; use_m=False

        optimizer.zero_grad()
        out, ar, ac, ap = model(r_,c_,p_)
        loss_main = focal_loss(out, la)

        if use_m:
            loss_ar = lam*focal_loss(ar,la)+(1-lam)*focal_loss(ar,lb)
            loss_ac = lam*focal_loss(ac,la)+(1-lam)*focal_loss(ac,lb)
            loss_ap = lam*focal_loss(ap,la)+(1-lam)*focal_loss(ap,lb)
        else:
            loss_ar = focal_loss(ar,la)
            loss_ac = focal_loss(ac,la)
            loss_ap = focal_loss(ap,la)

        loss = loss_main + 0.3*(loss_ar+loss_ac+loss_ap)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, (
            optim.lr_scheduler.OneCycleLR,
            optim.lr_scheduler.CyclicLR
        )):
            scheduler.step()

        running_loss += loss.item() * raw.size(0)
        preds = out.argmax(dim=2)
        if not use_m:
            correct += (preds==labels).sum().item()
            total   += labels.numel()

    return running_loss/len(loader.dataset), (correct/total if total>0 else 0.0)

def eval_epoch(model, loader, apply_smoothing=True):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for raw,c22,psd,labels in loader:
            raw   = torch.nan_to_num(raw).to(device)
            c22   = torch.nan_to_num(c22).to(device)
            psd   = torch.nan_to_num(psd).to(device)
            labels= labels.to(device)

            out = model(raw,c22,psd)[0]
            if not torch.isfinite(out).all(): continue
            loss = focal_loss(out, labels)
            running_loss += loss.item()*raw.size(0)

            preds = out.argmax(dim=2).cpu().numpy().ravel()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().ravel())

    if not all_preds:
        return float("nan"),0,0,[],[]
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    raw_acc = (all_preds==all_labels).mean()

    # no specialized smoothing here; just return raw
    return running_loss/len(loader.dataset), raw_acc, raw_acc, all_preds, all_labels

def eval_epoch_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for raw,c22,psd,labels in loader:
            raw = raw.to(device); c22=c22.to(device); psd=psd.to(device)
            logits = model(raw,c22,psd)[0]
            probs  = F.softmax(logits,dim=2).cpu().numpy().reshape(-1,5)
            lbls   = labels.cpu().numpy().reshape(-1)
            all_probs.append(probs)
            all_labels.append(lbls)
    return np.concatenate(all_probs,0), np.concatenate(all_labels,0)

# ─── Grid parameters ────────────────────────────────────────────────────────────
PARAM_GRID = {
    "lr":         [1e-4, 2e-4, 3e-4],
    "batch_size":[32,     64      ],
    "seq_length":[20,     30      ],
    "seq_stride":[5,      10      ],
    "num_layers":[2,      3       ],
    "num_heads": [4,      8       ],
    "dropout":   [0.1,    0.2     ]
}

def run_cv(hp):
    """Run 5-fold CV under one hyperparameter dict hp, return mean raw accuracy."""
    subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
    subs     = list(subj_map.keys())
    np.random.seed(SEED); np.random.shuffle(subs)
    folds    = np.array_split(subs, 5)

    accs = []
    for k in range(5):
        # split ids
        train_sub = [s for i,f in enumerate(folds) if i!=k for s in f]
        test_sub  = folds[k]
        train_ids = [rid for s in train_sub for rid in subj_map[s]]
        test_ids  = [rid for s in test_sub  for rid in subj_map[s]]

        # datasets
        train_ds = HybridSleepDataset(
            PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR,
            recording_ids=train_ids,
            seq_length=hp["seq_length"], seq_stride=hp["seq_stride"]
        )
        test_ds  = HybridSleepDataset(
            PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR,
            recording_ids=test_ids,
            seq_length=hp["seq_length"], seq_stride=hp["seq_stride"]
        )

        # sampler & loaders
        flat_lbl = train_ds.seq_labels.reshape(-1).numpy()
        counts   = np.bincount(flat_lbl, minlength=5)
        wts      = 1.0 / np.sqrt(counts + 1e-6)
        wts[1]  *= 1.5
        seq_w    = np.array([
            wts[train_ds.seq_labels[i].numpy()].mean()
            for i in range(len(train_ds))
        ])
        sampler = WeightedRandomSampler(seq_w, len(seq_w), replacement=True)

        tr_loader = DataLoader(
            train_ds,
            batch_size=hp["batch_size"],
            sampler=sampler,
            num_workers=args.n_jobs,
            pin_memory=True
        )
        te_loader = DataLoader(
            test_ds,
            batch_size=hp["batch_size"],
            shuffle=False,
            num_workers=args.n_jobs,
            pin_memory=True
        )

        # model
        c22_d = train_ds.c22_feats.shape[-1]
        psd_d = train_ds.psd_feats.shape[-1]
        model = HybridSleepTransformer(
            c22_dim   = c22_d,
            psd_dim   = psd_d,
            raw_emb   = 128,
            c22_emb   = 64,
            psd_emb   = 64,
            num_classes =5,
            num_layers = hp["num_layers"],
            num_heads  = hp["num_heads"],
            dropout    = hp["dropout"],
            seq_length = hp["seq_length"]
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr    = hp["lr"],
            epochs    = args.epochs,
            steps_per_epoch = len(tr_loader),
            pct_start = 0.3,
            div_factor= 25,
            final_div_factor=1000
        )

        # train
        for ep in range(args.epochs):
            _ , _ = train_epoch(model, tr_loader, optimizer, scheduler)

        # eval
        _, raw_acc, _, _, _ = eval_epoch(model, te_loader, apply_smoothing=False)
        accs.append(raw_acc)

    return float(np.mean(accs))

def do_grid_search():
    best_acc = -1
    best_hp  = None

    for combo in itertools.product(*PARAM_GRID.values()):
        hp = dict(zip(PARAM_GRID.keys(), combo))
        # hash key for results dir
        key = hashlib.md5(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:8]
        outdir = RESULTS_DIR/"grid_search"/key
        outdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Grid hp={hp} → running CV...")
        acc = run_cv(hp)
        logger.info(f"→ got mean accuracy={acc:.4f}")

        with open(outdir/"result.json","w") as f:
            json.dump({"hp":hp,"mean_accuracy":acc}, f, indent=2)

        if acc > best_acc:
            best_acc, best_hp = acc, hp

    logger.info(f"*** BEST HP = {best_hp} with mean_acc={best_acc:.4f} ***")

# ─── Main entrypoint ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    if args.grid_search:
        do_grid_search()
    else:
        # for just a single CV run at --lr/--batch-size/etc.
        hp = {
          "lr":           args.lr,
          "batch_size":   args.batch_size,
          "seq_length":   args.seq_length,
          "seq_stride":   args.seq_stride,
          "num_layers":   3,
          "num_heads":    8,
          "dropout":      0.2
        }
        # reuse the grid‐search function to run exactly one CV:
        acc = run_cv(hp)
        logger.info(f"Single-run mean accuracy = {acc:.4f}")
