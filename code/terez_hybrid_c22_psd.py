#!/usr/bin/env python3
import os
import sys
import glob
import logging
from pathlib import Path
from datetime import datetime

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

# Logging setup
SCRIPT_DIR = Path(__file__).parent
LOGFILE = SCRIPT_DIR/"terez_hybrid_c22_psd.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Paths
HOME = Path.home()
BASE = Path(os.environ.get(
    "DATA_BASE",
    HOME/"sleep"/"STAT-4830-GOALZ-project"/"data"
))
PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
PSD_DATA_DIR       = BASE/"features_psd_sleep_edf"
RESULTS_DIR        = BASE/"hybrid_model_results"
FIGURES_DIR        = Path("/users/okalova/sleep/sleep_staging/figures")

for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
for sub in ("plots","models","metrics"):
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

# Hyperparameters
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 2e-4
TRAIN_RATIO   = 0.8
SEQ_LENGTH    = 30
SEQ_STRIDE    = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"BASE directory: {BASE}")
for d in (PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR):
    logger.info(f"{d}: exists = {d.exists()}")
if not (PROCESSED_DATA_DIR.exists() and CATCH22_DATA_DIR.exists() and PSD_DATA_DIR.exists()):
    sys.exit("ERROR: Missing data directories")

def get_true_subject_id(filename):
    b = Path(filename).stem
    return b[:5] if b.startswith(("SC4","ST7")) else b[:6]

class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, psd_dir, recording_ids=None):
        raw_files = glob.glob(str(raw_dir/"*_sequences.npz"))
        c22_files = glob.glob(str(c22_dir/"*_c22.csv"))
        psd_files = glob.glob(str(psd_dir/"*_psd.npz"))
        if recording_ids is not None:
            raw_files = [p for p in raw_files if any(r in p for r in recording_ids)]
            c22_files= [p for p in c22_files if any(r in p for r in recording_ids)]
            psd_files= [p for p in psd_files if any(r in p for r in recording_ids)]
        self.raw_map = {Path(p).stem.split("_")[0]: p for p in raw_files}
        self.c22_map = {Path(p).stem.split("_")[0]: p for p in c22_files}
        self.psd_map= {Path(p).stem.split("_")[0]: p for p in psd_files}
        common = sorted(set(self.raw_map)&set(self.c22_map)&set(self.psd_map))
        if not common:
            raise ValueError("No overlapping recordings!")
        self.recording_ids = common

        seq_list, c22_list, psd_list, lbl_list = [], [], [], []
        for rid in common:
            dat = np.load(self.raw_map[rid])
            seqs, labels = dat["sequences"], dat["seq_labels"]
            df = pd.read_csv(self.c22_map[rid])
            feats_c22 = df.drop(columns=["label"]).values
            dat2 = np.load(self.psd_map[rid])
            feats_psd = dat2["psd_features"]

            for feats, store in ((feats_c22, c22_list), (feats_psd, psd_list)):
                n_seq, D = seqs.shape[0], feats.shape[1]
                exp = n_seq * SEQ_LENGTH
                if feats.shape[0] != exp:
                    newf = np.zeros((n_seq, SEQ_LENGTH, D), np.float32)
                    for i in range(n_seq):
                        s, e = i*SEQ_STRIDE, i*SEQ_STRIDE+SEQ_LENGTH
                        if e <= feats.shape[0]:
                            newf[i] = feats[s:e]
                        else:
                            a = feats.shape[0]-s
                            newf[i,:a] = feats[s:]
                            newf[i,a:] = feats[-1]
                    feats = newf
                else:
                    feats = feats.reshape(n_seq, SEQ_LENGTH, -1).astype(np.float32)
                store.append(feats)

            seq_list.append(seqs.astype(np.float32))
            lbl_list.append(labels.astype(np.int64))

        self.sequences  = torch.from_numpy(np.concatenate(seq_list,axis=0))
        self.c22_feats  = torch.from_numpy(np.concatenate(c22_list,axis=0))
        self.psd_feats  = torch.from_numpy(np.concatenate(psd_list,axis=0))
        self.seq_labels = torch.from_numpy(np.concatenate(lbl_list,axis=0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (self.sequences[idx],
                self.c22_feats[idx],
                self.psd_feats[idx],
                self.seq_labels[idx])

class EpochEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2,32,3,padding=1); self.bn1=nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,3,padding=1); self.bn2=nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3,padding=1);self.bn3=nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,128,3,padding=1);self.bn4=nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.attn  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128,32,1), nn.ReLU(),
            nn.Conv1d(32,128,1), nn.Sigmoid()
        )
        self.apool = nn.AdaptiveAvgPool1d(48)
        self.fc    = nn.Linear(128*48, emb)
        self.ln    = nn.LayerNorm(emb)
        self.drop  = nn.Dropout(0.2)

    def forward(self, x):
        B,S,C,T = x.shape
        x = x.view(B*S,C,T)
        for conv,bn in ((self.conv1,self.bn1),(self.conv2,self.bn2),(self.conv3,self.bn3)):
            x = F.relu(bn(conv(x))); x=self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = x * self.attn(x)
        x = self.apool(x).view(B*S,-1)
        x = self.drop(F.relu(self.fc(x))); x=self.ln(x)
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self, inp, emb=64):
        super().__init__()
        self.ln0=nn.LayerNorm(inp)
        self.fc1=nn.Linear(inp,256); self.ln1=nn.LayerNorm(256)
        self.fc2=nn.Linear(256,128); self.ln2=nn.LayerNorm(128)
        self.fc3=nn.Linear(128,emb);  self.ln3=nn.LayerNorm(emb)
        self.drop=nn.Dropout(0.2)

    def forward(self,x):
        B,S,D = x.shape
        x = x.view(B*S,D)
        x = self.drop(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x = self.drop(F.relu(self.ln2(self.fc2(x))))
        x = self.drop(F.relu(self.ln3(self.fc3(x))))
        return x.view(B,S,-1)

class PSDEncoder(nn.Module):
    def __init__(self, inp, emb=64):
        super().__init__()
        self.ln0=nn.LayerNorm(inp)
        self.fc1=nn.Linear(inp,128); self.ln1=nn.LayerNorm(128)
        self.fc2=nn.Linear(128,emb);  self.ln2=nn.LayerNorm(emb)
        self.drop=nn.Dropout(0.2)

    def forward(self,x):
        B,S,D = x.shape
        x = x.view(B*S,D)
        x = self.drop(F.relu(self.ln1(self.fc1(self.ln0(x)))))
        x = self.drop(F.relu(self.ln2(self.fc2(x))))
        return x.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, psd_dim,
                 raw_emb=128, c22_emb=64, psd_emb=64,
                 num_classes=5, num_layers=3, num_heads=8,
                 dropout=0.2, seq_length=SEQ_LENGTH):
        super().__init__()
        self.epoch_enc = EpochEncoder(raw_emb)
        self.c22_enc   = C22Encoder(c22_dim, c22_emb)
        self.psd_enc   = PSDEncoder(psd_dim, psd_emb)
        self.combined_dim = raw_emb + c22_emb + psd_emb
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim,self.combined_dim),
            nn.LayerNorm(self.combined_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.combined_dim,self.combined_dim)
        )
        self.ln_fuse = nn.LayerNorm(self.combined_dim)
        self.pos_enc = nn.Parameter(torch.randn(1,seq_length,self.combined_dim))
        self.cls_tok = nn.Parameter(torch.randn(1,num_classes,self.combined_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_dim, nhead=num_heads,
            dim_feedforward=8*self.combined_dim,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.aux_raw = nn.Sequential(
            nn.Linear(raw_emb,128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,num_classes)
        )
        self.aux_c22 = nn.Sequential(
            nn.Linear(c22_emb,128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,num_classes)
        )
        self.aux_psd = nn.Sequential(
            nn.Linear(psd_emb,128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,num_classes)
        )
        self.fc_shared = nn.Linear(self.combined_dim,256)
        self.ln_shared = nn.LayerNorm(256)
        self.drop      = nn.Dropout(dropout)
        self.fc_classes= nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self, raw, c22, psd):
        r = self.epoch_enc(raw)
        c = self.c22_enc(c22)
        p = self.psd_enc(psd)
        B,S,_ = r.shape
        if c.size(1)!=S or p.size(1)!=S:
            Smin = min(r.size(1),c.size(1),p.size(1))
            r, c, p = r[:,:Smin], c[:,:Smin], p[:,:Smin]
            S = Smin
        ar = self.aux_raw(r)
        ac = self.aux_c22(c)
        ap = self.aux_psd(p)
        x = torch.cat([r,c,p],dim=2).view(B*S,-1)
        x = self.fusion(x); x = F.relu(self.ln_fuse(x))
        x = x.view(B,S,-1) + self.pos_enc[:,:S]
        ct = self.cls_tok.expand(B,-1,-1)
        x = torch.cat([x, ct],dim=1)
        x = self.transformer(x)
        main = x[:,:S]
        return main, ar, ac, ap

def focal_loss(inputs, targets,
               alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    B,S,C = inputs.shape
    logits = inputs.view(-1,C)
    tgt    = targets.view(-1)
    logp   = F.log_softmax(logits,dim=1)
    p      = torch.exp(logp).clamp(min=1e-7)
    ce     = F.nll_loss(logp,tgt,reduction='none')
    pt     = p.gather(1,tgt.unsqueeze(1)).squeeze(1)
    n1 = (tgt==1).float(); n3=(tgt==3).float(); rem=(tgt==4).float()
    alpha = alpha_general*(1-n1-n3-rem) + alpha_n1*n1 + 0.5*n3 + 0.45*rem
    return (alpha * ((1-pt)**gamma) * ce).mean()

def mixup_batch(raw,c22,psd,labels,alpha=0.2):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1
    idx = torch.randperm(raw.size(0)).to(raw.device)
    return (lam*raw + (1-lam)*raw[idx],
            lam*c22+ (1-lam)*c22[idx],
            lam*psd+ (1-lam)*psd[idx],
            labels, labels[idx], lam)

def train_epoch(model, loader, optimizer, scheduler=None, mixup_alpha=0.2):
    model.train()
    running_loss=0; correct=0; total=0
    for raw,c22,psd,labels in loader:
        raw = torch.nan_to_num(raw).to(device, non_blocking=True)
        c22 = torch.nan_to_num(c22).to(device, non_blocking=True)
        psd = torch.nan_to_num(psd).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if np.random.rand()<0.5:
            r,c,p,la,lb,lam = mixup_batch(raw,c22,psd,labels,mixup_alpha)
            use_mixup=True
        else:
            r,c,p = raw,c22,psd; la=labels; use_mixup=False
        optimizer.zero_grad()
        main, ar, ac, ap = model(r,c,p)
        main_loss = focal_loss(main, la)
        if use_mixup:
            lr = lam*focal_loss(ar,la)+(1-lam)*focal_loss(ar,lb)
            lc = lam*focal_loss(ac,la)+(1-lam)*focal_loss(ac,lb)
            lp = lam*focal_loss(ap,la)+(1-lam)*focal_loss(ap,lb)
        else:
            lr = focal_loss(ar,la); lc = focal_loss(ac,la); lp = focal_loss(ap,la)
        loss = main_loss + 0.3*(lr+lc+lp)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if scheduler is not None and isinstance(scheduler,(
            optim.lr_scheduler.OneCycleLR, optim.lr_scheduler.CyclicLR
        )):
            scheduler.step()
        running_loss += loss.item()*raw.size(0)
        preds = main.argmax(dim=-1)
        if not use_mixup:
            correct += (preds==labels).sum().item()
            total += labels.numel()
    return running_loss/len(loader.dataset), correct/total if total>0 else 0

def eval_epoch_probs(model, loader):
    model.eval()
    probs_list, labels_list = [], []
    with torch.no_grad():
        for raw,c22,psd,labels in loader:
            raw = raw.to(device, non_blocking=True)
            c22 = c22.to(device, non_blocking=True)
            psd = psd.to(device, non_blocking=True)
            outputs = model(raw,c22,psd)[0]
            probs = F.softmax(outputs, dim=-1).cpu().numpy().reshape(-1,5)
            labs  = labels.cpu().numpy().reshape(-1)
            probs_list.append(probs); labels_list.append(labs)
    return np.concatenate(probs_list,axis=0), np.concatenate(labels_list,axis=0)

def main():
    subj_map = {}
    for f in glob.glob(str(PROCESSED_DATA_DIR/"*_sequences.npz")):
        rid = Path(f).stem.split("_")[0]
        subj = get_true_subject_id(rid)
        subj_map.setdefault(subj, []).append(rid)
    subjects = list(subj_map.keys())
    np.random.seed(SEED); np.random.shuffle(subjects)
    folds = np.array_split(subjects,5)

    all_conf = np.zeros((5,5),dtype=int)
    all_probs = []
    all_lbls  = []

    for k in range(5):
        train_subs = [s for i,f in enumerate(folds) if i!=k for s in f]
        test_subs  =    folds[k]
        train_ids = [rid for s in train_subs for rid in subj_map[s]]
        test_ids  = [rid for s in test_subs  for rid in subj_map[s]]

        train_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR, train_ids)
        test_ds  = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR, test_ids)

        flat_lbl = train_ds.seq_labels.view(-1).numpy()
        counts   = np.bincount(flat_lbl, minlength=5)
        weights  = 1.0/np.sqrt(counts+1e-6); weights[1]*=1.5
        seq_w = np.zeros(len(train_ds))
        for i in range(len(train_ds)):
            lw = train_ds.seq_labels[i].numpy()
            seq_w[i] = np.mean(weights[lw])
        sampler = WeightedRandomSampler(seq_w, len(seq_w), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=16, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=16, pin_memory=True)

        c22_dim = train_ds.c22_feats.shape[-1]
        psd_dim = train_ds.psd_feats.shape[-1]
        model = HybridSleepTransformer(c22_dim, psd_dim).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE,
            epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
            pct_start=0.3, div_factor=25, final_div_factor=1000
        )

        train_losses = []; val_losses = []
        # Early stopping disabled: do not break out early
        for ep in range(NUM_EPOCHS):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
            vl_loss, _, _, _, _ = eval_epoch_probs(model, test_loader)
            train_losses.append(tr_loss); val_losses.append(vl_loss)
            logger.info(f"Fold {k+1} Epoch {ep+1}/{NUM_EPOCHS}  train_loss={tr_loss:.4f}")

        # Convergence plot for this fold
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses,   label="Val PSD Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"Convergence (Fold {k+1})")
        plt.legend()
        plt.savefig(FIGURES_DIR/f"convergence_fold{k+1}.png")
        plt.clf()

        # Final eval & metrics
        probs, lbls = eval_epoch_probs(model, test_loader)
        preds = probs.argmax(axis=1)
        cm = confusion_matrix(lbls, preds)
        all_conf += cm
        all_probs.append(probs)
        all_lbls.append(lbls)

        report = classification_report(lbls, preds, target_names=["W","N1","N2","N3","REM"])
        logger.info(f"\nFold {k+1} classification report:\n{report}")

    # Confusion matrix plot (all folds)
    plt.matshow(all_conf)
    plt.colorbar()
    plt.xticks(range(5), ["W","N1","N2","N3","REM"])
    plt.yticks(range(5), ["W","N1","N2","N3","REM"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Cross-Validation Confusion Matrix")
    plt.savefig(FIGURES_DIR/"confusion_matrix.png")
    plt.clf()

    # ROC AUC (all folds)
    all_probs_arr = np.concatenate(all_probs, axis=0)
    all_lbls_arr  = np.concatenate(all_lbls,  axis=0)
    bin_lbls = label_binarize(all_lbls_arr, classes=[0,1,2,3,4])
    for i,cls in enumerate(["W","N1","N2","N3","REM"]):
        fpr, tpr, _ = roc_curve(bin_lbls[:,i], all_probs_arr[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (All Folds)")
    plt.legend(loc="lower right")
    plt.savefig(FIGURES_DIR/"roc_auc.png")
    plt.clf()

    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    main()
