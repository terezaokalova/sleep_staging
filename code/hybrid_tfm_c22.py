#!/usr/bin/env python3
import os
import sys
import glob
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# ==== Initialization ====
# -------------------------
HOME = Path.home()
BASE = HOME / "sleep" / "STAT-4830-GOALZ-project" / "data"
PROCESSED_DATA_DIR = BASE / "processed_sleepedf"
CATCH22_DATA_DIR   = BASE / "c22_processed_sleepedf"
RESULTS_DIR        = BASE / "hybrid_model_results"

print("Checking directory access:")
print(f"  {PROCESSED_DATA_DIR!s} exists: {PROCESSED_DATA_DIR.exists()}")
print(f"  {CATCH22_DATA_DIR!s} exists: {CATCH22_DATA_DIR.exists()}")

if not PROCESSED_DATA_DIR.exists() or not CATCH22_DATA_DIR.exists():
    print("ERROR: Data directories not found. Please re-run preprocessing & Catch22 steps.")
    sys.exit(1)

# Create results subdirs
for sub in ["plots","models","metrics"]:
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
BATCH_SIZE    = 32
NUM_EPOCHS    = 35
LEARNING_RATE = 1e-5
TRAIN_RATIO   = 0.8
SEQ_LENGTH    = 20
SEQ_STRIDE    = 10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------
# ==== Utilities  =========
# -------------------------
def get_true_subject_id(filename):
    basename = Path(filename).stem
    if basename.startswith(("SC4","ST7")):
        return basename[:5]
    return basename[:6]

def group_by_true_subjects(data_dir):
    files = glob.glob(str(data_dir/"*_sequences.npz"))
    subj_map = {}
    for f in files:
        rid = Path(f).stem.split("_")[0]
        subj = get_true_subject_id(rid)
        subj_map.setdefault(subj, []).append(rid)
    return subj_map

def split_true_subjects(data_dir, train_ratio=TRAIN_RATIO, random_state=SEED):
    subj_map = group_by_true_subjects(data_dir)
    subs = list(subj_map.keys())
    np.random.seed(random_state)
    np.random.shuffle(subs)
    n_train = int(len(subs)*train_ratio)
    train_subs = subs[:n_train]
    test_subs  = subs[n_train:]
    train_ids = [rid for s in train_subs for rid in subj_map[s]]
    test_ids  = [rid for s in test_subs  for rid in subj_map[s]]
    return train_ids, test_ids, train_subs, test_subs

# -------------------------
# ==== Dataset    =========
# -------------------------
class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, recording_ids=None):
        all_raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        all_c22 = glob.glob(str(c22_dir  /"*_c22.csv"))
        if recording_ids is not None:
            all_raw = [p for p in all_raw if any(rid in p for rid in recording_ids)]
            all_c22 = [p for p in all_c22 if any(rid in p for rid in recording_ids)]
        self.raw_map = {Path(p).stem.split("_")[0]: p for p in all_raw}
        self.c22_map = {Path(p).stem.split("_")[0]: p for p in all_c22}
        common = sorted(set(self.raw_map) & set(self.c22_map))
        if not common:
            raise ValueError("No overlapping recordings between raw & Catch22!")
        self.recording_ids = common

        seq_list, c22_list, lbl_list = [], [], []
        for rid in common:
            # load raw
            data = np.load(self.raw_map[rid])
            seqs, labels = data["sequences"], data["seq_labels"]
            # load catch22
            df = pd.read_csv(self.c22_map[rid])
            feats = df.drop(columns=["label"]).values
            # if mismatch, pad/truncate
            n_seq = seqs.shape[0]
            expected = n_seq * SEQ_LENGTH
            if feats.shape[0] != expected:
                # reconstruct windows
                feat_dim = feats.shape[1]
                newf = np.zeros((n_seq, SEQ_LENGTH, feat_dim), dtype=np.float32)
                for i in range(n_seq):
                    start = i*SEQ_STRIDE
                    end   = start+SEQ_LENGTH
                    if end <= feats.shape[0]:
                        newf[i] = feats[start:end]
                    else:
                        avail = feats.shape[0]-start
                        if avail>0:
                            newf[i,:avail] = feats[start:]
                            newf[i,avail:] = feats[start+avail-1]
                        else:
                            newf[i] = newf[i-1]
                feats = newf
            else:
                feats = feats.reshape(n_seq, SEQ_LENGTH, -1).astype(np.float32)
            seq_list.append(seqs.astype(np.float32))
            c22_list.append(feats)
            lbl_list.append(labels.astype(np.int64))

        self.sequences   = torch.from_numpy(np.concatenate(seq_list,axis=0))
        self.c22_feats   = torch.from_numpy(np.concatenate(c22_list,axis=0))
        self.seq_labels  = torch.from_numpy(np.concatenate(lbl_list,axis=0))

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.c22_feats[idx], self.seq_labels[idx]

# -------------------------
# ==== Model      =========
# -------------------------
class EpochEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2,16,kernel_size=5,padding=2)
        self.conv2 = nn.Conv1d(16,32,kernel_size=3,padding=1)
        self.conv3 = nn.Conv1d(32,64,kernel_size=3,padding=1)
        self.pool  = nn.MaxPool1d(2)
        self.fc     = nn.Linear(64*375, embedding_dim)
        self.ln     = nn.LayerNorm(embedding_dim)
        self.dropout= nn.Dropout(0.1)
    def forward(self, x):
        B,S,C,T = x.shape
        x = x.view(B*S, C, T)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(B*S, -1)
        x = self.dropout(F.relu(self.fc(x)))
        x = self.ln(x)
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.ln0   = nn.LayerNorm(input_dim)
        self.fc1   = nn.Linear(input_dim,128)
        self.ln1   = nn.LayerNorm(128)
        self.fc2   = nn.Linear(128, embedding_dim)
        self.ln2   = nn.LayerNorm(embedding_dim)
        self.dropout= nn.Dropout(0.1)
    def forward(self, x):
        B,S,D = x.shape
        x = x.view(B*S, D)
        x = self.dropout(F.relu(self.ln0(x)))
        x = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        return x.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_emb=128, c22_emb=64, num_classes=5,
                 num_layers=2, num_heads=4, dropout=0.1, seq_length=20):
        super().__init__()
        self.epoch_enc = EpochEncoder(raw_emb)
        self.c22_enc   = C22Encoder(c22_dim, c22_emb)
        self.combined_dim = raw_emb + c22_emb
        self.fusion      = nn.Linear(self.combined_dim, self.combined_dim)
        self.ln_fusion   = nn.LayerNorm(self.combined_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1,seq_length,self.combined_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_dim,
            nhead=num_heads,
            dim_feedforward=4*self.combined_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out      = nn.Linear(self.combined_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)

    def forward(self, raw, c22):
        r = self.epoch_enc(raw)
        c = self.c22_enc(c22)
        x = torch.cat([r,c], dim=2)                # (B, S, D)
        B,S,D = x.shape
        x = x.view(B*S, D)
        x = F.relu(self.ln_fusion(self.fusion(x)))
        x = x.view(B,S,D) + self.pos_encoder       # add positional
        x = self.transformer(x)
        return self.fc_out(x)

# -------------------------
# ==== Loss       =========
# -------------------------
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    """
    Stable focal loss: uses log_softmax + clamp.
    """
    B, S, C = inputs.shape
    logits = inputs.view(-1, C)
    tgt    = targets.view(-1)
    logp   = F.log_softmax(logits, dim=1)
    p      = logp.exp().clamp(min=1e-7)
    at     = torch.where(tgt==1,
                         alpha+0.5,    # boost weight for N1
                         alpha)
    ce     = F.nll_loss(logp, tgt, reduction='none')
    fl     = at * ((1-p)**gamma) * ce
    return fl.mean()

# -------------------------
# ==== Training Loop ======
# -------------------------
def train_epoch(model, loader, optimizer):
    model.train()
    running_loss = 0.0
    for raw, c22, labels in loader:
        # sanitize inputs
        raw    = torch.nan_to_num(raw,    nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        c22    = torch.nan_to_num(c22,    nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(raw, c22)
        if not torch.isfinite(logits).all():
            print("Skipping batch: non-finite logits")
            continue

        loss = focal_loss(logits, labels)
        if torch.isnan(loss):
            print("Skipping batch: NaN loss")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        running_loss += loss.item() * raw.size(0)

    return running_loss / len(loader.dataset)

def eval_epoch(model, loader):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for raw, c22, labels in loader:
            raw    = torch.nan_to_num(raw, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
            c22    = torch.nan_to_num(c22, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
            labels = labels.to(device)

            logits = model(raw, c22)
            if not torch.isfinite(logits).all():
                continue

            loss = focal_loss(logits, labels)
            running_loss += loss.item() * raw.size(0)

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if not all_preds:
        return float('nan'), 0.0, [], []
    all_preds = np.concatenate(all_preds).ravel()
    all_labels= np.concatenate(all_labels).ravel()
    acc = (all_preds==all_labels).mean()
    return running_loss/len(loader.dataset), acc, all_preds, all_labels

# -------------------------
# ==== Main       =========
# -------------------------
def main():
    torch.autograd.set_detect_anomaly(True)

    # prepare CV splits
    subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
    subjects = list(subj_map.keys())
    np.random.seed(SEED)
    np.random.shuffle(subjects)
    folds = np.array_split(subjects, 5)

    fold_results = []
    for k in range(5):
        test_subs  = folds[k]
        train_subs = [s for i, f in enumerate(folds) if i!=k for s in f]
        train_ids  = [rid for s in train_subs for rid in subj_map[s]]
        test_ids   = [rid for s in test_subs  for rid in subj_map[s]]

        train_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, train_ids)
        test_ds  = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, test_ids)

        # weighted sampler
        labels = train_ds.seq_labels[:,0].numpy()
        class_counts = np.bincount(labels, minlength=5)
        class_weights= 1.0 / (class_counts + 1e-6)
        sample_weights= class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4, pin_memory=True)

        # model & optimizer
        c22_dim = train_ds.c22_feats.shape[-1]
        model   = HybridSleepTransformer(c22_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                         mode='min', factor=0.5, patience=3, verbose=True)

        best_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            tloss = train_epoch(model, train_loader, optimizer)
            vloss, vacc, vp, vl = eval_epoch(model, test_loader)
            if np.isfinite(vloss):
                scheduler.step(vloss)
            print(f"Fold{k+1} E{epoch+1}/{NUM_EPOCHS} TL={tloss:.4f} VL={vloss:.4f} VA={vacc:.4f}")
            if np.isfinite(vloss) and vloss < best_loss:
                best_loss = vloss
                torch.save(model.state_dict(),
                           RESULTS_DIR/f"models/best_fold{k+1}.pth")

        # final eval on best model
        model.load_state_dict(torch.load(RESULTS_DIR/f"models/best_fold{k+1}.pth"))
        _, vacc, vp, vl = eval_epoch(model, test_loader)
        print(f"==> Fold{k+1} final accuracy: {vacc:.4f}")
        print(classification_report(vl, vp, target_names=["W","N1","N2","N3","REM"]))
        fold_results.append(vacc)

    print("=== CV Summary ===")
    print("Accuracies:", fold_results)
    print("Mean  :", np.mean(fold_results))
    print("Std   :", np.std(fold_results))

if __name__ == "__main__":
    main()
