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
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# ─── Argument parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data-root",    type=Path, required=True,
                    help="Base folder for processed_sleepedf, c22_processed_sleepedf, psd_features_sleepedf")
parser.add_argument("--results-root", type=Path, required=True,
                    help="Where to write hybrid_model_results")
parser.add_argument("--figures-root", type=Path, required=True,
                    help="Where to save figures")
parser.add_argument("--n-jobs",       type=int, default=16,
                    help="num_workers for DataLoader")
parser.add_argument("--batch-size",   type=int, default=32)
parser.add_argument("--epochs",       type=int, default=50)
parser.add_argument("--lr",           type=float, default=2e-4)
parser.add_argument("--seq-length",   type=int, default=30)
parser.add_argument("--seq-stride",   type=int, default=5)
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
logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────────
BASE               = args.data_root
PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
PSD_DATA_DIR       = BASE/"psd_features_sleepedf"
RESULTS_DIR        = args.results_root
FIGURES_DIR        = args.figures_root

for d in (RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)
for sub in ("plots","models","metrics"):
    (RESULTS_DIR/sub).mkdir(parents=True, exist_ok=True)

# ─── Hyperparameters & seed ─────────────────────────────────────────────────────
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

logger.info(f"BASE directory: {BASE}")
for d in (PROCESSED_DATA_DIR, CATCH22_DATA_DIR, PSD_DATA_DIR):
    logger.info(f"{d}: exists = {d.exists()}")
if not (PROCESSED_DATA_DIR.exists() and
        CATCH22_DATA_DIR.exists() and
        PSD_DATA_DIR.exists()):
    sys.exit("ERROR: Missing one or more data directories")

# ─── Utility ────────────────────────────────────────────────────────────────────
def get_true_subject_id(filename):
    b = Path(filename).stem
    return b[:5] if b.startswith(("SC4","ST7")) else b[:6]

# ─── Dataset ───────────────────────────────────────────────────────────────────
class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, psd_dir, recording_ids=None):
        raw_files = glob.glob(str(raw_dir/"*_sequences.npz"))
        c22_files = glob.glob(str(c22_dir/"*_c22.csv"))
        psd_files = glob.glob(str(psd_dir/"*_psd.npz"))
        if recording_ids is not None:
            raw_files = [p for p in raw_files if any(r in p for r in recording_ids)]
            c22_files = [p for p in c22_files if any(r in p for r in recording_ids)]
            psd_files = [p for p in psd_files if any(r in p for r in recording_ids)]
        self.raw_map = {Path(p).stem.split("_")[0]: p for p in raw_files}
        self.c22_map = {Path(p).stem.split("_")[0]: p for p in c22_files}
        self.psd_map = {Path(p).stem.split("_")[0]: p for p in psd_files}
        common = sorted(set(self.raw_map) & set(self.c22_map) & set(self.psd_map))
        if not common:
            raise ValueError("No overlapping recordings!")
        self.recording_ids = common

        seq_list, c22_list, psd_list, lbl_list = [], [], [], []
        for rid in common:
            dat          = np.load(self.raw_map[rid])
            seqs, labels = dat["sequences"], dat["seq_labels"]
            feats_c22    = pd.read_csv(self.c22_map[rid]).drop(columns=["label"]).values
            npz_data     = np.load(self.psd_map[rid])
            feats_psd    = npz_data["features"]

            for feats, store in ((feats_c22, c22_list),
                                 (feats_psd, psd_list)):
                n_seq, D = seqs.shape[0], feats.shape[1]
                exp      = n_seq * SEQ_LENGTH
                if feats.shape[0] != exp:
                    newf = np.zeros((n_seq, SEQ_LENGTH, D), np.float32)
                    for i in range(n_seq):
                        s, e = i*SEQ_STRIDE, i*SEQ_STRIDE + SEQ_LENGTH
                        if e <= feats.shape[0]:
                            newf[i] = feats[s:e]
                        else:
                            a = feats.shape[0] - s
                            newf[i,:a] = feats[s:]
                            newf[i,a:] = feats[-1]
                    feats = newf
                else:
                    feats = feats.reshape(n_seq, SEQ_LENGTH, -1).astype(np.float32)
                store.append(feats)

            seq_list.append(seqs.astype(np.float32))
            lbl_list.append(labels.astype(np.int64))

        self.sequences  = torch.from_numpy(np.concatenate(seq_list, axis=0))
        self.c22_feats  = torch.from_numpy(np.concatenate(c22_list, axis=0))
        self.psd_feats  = torch.from_numpy(np.concatenate(psd_list, axis=0))
        self.seq_labels = torch.from_numpy(np.concatenate(lbl_list, axis=0))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.c22_feats[idx],
            self.psd_feats[idx],
            self.seq_labels[idx]
        )

# ─── Model + Loss + Helpers ─────────────────────────────────────────────────────
# (EpochEncoder, C22Encoder, PSDEncoder, HybridSleepTransformer same as before)

def focal_loss(inputs, targets,
               alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    # <-- here we accept either (B,S,C) or flatten from eval_epoch
    if inputs.ndim == 3:
        B,S,C = inputs.shape
        logits = inputs.reshape(-1, C)
    else:
        # e.g. if somebody passes (N,C)
        logits = inputs.reshape(-1, inputs.shape[-1])
    tgt = targets.reshape(-1)

    logp = F.log_softmax(logits, dim=1)
    p    = torch.exp(logp).clamp(min=1e-7)
    ce   = F.nll_loss(logp, tgt, reduction='none')
    pt   = p.gather(1, tgt.unsqueeze(1)).squeeze(1)

    n1   = (tgt==1).float()
    n3   = (tgt==3).float()
    rem  = (tgt==4).float()
    alpha = alpha_general*(1-n1-n3-rem) + alpha_n1*n1 + 0.5*n3 + 0.45*rem

    return (alpha * ((1-pt)**gamma) * ce).mean()

# (mixup_batch, train_epoch, eval_epoch, eval_epoch_probs unchanged)

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    subj_map = {}
    for f in glob.glob(str(PROCESSED_DATA_DIR/"*_sequences.npz")):
        rid = Path(f).stem.split("_")[0]
        subj_map.setdefault(get_true_subject_id(rid), []).append(rid)

    subjects = list(subj_map.keys())
    np.random.seed(SEED)
    np.random.shuffle(subjects)
    folds = np.array_split(subjects, 5)

    all_conf = np.zeros((5,5), dtype=int)
    all_probs, all_lbls = [], []

    for k in range(5):
        train_subs = [s for i,f in enumerate(folds) if i!=k for s in f]
        test_subs  = folds[k]
        train_ids  = [rid for s in train_subs for rid in subj_map[s]]
        test_ids   = [rid for s in test_subs  for rid in subj_map[s]]

        train_ds = HybridSleepDataset(PROCESSED_DATA_DIR,
                                      CATCH22_DATA_DIR,
                                      PSD_DATA_DIR,
                                      train_ids)
        test_ds  = HybridSleepDataset(PROCESSED_DATA_DIR,
                                      CATCH22_DATA_DIR,
                                      PSD_DATA_DIR,
                                      test_ids)

        # sampler setup omitted for brevity...

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=NUM_WORKERS,
                                  pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS,
                                  pin_memory=True)

        # model, optimizer, scheduler setup...

        train_losses, val_losses = [], []
        for ep in range(NUM_EPOCHS):
            tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler)
            vl_loss, raw_acc, smooth_acc, _, _ = eval_epoch(
                model, test_loader, apply_smoothing=True
            )
            train_losses.append(tr_loss)
            val_losses.append(vl_loss)
            logger.info(
                f"Fold {k+1} Epoch {ep+1}/{NUM_EPOCHS}  "
                f"train_loss={tr_loss:.4f} val_loss={vl_loss:.4f} "
                f"raw_acc={raw_acc:.4f} smooth_acc={smooth_acc:.4f}"
            )

        # plotting and final eval...
        # (exactly as before)

    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    main()
