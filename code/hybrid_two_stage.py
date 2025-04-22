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
from torchcrf import CRF

# paths
BASE = Path("/users/okalova/sleep/STAT-4830-GOALZ-project/data")
PROCESSED_DIR = BASE/"processed_sleepedf"
CATCH22_DIR   = BASE/"c22_processed_sleepedf"
RESULTS_DIR   = BASE/"hybrid_two_stage_crf"
for d in ("models","metrics"): (RESULTS_DIR/d).mkdir(parents=True, exist_ok=True)

# hyperparams
SEED       = 42
BATCH_SIZE = 32
NUM_EPOCHS = 35
LR1        = 1e-4    # stage1
LR2        = 2e-4    # stage2
SEQ_LENGTH = 30
SEQ_STRIDE = 5
N_CLASSES  = 5
DESIRED_N1 = 0.2

# focal‐loss defaults
ALPHA_GEN = 0.25
ALPHA_N1  = 1.0
GAMMA      = 3.0

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage 1: N1 vs not 
class Stage1Dataset(Dataset):
    def __init__(self, raw_dir):
        # find all of your .npz sequence files
        files = glob.glob(str(raw_dir/"*_sequences.npz"))
        seqs, labs = [], []

        for f in files:
            d = np.load(f)
            seq_arr = d["sequences"]    # could be (S, C, T) or (Nseg, S, C, T)
            lab_arr = d["seq_labels"]   # could be (S,) or (Nseg, S)

            # if it's a 4‑D array, flatten segments × windows → individual windows
            if seq_arr.ndim == 4:
                Nseg, S, C, T = seq_arr.shape
                seq_arr = seq_arr.reshape(Nseg * S, C, T)

            # same for labels
            if lab_arr.ndim == 2:
                Nseg, S = lab_arr.shape
                lab_arr = lab_arr.reshape(Nseg * S)

            seqs.append(seq_arr)
            # binarize to N1 vs not
            labs.append((lab_arr == 1).astype(np.int64))

        # now concatenate all windows into one big tensor of shape (M, C, T)
        self.X = torch.from_numpy(np.concatenate(seqs, axis=0)).float()

        # same for labels → (M,)
        lab_arr = np.concatenate(labs, axis=0)
        self.Y = torch.from_numpy(lab_arr).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

# class Stage1Dataset(Dataset):
#     def __init__(self, raw_dir):
#         files = glob.glob(str(raw_dir/"*_sequences.npz"))
#         seqs, labs = [], []
#         for f in files:
#             d = np.load(f)
#             seqs.append(d["sequences"])
#             labs.append((d["seq_labels"] == 1).astype(np.int64))
#         # concatenate all windows
#         self.X = torch.from_numpy(
#             np.concatenate(seqs, axis=0)
#         ).float()  # (M, C, T)

#         lab_arr = np.concatenate(labs, axis=0)
#         # if for some reason seq_labels came in as (M,window_len), pick center
#         if lab_arr.ndim == 2:
#             center = lab_arr.shape[1] // 2
#             lab_arr = lab_arr[:, center]
#         # now guaranteed (M,)
#         self.Y = torch.from_numpy(lab_arr).long()

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, i):
#         return self.X[i], self.Y[i]

class Stage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2,16,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64,2)

    def forward(self, x):
        # x: (batch, channels, time)
        h = self.conv(x).squeeze(-1)   # -> (batch, 64)
        return self.fc(h)              # -> (batch, 2)

def train_stage1(detector, loader, opt):
    detector.train()
    total_loss = 0.0
    total = 0
    for x,y in loader:
        x, y = x.to(device), y.to(device)       # x: (B, C, T), y: (B,)
        opt.zero_grad()
        logits = detector(x)                    # (B, 2)
        loss   = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        total      += x.size(0)
    return total_loss/total

def eval_stage1(detector, loader):
    detector.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for x,y in loader:
            x, y   = x.to(device), y.to(device)
            logits = detector(x)                # (B, 2)
            preds  = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total   += y.size(0)
    return correct/total

#  Hybrid 5‑way dataset 
def get_true_subject_id(fn):
    b = Path(fn).stem
    return b[:5] if b.startswith(("SC4","ST7")) else b[:6]

class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, recording_ids):
        all_raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        all_c22 = glob.glob(str(c22_dir/"*_c22.csv"))
        if recording_ids is not None:
            all_raw = [p for p in all_raw if any(rid in p for rid in recording_ids)]
            all_c22 = [p for p in all_c22 if any(rid in p for rid in recording_ids)]
        raw_map = {Path(p).stem.split("_")[0]:p for p in all_raw}
        c22_map = {Path(p).stem.split("_")[0]:p for p in all_c22}
        ids = sorted(raw_map.keys() & c22_map.keys())
        seqs, c22s, labels = [], [], []
        
        for rid in ids:
            d = np.load(raw_map[rid])
            s, lab = d["sequences"], d["seq_labels"]  # (S,C,T), (S,)
            
            # Reshape s if needed to ensure it's (S, C, T)
            if s.ndim == 4:  # Handle (Nseg, S, C, T) case
                Nseg, S, C, T = s.shape
                s = s.reshape(Nseg * S, C, T)
                lab = lab.reshape(Nseg * S)
            
            df = pd.read_csv(c22_map[rid])
            f = df.drop(columns="label").values
            
            # reshape/pad f→(S,SEQ_LENGTH,feat_dim)
            nS = s.shape[0]
            feat_dim = f.shape[1]
            exp = nS*SEQ_LENGTH
            
            if f.shape[0] != exp:
                newf = np.zeros((nS, SEQ_LENGTH, feat_dim), np.float32)
                for i in range(nS):
                    st = i*SEQ_STRIDE; en = st+SEQ_LENGTH
                    if en <= f.shape[0]:
                        newf[i] = f[st:en]
                    else:
                        av = f.shape[0]-st
                        if av > 0:
                            newf[i, :av] = f[st:]
                            newf[i, av:] = f[-1]
                        else:
                            newf[i] = newf[i-1]
                f = newf
            else:
                f = f.reshape(nS, SEQ_LENGTH, feat_dim).astype(np.float32)
            
            seqs.append(s.astype(np.float32))
            c22s.append(f)
            labels.append(lab.astype(np.int64))
            
        self.raw = torch.from_numpy(np.concatenate(seqs, 0))
        self.c22 = torch.from_numpy(np.concatenate(c22s, 0))
        self.labels = torch.from_numpy(np.concatenate(labels, 0))
        
        # Print shape information for debugging
        print(f"Raw data shape: {self.raw.shape}")
        print(f"C22 data shape: {self.c22.shape}")
        print(f"Labels shape: {self.labels.shape}")
        
    def __len__(self): 
        return len(self.labels)
        
    def __getitem__(self, i):
        # Reshape raw to ensure it's (S, C, T) where S=1 for a single window
        raw_item = self.raw[i].unsqueeze(0) if self.raw[i].ndim == 2 else self.raw[i].unsqueeze(0)
        c22_item = self.c22[i].unsqueeze(0) if self.c22[i].ndim == 2 else self.c22[i]
        label_item = self.labels[i].unsqueeze(0) if self.labels[i].ndim == 0 else self.labels[i]
        
        return raw_item, c22_item, label_item

#  Encoders & Transformer 
class EpochEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        # Replace fixed size with adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=16)  # Adjust this size as needed
        self.fc = nn.Linear(64 * 16, emb)  # 64 channels × 16 output size
        self.ln = nn.LayerNorm(emb)
        
    def forward(self, x):
        B, S, C, T = x.shape
        h = x.view(B*S, C, T)
        h = self.pool(F.relu(self.conv1(h)))
        h = self.pool(F.relu(self.conv2(h)))
        h = self.pool(F.relu(self.conv3(h)))
        # Add adaptive pooling to ensure fixed size for the FC layer
        h = self.adaptive_pool(h)
        # Get actual flattened size
        h = h.view(B*S, -1)
        h = self.ln(self.fc(h))
        return h.view(B, S, -1)

class C22Encoder(nn.Module):
    def __init__(self,in_d,emb=64):
        super().__init__()
        self.ln0 = nn.LayerNorm(in_d)
        self.fc1 = nn.Linear(in_d,128); self.ln1=nn.LayerNorm(128)
        self.fc2 = nn.Linear(128,emb); self.ln2=nn.LayerNorm(emb)
    def forward(self,x):
        B,S,D = x.shape
        h = x.view(B*S,D)
        h = F.relu(self.ln1(self.fc1(self.ln0(h))))
        h = self.ln2(self.fc2(h))
        return h.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_emb=128, c22_emb=64,
                 num_classes=5, num_layers=3, num_heads=8,
                 dropout=0.1, seq_len=SEQ_LENGTH):
        super().__init__()
        self.eenc = EpochEncoder(raw_emb)
        self.cenc = C22Encoder(c22_dim, c22_emb)
        D = raw_emb + c22_emb
        self.fuse = nn.LayerNorm(D)
        self.lin = nn.Linear(D, D)
        
        # Create positional encoding that can handle variable sequence lengths
        self.register_buffer("pos", torch.zeros(1, seq_len, D))
        self._init_positional_encoding(seq_len, D)
        
        ff = 8*D
        layer = nn.TransformerEncoderLayer(d_model=D, nhead=num_heads,
                                           dim_feedforward=ff,
                                           dropout=dropout,
                                           batch_first=True)
        self.tfm = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(D, num_classes)
        self._init_weights()
        
    def _init_positional_encoding(self, seq_len, d_model):
        """Initialize positional encoding parameter"""
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pos_enc = torch.zeros(1, seq_len, d_model)
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        
        self.pos = pos_enc
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.zero_()
                
    def forward(self, raw, c22):
        # Print input shapes for debugging
        print(f"Raw input shape: {raw.shape}")
        print(f"C22 input shape: {c22.shape}")
        
        r = self.eenc(raw)
        c = self.cenc(c22)
        
        # Print encoded shapes for debugging
        print(f"Raw encoded shape: {r.shape}")
        print(f"C22 encoded shape: {c.shape}")
        
        x = torch.cat([r, c], dim=2)
        x = F.relu(self.lin(self.fuse(x)))
        
        # Use only as much of the positional encoding as needed
        seq_len = x.size(1)
        x = x + self.pos[:, :seq_len]
        
        x = self.tfm(x)
        return self.out(x)  # (B, S, 5)

#  Loss & Train/Eval 
def focal_loss(inputs, targets,
               alpha_general=ALPHA_GEN, alpha_n1=ALPHA_N1,
               gamma=GAMMA):
    B,S,C = inputs.shape
    logits = inputs.view(-1,C)
    tgt    = targets.view(-1)
    logp   = F.log_softmax(logits,1)
    p      = torch.exp(logp).clamp(min=1e-7)
    ce     = F.nll_loss(logp,tgt,reduction='none')
    pt     = p.gather(1,tgt.unsqueeze(1)).squeeze(1)
    n1m    = (tgt==1).float()
    alpha  = alpha_general*(1-n1m) + alpha_n1*n1m
    fl     = alpha * ((1-pt)**gamma) * ce
    return fl.mean()

def train_epoch2(model, detector, loader, opt):
    model.train()
    total = 0
    loss_sum = 0.0

    for raw, c22, labels in loader:
        # raw: (B, S, C, T), c22: (B, S, feat_dim), labels: (B, S)
        raw, c22, labels = raw.to(device), c22.to(device), labels.to(device)
        
        # Reset gradients
        opt.zero_grad()
        
        # Reshape raw for detector (which expects (B, C, T))
        B, S, C, T = raw.shape
        raw_reshaped = raw.view(B*S, C, T)
        
        # 1) Stage 1 detector on each window
        with torch.no_grad():
            det_logits = detector(raw_reshaped)      # -> (B*S, 2)
            det_score = det_logits[:, 1].view(B, S)  # -> (B, S)
        
        # Forward through transformer + fusion
        out = model(raw, c22)   # -> (B, S, num_classes)
        
        # Add N1 bias
        out[:, :, 1] = out[:, :, 1] + det_score
        
        # Compute loss - reshape outputs and labels
        out_flat = out.view(-1, N_CLASSES)
        labels_flat = labels.view(-1)
        
        # Use focal loss or cross-entropy
        loss = F.cross_entropy(out_flat, labels_flat)
        
        loss.backward()
        opt.step()

        loss_sum += loss.item() * B
        total += B

    return loss_sum / total


def eval_epoch2(model, detector, loader, crf):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for raw, c22, labels in loader:
            raw, c22, labels = raw.to(device), c22.to(device), labels.to(device)
            
            # Reshape raw for detector
            B, S, C, T = raw.shape
            raw_reshaped = raw.view(B*S, C, T)

            # 1) Stage 1 detector
            det_logits = detector(raw_reshaped)     # -> (B*S, 2)
            det_score = det_logits[:, 1].view(B, S)  # -> (B, S)

            # 2) Transformer
            out = model(raw, c22)  # (B, S, num_classes)
            out[:, :, 1] = out[:, :, 1] + det_score
            
            # Use CRF for sequence modeling
            emissions = out
            
            # Get best sequence path
            predictions = crf.decode(emissions)
            
            # Flatten predictions and labels for metrics
            for pred_seq, label_seq in zip(predictions, labels):
                all_preds.extend(pred_seq)
                all_labels.extend(label_seq.cpu().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return all_preds, all_labels, acc

#  Main 
def main():
    # stage1
    ds1 = Stage1Dataset(PROCESSED_DIR)
    w = np.bincount(ds1.Y.numpy(),minlength=2)
    w1 = (w[0]/w[1])*(DESIRED_N1/(1-DESIRED_N1))
    sw = np.where(ds1.Y.numpy()==1, w1, 1.0)
    loader1 = DataLoader(ds1, BATCH_SIZE,
                         sampler=WeightedRandomSampler(sw,len(sw),True))
    detector = Stage1Detector().to(device)
    opt1 = optim.Adam(detector.parameters(),lr=LR1)
    for ep in range(10):
        l1 = train_stage1(detector,loader1,opt1)
        a1 = eval_stage1(detector,loader1)
        print(f"Det Epoch{ep+1}: loss={l1:.4f} acc={a1:.4f}")
    detector.eval()
    for p in detector.parameters(): p.requires_grad=False

    # prepare CV
    # split by subject
        # prepare CV
    files = glob.glob(str(PROCESSED_DIR/"*_sequences.npz"))
    subj_map = {}
    for f in files:
        rid = Path(f).stem.split("_")[0]
        subj_map.setdefault(get_true_subject_id(rid), []).append(rid)

    subs = list(subj_map.keys())
    np.random.seed(SEED)
    np.random.shuffle(subs)
    folds = np.array_split(subs, 5)

    # crf = CRF(N_CLASSES, batch_first=True)
    crf = CRF(N_CLASSES, batch_first=True).to(device)
    # For Stage 2
    for k in range(2):  # or range(5)
        # which subjects go in train vs test
        test_subs  = folds[k].tolist()
        train_subs = [s for i, f in enumerate(folds) if i != k for s in f]

        # now expand subjects into recording IDs
        train_ids = [rid for s in train_subs for rid in subj_map[s]]
        test_ids  = [rid for s in test_subs  for rid in subj_map[s]]

        ds2_tr = HybridSleepDataset(PROCESSED_DIR, CATCH22_DIR, train_ids)
        ds2_te = HybridSleepDataset(PROCESSED_DIR, CATCH22_DIR, test_ids)
        
        # Print more information about the first batch
        sample_batch = next(iter(DataLoader(ds2_tr, batch_size=4)))
        raw_batch, c22_batch, label_batch = sample_batch
        print(f"Sample raw batch shape: {raw_batch.shape}")
        print(f"Sample c22 batch shape: {c22_batch.shape}")
        print(f"Sample label batch shape: {label_batch.shape}")
        
        # Correct way to get binary N1 indicator from labels
        # Labels now have shape (B, 1)
        seg_has_n1 = (ds2_tr.labels == 1).any(dim=1).cpu().numpy()  
        n1_cnt = seg_has_n1.sum()
        n0_cnt = len(ds2_tr) - n1_cnt
        w_n1 = (n0_cnt / n1_cnt) * (DESIRED_N1 / (1 - DESIRED_N1))
        sw2 = np.where(seg_has_n1, w_n1, 1.0)

        loader2_tr = DataLoader(
            ds2_tr,
            BATCH_SIZE,
            sampler=WeightedRandomSampler(sw2, len(sw2), True),
            num_workers=4,
            pin_memory=True
        )
        loader2_te = DataLoader(
            ds2_te,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        model = HybridSleepTransformer(ds2_tr.c22.size(-1)).to(device)
        opt2 = optim.AdamW(model.parameters(), lr=LR2, weight_decay=1e-4)
        for ep in range(NUM_EPOCHS):
            l2 = train_epoch2(model, detector, loader2_tr, opt2)
            _,_,acc2 = eval_epoch2(model, detector, loader2_te, crf)
            print(f"Fold{k+1} Ep{ep+1}: train_loss={l2:.4f} val_acc={acc2:.4f}")
        # final
        preds, labels, acc2 = eval_epoch2(model, detector, loader2_te, crf)
        print(f"Fold{k+1} final acc={acc2:.4f}")
        print(classification_report(labels,preds,
              target_names=["W","N1","N2","N3","REM"]))
        torch.save(model.state_dict(), RESULTS_DIR/f"models/fold{k+1}.pth")

if __name__=="__main__":
    main()
