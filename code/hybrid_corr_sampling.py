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

from pathlib import Path
# On Pioneer your data live under /users/okalova/…
BASE = Path("/users/okalova/sleep/STAT-4830-GOALZ-project/data")

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
# BATCH_SIZE    = 32
# NUM_EPOCHS    = 35
# LEARNING_RATE = 1e-5
# TRAIN_RATIO   = 0.8
# SEQ_LENGTH    = 20
# SEQ_STRIDE    = 10
BATCH_SIZE    = 32 
NUM_EPOCHS    = 50  
LEARNING_RATE = 2e-4  
TRAIN_RATIO   = 0.8 
SEQ_LENGTH    = 30  # for better temporal context
SEQ_STRIDE    = 5   # for denser temporal sampling

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

        # self.sequences   = torch.from_numpy(np.concatenate(seq_list,axis=0))
        # self.c22_feats   = torch.from_numpy(np.concatenate(c22_list,axis=0))
        # self.seq_labels  = torch.from_numpy(np.concatenate(lbl_list,axis=0))
        self.raw    = torch.from_numpy(np.stack(seqs,   axis=0))  # (N_segments, S, 2, T)
        self.c22    = torch.from_numpy(np.stack(c22s,   axis=0))  # (N_segments, S, feat_dim)
        self.labels = torch.from_numpy(np.stack(labels, axis=0))  # (N_segments, S)


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
        # More filters and smaller kernels for finer feature detection
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Add a fourth convolutional layer for more depth
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        
        # Feature refinement with attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Use adaptive pooling to handle any input size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(48)
        
        # Fixed output size after adaptive pooling
        self.fc = nn.Linear(128 * 48, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Debug prints
        # print(f"EpochEncoder initialized with embedding_dim={embedding_dim}")
    
    def forward(self, x):
        B, S, C, T = x.shape
        # print(f"EpochEncoder input shape: B={B}, S={S}, C={C}, T={T}")
        
        x = x.view(B*S, C, T)
        # print(f"Reshaped to: {x.shape}")
        
        # First conv block with BN and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        # print(f"After first conv block: {x.shape}")
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        # print(f"After second conv block: {x.shape}")
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        # print(f"After third conv block: {x.shape}")
        
        # Fourth conv block without pooling for finer features
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        # print(f"After fourth conv block: {x.shape}")
        
        # Apply channel attention
        attn = self.channel_attn(x)
        x = x * attn
        # print(f"After attention: {x.shape}")
        
        # Use adaptive pooling to get fixed output size
        x = self.adaptive_pool(x)
        # print(f"After adaptive pooling: {x.shape}")
        
        # Flatten and project
        x = x.view(B*S, -1)
        # print(f"After flattening: {x.shape}")
        
        x = self.dropout(F.relu(self.fc(x)))
        x = self.ln(x)
        # print(f"After FC and LN: {x.shape}")
        
        output = x.view(B, S, -1)
        # print(f"EpochEncoder final output shape: {output.shape}")
        
        return output

class C22Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        # print(f"Initializing C22Encoder with input_dim={input_dim}, embedding_dim={embedding_dim}")
        
        # Initial normalization
        self.ln0 = nn.LayerNorm(input_dim)
        
        # Simplified architecture - just use a standard MLP
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, embedding_dim)
        self.ln3 = nn.LayerNorm(embedding_dim)
        
        # Increased dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        B, S, D = x.shape
        # print(f"C22Encoder input shape: B={B}, S={S}, D={D}")
        
        x_flat = x.view(B*S, D)
        # print(f"x_flat shape: {x_flat.shape}")
        
        # Main encoder path
        norm_x = self.ln0(x_flat)
        x1 = self.dropout(F.relu(self.ln1(self.fc1(norm_x))))
        x2 = self.dropout(F.relu(self.ln2(self.fc2(x1))))
        x3 = self.dropout(F.relu(self.ln3(self.fc3(x2))))
        
        # print(f"C22Encoder output shape before reshape: {x3.shape}")
        output = x3.view(B, S, -1)
        # print(f"C22Encoder final output shape: {output.shape}")
        
        return output

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_emb=128, c22_emb=64, num_classes=5,
                 num_layers=3, num_heads=8, dropout=0.2, seq_length=30):
        super().__init__()
        # print(f"Initializing HybridSleepTransformer with seq_length={seq_length}")
        
        self.seq_length = seq_length  # Store for reference
        self.epoch_enc = EpochEncoder(raw_emb)
        self.c22_enc = C22Encoder(c22_dim, c22_emb)
        self.combined_dim = raw_emb + c22_emb
        
        # Enhanced fusion mechanism
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, self.combined_dim),
            nn.LayerNorm(self.combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.combined_dim, self.combined_dim)
        )
        self.ln_fusion = nn.LayerNorm(self.combined_dim)
        
        # Positional encoding with learnable parameters
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, self.combined_dim))
        
        # Class token approach (similar to BERT's CLS token)
        self.class_tokens = nn.Parameter(torch.randn(1, num_classes, self.combined_dim))
        
        # Enhanced transformer with more layers, heads and larger feedforward
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_dim,
            nhead=num_heads,
            dim_feedforward=8*self.combined_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        
        # Auxiliary classifiers for intermediate supervision
        self.aux_raw_classifier = nn.Sequential(
            nn.Linear(raw_emb, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.aux_c22_classifier = nn.Sequential(
            nn.Linear(c22_emb, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Class-specific output heads
        self.fc_shared = nn.Linear(self.combined_dim, 256)
        self.ln_shared = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout)
        
        # Individual classifiers for each sleep stage
        self.fc_classes = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(num_classes)
        ])
        
        # Specialized N1 detector using bidirectional LSTM
        self.n1_detector = nn.Sequential(
            nn.Linear(self.combined_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.n1_lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.n1_attn = nn.MultiheadAttention(128, 4, batch_first=True)
        self.n1_output = nn.Linear(128, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, raw, c22):
        # print(f"HybridSleepTransformer input shapes: raw={raw.shape}, c22={c22.shape}")
        
        # Encode raw signals and Catch22 features
        r = self.epoch_enc(raw)
        # print(f"EpochEncoder output shape: {r.shape}")
        
        c = self.c22_enc(c22)
        # print(f"C22Encoder output shape: {c.shape}")
        
        # Ensure sequence dimensions match by truncating or padding
        B, Sr, D_r = r.shape
        B, Sc, D_c = c.shape
        
        if Sr != Sc:
            # print(f"WARNING: Sequence length mismatch! Sr={Sr}, Sc={Sc}")
            # Use the minimum sequence length to ensure compatibility
            S_min = min(Sr, Sc)
            r = r[:, :S_min, :]
            c = c[:, :S_min, :]
            # print(f"After adjustment: r={r.shape}, c={c.shape}")
        
        # Get auxiliary predictions for intermediate supervision
        aux_raw_preds = self.aux_raw_classifier(r)
        aux_c22_preds = self.aux_c22_classifier(c)
        # print(f"Auxiliary prediction shapes: raw={aux_raw_preds.shape}, c22={aux_c22_preds.shape}")
        
        # Combine features
        x = torch.cat([r, c], dim=2)
        # print(f"Combined features shape: {x.shape}")
        
        B, S, D = x.shape
        
        # Apply fusion
        x_flat = x.view(B*S, D)
        # print(f"Flattened combined shape: {x_flat.shape}")
        
        x_fused = self.fusion(x_flat)
        x_fused = F.relu(self.ln_fusion(x_fused))
        x = x_fused.view(B, S, D)
        # print(f"Fused features shape: {x.shape}")
        
        # Add positional encoding, ensuring dimensions match
        x = x + self.pos_encoder[:, :S, :]
        # print(f"After positional encoding: {x.shape}")
        
        # Append class tokens to sequence
        class_tokens = self.class_tokens.expand(B, -1, -1)
        # print(f"Class tokens shape: {class_tokens.shape}")
        
        x_with_class = torch.cat([x, class_tokens], dim=1)
        # print(f"With class tokens shape: {x_with_class.shape}")
        
        # Apply transformer
        transformer_out = self.transformer(x_with_class)
        # print(f"Transformer output shape: {transformer_out.shape}")
        
        # Split sequence and class tokens
        seq_out = transformer_out[:, :S, :]
        class_out = transformer_out[:, S:, :]
        # print(f"Split shapes - seq_out: {seq_out.shape}, class_out: {class_out.shape}")
        
        # Apply specialized N1 detector
        n1_features = self.n1_detector(seq_out)
        # print(f"N1 features shape: {n1_features.shape}")
        
        n1_lstm_out, _ = self.n1_lstm(n1_features)
        # print(f"N1 LSTM output shape: {n1_lstm_out.shape}")
        
        n1_attn_out, _ = self.n1_attn(n1_lstm_out, n1_lstm_out, n1_lstm_out)
        # print(f"N1 attention output shape: {n1_attn_out.shape}")
        
        n1_score = self.n1_output(n1_attn_out)
        # print(f"N1 score shape: {n1_score.shape}")
        
        # Apply shared dense layer
        shared_features = self.dropout(F.relu(self.ln_shared(self.fc_shared(seq_out))))
        # print(f"Shared features shape: {shared_features.shape}")
        
        # Apply class-specific heads
        class_outputs = []
        for i, fc in enumerate(self.fc_classes):
            # For N1 class, incorporate the specialized detector
            if i == 1:  # N1 class index
                class_score = fc(shared_features) + n1_score
            else:
                class_score = fc(shared_features)
            class_outputs.append(class_score)
            # print(f"Class {i} output shape: {class_score.shape}")
        
        # Concatenate outputs
        output = torch.cat(class_outputs, dim=2)
        # print(f"Final output shape: {output.shape}")
        
        if self.training:
            return output, aux_raw_preds, aux_c22_preds
        else:
            return output

# -------------------------
# ==== Loss       =========
# -------------------------
def focal_loss(inputs, targets, alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    """
    Enhanced Focal Loss with stronger focus on N1 class.
    - inputs: logits of shape (B, S, C)
    - targets: ground‑truth indices of shape (B, S)
    """
    # Flatten to (N, C) and (N,)
    B, S, C = inputs.shape
    logits = inputs.view(-1, C)      # (N, C)
    tgt    = targets.view(-1)        # (N,)

    # Log‑softmax + probabilities
    logp = F.log_softmax(logits, dim=1)         # (N, C)
    p    = torch.exp(logp).clamp(min=1e-7)      # (N, C)

    # Cross‑entropy per sample
    ce   = F.nll_loss(logp, tgt, reduction='none')  # (N,)

    # p_t = probability of the true class for each sample
    pt   = p.gather(1, tgt.unsqueeze(1)).squeeze(1)  # (N,)

    # Enhanced alpha weighting for difficult classes
    # N1 gets highest weight, but also boost N3 and REM moderately
    n1_mask = (tgt == 1).float()
    n3_mask = (tgt == 3).float()
    rem_mask = (tgt == 4).float()
    
    # Apply different alpha weights to different classes
    alpha = alpha_general * (1 - n1_mask - n3_mask - rem_mask) + \
            alpha_n1 * n1_mask + \
            0.5 * n3_mask + \
            0.45 * rem_mask

    # Enhanced focal term with higher gamma
    loss = alpha * ((1 - pt) ** gamma) * ce   # (N,)

    return loss.mean()

# -------------------------
# ==== Training Loop ======
# -------------------------
def mixup_batch(raw, c22, labels, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = raw.size(0)
    index = torch.randperm(batch_size).to(raw.device)
    
    mixed_raw = lam * raw + (1 - lam) * raw[index]
    mixed_c22 = lam * c22 + (1 - lam) * c22[index]
    return mixed_raw, mixed_c22, labels, labels[index], lam

def train_epoch(model, loader, optimizer, scheduler=None, mixup_alpha=0.2):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for raw, c22, labels in loader:
        # Sanitize inputs
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        c22 = torch.nan_to_num(c22, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        labels = labels.to(device)
        
        # Apply mixup data augmentation with probability 0.5
        if np.random.random() < 0.5:
            mixed_raw, mixed_c22, labels_a, labels_b, lam = mixup_batch(raw, c22, labels, alpha=mixup_alpha)
            use_mixup = True
        else:
            mixed_raw, mixed_c22 = raw, c22
            labels_a = labels
            use_mixup = False
        
        optimizer.zero_grad()
        
        # Forward pass with auxiliary outputs during training
        outputs = model(mixed_raw, mixed_c22)
        
        # Handle outputs based on model training mode
        if isinstance(outputs, tuple):
            main_preds, aux_raw_preds, aux_c22_preds = outputs
            
            # Calculate losses
            main_loss = focal_loss(main_preds, labels_a)
            
            if use_mixup:
                aux_raw_loss = lam * focal_loss(aux_raw_preds, labels_a) + (1 - lam) * focal_loss(aux_raw_preds, labels_b)
                aux_c22_loss = lam * focal_loss(aux_c22_preds, labels_a) + (1 - lam) * focal_loss(aux_c22_preds, labels_b)
            else:
                aux_raw_loss = focal_loss(aux_raw_preds, labels_a)
                aux_c22_loss = focal_loss(aux_c22_preds, labels_a)
            
            # Combine losses with weights
            loss = main_loss + 0.3 * aux_raw_loss + 0.3 * aux_c22_loss
            
            # For accuracy calculation
            preds = main_preds.argmax(dim=-1)
        else:
            # Single output (should not happen in training, but just in case)
            if use_mixup:
                loss = lam * focal_loss(outputs, labels_a) + (1 - lam) * focal_loss(outputs, labels_b)
            else:
                loss = focal_loss(outputs, labels_a)
            preds = outputs.argmax(dim=-1)
        
        # Check for numerical stability
        if not torch.isfinite(loss):
            # print("Skipping batch: non-finite loss")
            continue
        
        # Backward pass and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased from 0.5
        optimizer.step()
        
        # Update learning rate if using OneCycleLR or similar
        if scheduler is not None and isinstance(scheduler, (
            torch.optim.lr_scheduler.OneCycleLR,
            torch.optim.lr_scheduler.CyclicLR
        )):
            scheduler.step()
        
        # Calculate running statistics
        running_loss += loss.item() * raw.size(0)
        
        # Calculate accuracy (only for non-mixup batches)
        if not use_mixup:
            correct += (preds == labels).sum().item()
            total += labels.numel()
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def smooth_predictions(predictions, window_size=5):
    """Apply temporal smoothing to predictions with special handling for N1 class.
    
    Args:
        predictions: 1D array of class predictions
        window_size: Size of smoothing window (should be odd)
    
    Returns:
        1D array of smoothed predictions
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd
        
    smoothed = np.copy(predictions)
    padded = np.pad(predictions, (window_size//2, window_size//2), mode='edge')
    
    for i in range(len(smoothed)):
        window = padded[i:i+window_size]
        
        # Special handling for N1 class (index 1)
        if 1 in window:  # If N1 is in the window
            n1_count = np.sum(window == 1)
            # If N1 appears multiple times or is in the center, preserve it
            if n1_count > 1 or window[window_size//2] == 1:
                smoothed[i] = 1
            # Otherwise use consensus voting
            else:
                counts = np.bincount(window, minlength=5)
                # Exclude N1 from voting if it's just a single occurrence
                if counts[1] == 1:
                    counts[1] = 0
                smoothed[i] = np.argmax(counts)
        else:
            # For non-N1 windows, use standard mode
            smoothed[i] = np.argmax(np.bincount(window, minlength=5))
            
    # Additional rule: prevent isolated W (0) or REM (4) classes
    for i in range(1, len(smoothed)-1):
        if (smoothed[i] == 0 or smoothed[i] == 4) and smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]
    
    return smoothed

def smooth_predictions(predictions, window_size=5):
    """Apply temporal smoothing to predictions with special handling for N1 class.
    
    Args:
        predictions: 1D array of class predictions
        window_size: Size of smoothing window (should be odd)
    
    Returns:
        1D array of smoothed predictions
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd
        
    smoothed = np.copy(predictions)
    padded = np.pad(predictions, (window_size//2, window_size//2), mode='edge')
    
    for i in range(len(smoothed)):
        window = padded[i:i+window_size]
        
        # Special handling for N1 class (index 1)
        if 1 in window:  # If N1 is in the window
            n1_count = np.sum(window == 1)
            # If N1 appears multiple times or is in the center, preserve it
            if n1_count > 1 or window[window_size//2] == 1:
                smoothed[i] = 1
            # Otherwise use consensus voting
            else:
                counts = np.bincount(window, minlength=5)
                # Exclude N1 from voting if it's just a single occurrence
                if counts[1] == 1:
                    counts[1] = 0
                smoothed[i] = np.argmax(counts)
        else:
            # For non-N1 windows, use standard mode
            smoothed[i] = np.argmax(np.bincount(window, minlength=5))
            
    # Additional rule: prevent isolated W (0) or REM (4) classes
    for i in range(1, len(smoothed)-1):
        if (smoothed[i] == 0 or smoothed[i] == 4) and smoothed[i-1] == smoothed[i+1] and smoothed[i] != smoothed[i-1]:
            smoothed[i] = smoothed[i-1]
    
    return smoothed

def eval_epoch(model, loader, apply_smoothing=True):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    all_raw_preds = []  # Store pre-smoothing predictions
    
    with torch.no_grad():
        for raw, c22, labels in loader:
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
            c22 = torch.nan_to_num(c22, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(raw, c22)
            
            # Handle outputs based on model training mode
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Extract main predictions
            else:
                logits = outputs
                
            if not torch.isfinite(logits).all():
                print("Warning: non-finite logits in evaluation")
                continue

            # Calculate loss
            loss = focal_loss(logits, labels)
            running_loss += loss.item() * raw.size(0)

            # Get predictions
            preds = logits.argmax(dim=-1)
            
            # Store predictions and labels
            all_raw_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Return early if no valid predictions
    if not all_raw_preds:
        return float('nan'), 0.0, 0.0, [], []
    
    # Concatenate and flatten predictions and labels
    # Concatenate and flatten predictions and labels
    all_raw_preds = np.concatenate(all_raw_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    
    # Calculate raw accuracy
    raw_acc = (all_raw_preds == all_labels).mean()
    
    # Apply temporal smoothing if requested
    if apply_smoothing:
        all_smoothed_preds = smooth_predictions(all_raw_preds)
        smoothed_acc = (all_smoothed_preds == all_labels).mean()
        all_preds = all_smoothed_preds
    else:
        smoothed_acc = raw_acc
        all_preds = all_raw_preds
    
    # Calculate per-class metrics
    class_accs = []
    for cls in range(5):
        mask = (all_labels == cls)
        if np.sum(mask) > 0:
            cls_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_accs.append(cls_acc)
        else:
            class_accs.append(0.0)
    
    return running_loss/len(loader.dataset), raw_acc, smoothed_acc, all_preds, all_labels


# -------------------------
# ==== Main       =========
# -------------------------
def main():
    torch.autograd.set_detect_anomaly(True)

    # Hyperparameters (updated)
    BATCH_SIZE = 32
    NUM_EPOCHS = 50  # Increased from 35
    LEARNING_RATE = 2e-4  # Increased from 1e-5
    SEQ_LENGTH = 30  # Increased from 20
    SEQ_STRIDE = 5   # Decreased from 10 for more temporal context

    # prepare CV splits
    subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
    subjects = list(subj_map.keys())
    np.random.seed(SEED)
    np.random.shuffle(subjects)
    folds = np.array_split(subjects, 5)

    # Track metrics for all folds
    fold_results = {
        'accuracy': [],
        'f1_n1': [],
        'f1_n2': [],
        'f1_n3': [],
        'f1_rem': [],
        'f1_wake': []
    }
    
    # Full per-class results for all folds
    all_fold_confusion = np.zeros((5, 5))  # 5 classes x 5 classes
    all_fold_f1 = np.zeros((5, 5))  # 5 folds x 5 classes
    
    for k in range(5):
        print(f"\n=== Starting Fold {k+1}/5 ===")
        test_subs = folds[k]
        train_subs = [s for i, f in enumerate(folds) if i!=k for s in f]
        train_ids = [rid for s in train_subs for rid in subj_map[s]]
        test_ids = [rid for s in test_subs for rid in subj_map[s]]

        # Create datasets with updated sequence length
        train_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, train_ids)
        test_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, test_ids)

        # Enhanced weighted sampler with stronger N1 focus
        # prev start
        # labels = train_ds.seq_labels[:,0].numpy()
        # class_counts = np.bincount(labels, minlength=5)
        
        # # Print class distribution
        # print("Class distribution in training set:")
        # classes = ["Wake", "N1", "N2", "N3", "REM"]
        # for i, cls in enumerate(classes):
        #     print(f"  {cls}: {class_counts[i]} samples ({class_counts[i]/len(labels)*100:.1f}%)")
        
        # # Enhanced class weights with extra boost for N1
        # class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
        # # Extra boost for N1 class
        # class_weights[1] *= 1.5
        
        # sample_weights = class_weights[labels]
        # sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        # prev end
        # Enhanced weighted sampler with more accurate class distribution

        # new start
        # Flatten all sequence labels to get true distribution
        flat_labels = train_ds.seq_labels.reshape(-1).numpy()
        class_counts = np.bincount(flat_labels, minlength=5)

        # Print class distribution
        print("Class distribution in training set (all epochs):")
        classes = ["Wake", "N1", "N2", "N3", "REM"]
        for i, cls in enumerate(classes):
            print(f"  {cls}: {class_counts[i]} samples ({class_counts[i]/len(flat_labels)*100:.1f}%)")

        # Calculate weights based on the true distribution
        class_weights = 1.0 / np.sqrt(class_counts + 1e-6)
        # Extra boost for N1 class
        class_weights[1] *= 1.5

        # For sampling, we need weights per sequence
        # Using the mean of the class weights for all epochs in each sequence
        sequence_weights = np.zeros(len(train_ds))
        for i in range(len(train_ds)):
            seq_labels = train_ds.seq_labels[i].numpy()
            seq_weights = class_weights[seq_labels]
            sequence_weights[i] = np.mean(seq_weights)

        sampler = WeightedRandomSampler(sequence_weights, len(sequence_weights), replacement=True)
        # new end

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                 sampler=sampler, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, pin_memory=True)

        # Create model with updated architecture
        c22_dim = train_ds.c22_feats.shape[-1]
        model = HybridSleepTransformer(
            c22_dim=c22_dim,
            raw_emb=128,
            c22_emb=64,
            num_classes=5,
            num_layers=3,  # Increased from 2
            num_heads=8,   # Increased from 4
            dropout=0.2,   # Increased from 0.1
            seq_length=SEQ_LENGTH
        ).to(device)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Use OneCycleLR for better convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Warm up for 30% of training
            div_factor=25,  # Initial LR is max_lr/25
            final_div_factor=1000  # End with LR 1000x smaller than max
        )
        
        # Early stopping setup
        patience = 7
        patience_counter = 0
        best_f1_n1 = 0  # Track best N1 F1 score specifically
        best_loss = float('inf')
        best_epoch = 0
        
        # Training loop
        for epoch in range(NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
            
            # Evaluate with smoothing
            val_loss, raw_acc, smoothed_acc, val_preds, val_labels = eval_epoch(model, test_loader, apply_smoothing=True)
            
            # Calculate per-class metrics
            report = classification_report(val_labels, val_preds, target_names=["W","N1","N2","N3","REM"], output_dict=True)
            f1_scores = [report[c]['f1-score'] for c in ["W","N1","N2","N3","REM"]]
            
            # Print detailed metrics
            print(f"Fold{k+1} E{epoch+1}/{NUM_EPOCHS}:")
            print(f"  Loss: train={train_loss:.4f}, val={val_loss:.4f}")
            print(f"  Acc : raw={raw_acc:.4f}, smoothed={smoothed_acc:.4f}")
            print(f"  F1  : W={f1_scores[0]:.4f}, N1={f1_scores[1]:.4f}, N2={f1_scores[2]:.4f}, N3={f1_scores[3]:.4f}, REM={f1_scores[4]:.4f}")
            
            # Adjust learning rate for ReduceLROnPlateau
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) and np.isfinite(val_loss):
                scheduler.step(val_loss)
                
            # Check if this is the best model focused on N1 performance
            current_f1_n1 = report['N1']['f1-score']
            if current_f1_n1 > best_f1_n1:
                best_f1_n1 = current_f1_n1
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau) else None,
                    'val_loss': val_loss,
                    'smoothed_acc': smoothed_acc,
                    'f1_scores': f1_scores,
                }, RESULTS_DIR/f"models/best_fold{k+1}.pth")
                
                print(f"  → New best model saved (F1-N1: {current_f1_n1:.4f})")
            else:
                patience_counter += 1
                print(f"  → No improvement for {patience_counter}/{patience} epochs. Best F1-N1: {best_f1_n1:.4f}")
                
            # Early stopping
            if patience_counter >= patience:
                print(f"  → Early stopping triggered after {epoch+1} epochs")
                break

        # Final evaluation on best model
        print(f"\nLoading best model from epoch {best_epoch+1}")
        checkpoint = torch.load(RESULTS_DIR/f"models/best_fold{k+1}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final evaluation with detailed metrics
        _, _, final_acc, final_preds, final_labels = eval_epoch(model, test_loader, apply_smoothing=True)
        final_report = classification_report(final_labels, final_preds, target_names=["W","N1","N2","N3","REM"], output_dict=True)
        final_cm = confusion_matrix(final_labels, final_preds)
        
        # Print final results
        print(f"\n==> Fold{k+1} final results:")
        print(f"  Accuracy: {final_acc:.4f}")
        print("  Per-class F1 scores:")
        for i, cls in enumerate(["W","N1","N2","N3","REM"]):
            print(f"    {cls}: {final_report[cls]['f1-score']:.4f}")
            all_fold_f1[k, i] = final_report[cls]['f1-score']
            
        print("\n  Confusion Matrix:")
        print("    W    N1    N2    N3    REM")
        for i in range(5):
            row_str = "  " + " ".join(f"{final_cm[i,j]:5d}" for j in range(5))
            print(f"{['W','N1','N2','N3','REM'][i]}: {row_str}")
            
        # Accumulate confusion matrix
        all_fold_confusion += final_cm
        
        # Store results for this fold
        fold_results['accuracy'].append(final_acc)
        fold_results['f1_wake'].append(final_report['W']['f1-score'])
        fold_results['f1_n1'].append(final_report['N1']['f1-score'])
        fold_results['f1_n2'].append(final_report['N2']['f1-score'])
        fold_results['f1_n3'].append(final_report['N3']['f1-score'])
        fold_results['f1_rem'].append(final_report['REM']['f1-score'])
        
        # Save detailed results for this fold
        np.savez(
            RESULTS_DIR/f"metrics/fold{k+1}_results.npz",
            predictions=final_preds,
            labels=final_labels,
            confusion_matrix=final_cm,
            report=final_report
        )

    # Print overall cross-validation summary
    print("\n=== Cross-Validation Summary ===")
    print("Fold accuracies:", fold_results['accuracy'])
    print(f"Mean accuracy: {np.mean(fold_results['accuracy']):.4f} ± {np.std(fold_results['accuracy']):.4f}")
    
    print("\nPer-class F1 scores across folds:")
    for cls, key in zip(["W","N1","N2","N3","REM"], 
                       ['f1_wake', 'f1_n1', 'f1_n2', 'f1_n3', 'f1_rem']):
        print(f"  {cls}: {np.mean(fold_results[key]):.4f} ± {np.std(fold_results[key]):.4f}")
    
    # Print overall confusion matrix
    print("\nOverall Confusion Matrix:")
    print("      W      N1      N2      N3      REM")
    for i in range(5):
        row_str = "  " + " ".join(f"{all_fold_confusion[i,j]:7.0f}" for j in range(5))
        print(f"{['W','N1','N2','N3','REM'][i]}: {row_str}")
    
    # Calculate per-class metrics from overall confusion matrix
    precision = np.zeros(5)
    recall = np.zeros(5)
    f1 = np.zeros(5)
    
    for i in range(5):
        # Precision: TP / (TP + FP)
        precision[i] = all_fold_confusion[i, i] / np.sum(all_fold_confusion[:, i]) if np.sum(all_fold_confusion[:, i]) > 0 else 0
        # Recall: TP / (TP + FN)
        recall[i] = all_fold_confusion[i, i] / np.sum(all_fold_confusion[i, :]) if np.sum(all_fold_confusion[i, :]) > 0 else 0
        # F1: 2 * precision * recall / (precision + recall)
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    print("\nOverall Per-class Metrics:")
    for i, cls in enumerate(["W","N1","N2","N3","REM"]):
        print(f"  {cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    
    np.savez(
        RESULTS_DIR/"metrics/cv_summary.npz",
        fold_accuracies=fold_results['accuracy'],
        fold_f1_scores=all_fold_f1,
        overall_confusion=all_fold_confusion,
        overall_precision=precision,
        overall_recall=recall,
        overall_f1=f1
    )
    
    print("\nTraining completed. Results saved to:", RESULTS_DIR)

# new
def mixup_batch(raw, c22, labels, alpha=0.2):
    """Apply mixup augmentation to a batch.
    
    Args:
        raw: Raw EEG/EOG data tensor
        c22: Catch22 features tensor
        labels: Target labels tensor
        alpha: Mixup interpolation strength parameter
        
    Returns:
        Tuple of (mixed_raw, mixed_c22, labels_a, labels_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = raw.size(0)
    index = torch.randperm(batch_size).to(raw.device)
    
    mixed_raw = lam * raw + (1 - lam) * raw[index]
    mixed_c22 = lam * c22 + (1 - lam) * c22[index]
    return mixed_raw, mixed_c22, labels, labels[index], lam

if __name__ == "__main__":
    main()