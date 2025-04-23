import os, sys, glob
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

# Paths
BASE               = Path("/users/kimliang/sleep/STAT-4830-GOALZ-project/data")
PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
RESULTS_DIR        = BASE/"hybrid_ensemble_results"
for d in ("plots","models","metrics"):
    (RESULTS_DIR/d).mkdir(parents=True, exist_ok=True)

# Reproducibility & hyperparameters
SEED           = 42
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
LEARNING_RATE  = 3e-3        # lowered LR
SEQ_LENGTH     = 30
SEQ_STRIDE     = 5

# Stage 1 params
LR1            = 1e-4
DESIRED_N1     = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Helper class to tee stdout to a file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # Ensure output is written immediately
    def flush(self) :
        for f in self.files:
            f.flush()

# Stage/step 1: N1-vs-not detector 
class Stage1Dataset(Dataset):
    def __init__(self, raw_dir):
        files, seqs, labs = glob.glob(str(raw_dir/"*_sequences.npz")), [], []
        for f in files:
            d = np.load(f)
            s, lab = d["sequences"], d["seq_labels"]
            if s.ndim == 4:
                N,S,C,T = s.shape
                s       = s.reshape(N*S, C, T)
                lab     = lab.reshape(N*S)
            seqs.append(s)
            labs.append((lab==1).astype(np.int64))
        X = torch.from_numpy(np.concatenate(seqs,0)).float()
        y = torch.from_numpy(np.concatenate(labs,0)).long()
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class Stage1Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2,16,3,padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16,32,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(32,2)
        )
    def forward(self, x):
        return self.net(x)

#  Helper to split by subject 
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

# Hybrid 5-way dataset
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
            if s.ndim == 4:
                N,S,C,T = s.shape
                s       = s.reshape(N*S, C, T)
                lab     = lab.reshape(N*S)
            df = pd.read_csv(c22_map[rid])
            f  = df.drop(columns="label").values
            nS, feat = s.shape[0], f.shape[1]
            exp = nS * SEQ_LENGTH
            if f.shape[0] != exp:
                newf = np.zeros((nS, SEQ_LENGTH, feat), np.float32)
                for i in range(nS):
                    st, en = i*SEQ_STRIDE, i*SEQ_STRIDE+SEQ_LENGTH
                    if en <= f.shape[0]:
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
                f = f.reshape(nS, SEQ_LENGTH, feat).astype(np.float32)
            seqs.append(s.astype(np.float32))
            c22s.append(f)
            labels.append(lab.astype(np.int64))
        self.raw    = torch.from_numpy(np.concatenate(seqs,0)).float()
        self.c22    = torch.from_numpy(np.concatenate(c22s,0)).float()
        self.labels = torch.from_numpy(np.concatenate(labels,0)).long()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return (
            self.raw[i].unsqueeze(0),
            self.c22[i].unsqueeze(0),
            self.labels[i].unsqueeze(0)
        )

# Encoders & Model
class EpochEncoder(nn.Module):
    def __init__(self, emb=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2,32,3,padding=1); self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,3,padding=1); self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3,padding=1);self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,128,3,padding=1);self.bn4 = nn.BatchNorm1d(128)
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
    def forward(self, x):
        B,S,C,T = x.shape
        x = x.view(B*S, C, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x * self.attn(x)
        x = self.adp(x).view(B*S, -1)
        x = self.ln(self.do(F.relu(self.fc(x))))
        return x.view(B, S, -1)

class C22Encoder(nn.Module):
    def __init__(self, in_d, emb=64):
        super().__init__()
        self.ln0 = nn.LayerNorm(in_d)
        self.fc1 = nn.Linear(in_d,256); self.ln1=nn.LayerNorm(256)
        self.fc2 = nn.Linear(256,128); self.ln2=nn.LayerNorm(128)
        self.fc3 = nn.Linear(128,emb); self.ln3=nn.LayerNorm(emb)
        self.do  = nn.Dropout(0.2)
    def forward(self, x):
        x = x.mean(dim=2)            # (B,S,D)
        B,S,D = x.shape
        h = x.view(B*S, D)
        h = self.do(F.relu(self.ln1(self.fc1(self.ln0(h)))))
        h = self.do(F.relu(self.ln2(self.fc2(h))))
        h = self.do(F.relu(self.ln3(self.fc3(h))))
        return h.view(B, S, -1)

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_emb=128, c22_emb=64,
                 num_classes=5, num_layers=3, num_heads=8,
                 dropout=0.2, seq_len=SEQ_LENGTH):
        super().__init__()
        self.eenc     = EpochEncoder(raw_emb)
        self.cenc     = C22Encoder(c22_dim, c22_emb)
        D              = raw_emb + c22_emb
        self.fuse      = nn.Sequential(
            nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(D,D)
        )
        self.lnf       = nn.LayerNorm(D)
        self.pos       = nn.Parameter(torch.randn(1, seq_len, D))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        layer          = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads,
            dim_feedforward=8*D,
            dropout=dropout, batch_first=True
        )
        self.tfm       = nn.TransformerEncoder(layer, num_layers)
        self.aux1      = nn.Sequential(
            nn.Linear(raw_emb,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.aux2      = nn.Sequential(
            nn.Linear(c22_emb,128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )
        self.fc_shared = nn.Linear(D,256); self.ln_fc=nn.LayerNorm(256)
        self.do_shared = nn.Dropout(dropout)
        self.cls_heads = nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])
        # Specialized N1 detector
        self.n1det     = nn.Sequential(
            nn.Linear(D,128), nn.LayerNorm(128), nn.ReLU()
        )
        self.n1lstm    = nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.n1attn    = nn.MultiheadAttention(128,4,batch_first=True)
        self.n1out     = nn.Linear(128,1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, raw, c22):
        r = self.eenc(raw)
        c = self.cenc(c22)
        B,Sr,_ = r.shape
        _,Sc,_ = c.shape
        if Sr != Sc:
            m = min(Sr,Sc)
            r, c = r[:,:m], c[:,:m]
        aux1 = self.aux1(r)
        aux2 = self.aux2(c)
        x    = torch.cat([r,c], dim=2).view(B*Sr, -1)
        x    = self.lnf(self.fuse(x)).view(B, Sr, -1)
        x    = x + self.pos[:,:x.size(1),:]
        cls_tok = self.cls_token.expand(B, -1, -1)
        x    = torch.cat([x, cls_tok], dim=1)
        out  = self.tfm(x)
        seq_out = out[:, :Sr, :]
        # Specialized N1
        nf       = self.n1det(seq_out)
        h,_      = self.n1lstm(nf)
        a,_      = self.n1attn(h, h, h)
        n1_score = self.n1out(a)
        shared   = self.do_shared(F.relu(self.ln_fc(self.fc_shared(seq_out))))
        class_outputs = []
        for i, head in enumerate(self.cls_heads):
            if i == 1:
                class_outputs.append(head(shared) + n1_score)
            else:
                class_outputs.append(head(shared))
        return torch.cat(class_outputs, dim=2), aux1, aux2

# Stable focal loss 
def focal_loss(inputs, targets,
               alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    B,S,C = inputs.shape
    logits = inputs.view(-1, C)
    t      = targets.view(-1)
    logp   = F.log_softmax(logits, dim=1)
    p      = torch.exp(logp).clamp(min=1e-6, max=1-6)
    ce     = F.nll_loss(logp, t, reduction='none')
    pt     = p.gather(1, t.unsqueeze(1)).squeeze(1)
    m1     = (t == 1).float()
    alpha  = alpha_general*(1-m1) + alpha_n1*m1
    weight = alpha * ((1 - pt).clamp(min=1e-6) ** gamma)
    return (weight * ce).mean()

# Mixup utility
def mixup_batch(raw, c22, labels, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    B   = raw.size(0)
    idx = torch.randperm(B).to(raw.device)
    return (lam * raw + (1-lam) * raw[idx],
            lam * c22 + (1-lam) * c22[idx],
            labels, labels[idx], lam)

def train_epoch(model, loader, optimizer, scheduler, mixup_alpha, detector):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    for raw, c22, labels in loader:
        raw, c22, labels = raw.to(device), c22.to(device), labels.to(device)
        # mixup
        if np.random.rand() < 0.5:
            raw2, c222, la, lb, lam = mixup_batch(raw, c22, labels, mixup_alpha)
            use_mixup = True
        else:
            raw2, c222, la, lb, lam = raw, c22, labels, labels, 1
            use_mixup = False
        optimizer.zero_grad()
        B,S,C,T = raw2.shape
        with torch.no_grad():
            det_logits = detector(raw2.view(B*S, C, T))
            det_score  = det_logits[:,1].view(B, S)
            det_bias   = torch.tanh(det_score).unsqueeze(-1) * 0.5
        out, aux1, aux2 = model(raw2, c222)
        out[:,:,1] = out[:,:,1] + det_bias.squeeze(-1)
        main_loss = focal_loss(out, la)
        l1        = focal_loss(aux1, la)
        l2        = focal_loss(aux2, la)
        loss      = main_loss + 0.3*l1 + 0.3*l2
        if torch.isnan(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # tighter clipping
        optimizer.step()
        if isinstance(scheduler, (optim.lr_scheduler.OneCycleLR, 
                                   optim.lr_scheduler.CosineAnnealingLR,
                                   optim.lr_scheduler.CyclicLR)):
            scheduler.step()
        total_loss += loss.item() * raw.size(0)
        if not use_mixup:
            preds = out.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.numel()
    return total_loss / len(loader.dataset), correct / total

# Temporal smoothing
def smooth_predictions(predictions, window_size=5):
    if window_size % 2 == 0:
        window_size += 1
    sm = predictions.copy()
    pad = np.pad(predictions, (window_size//2,)*2, mode='edge')
    for i in range(len(sm)):
        w = pad[i:i+window_size]
        if 1 in w:
            cnt = (w==1).sum()
            if cnt>1 or w[window_size//2]==1:
                sm[i] = 1
            else:
                bc = np.bincount(w, minlength=5)
                if bc[1] == 1: bc[1] = 0
                sm[i] = bc.argmax()
        else:
            sm[i] = np.bincount(w, minlength=5).argmax()
    for i in range(1, len(sm)-1):
        if sm[i] in (0,4) and sm[i-1]==sm[i+1]!=sm[i]:
            sm[i] = sm[i-1]
    return sm

# Evaluation epoch
def eval_epoch(model, loader, apply_smoothing, detector):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for raw, c22, labels in loader:
            raw, c22, labels = raw.to(device), c22.to(device), labels.to(device)
            B,S,C,T = raw.shape
            det_logits = detector(raw.view(B*S, C, T))
            det_score  = det_logits[:,1].view(B, S)
            det_bias   = torch.tanh(det_score).unsqueeze(-1) * 0.5
            out, aux1, aux2 = model(raw, c22)
            out[:,:,1] = out[:,:,1] + det_bias.squeeze(-1)
            loss = focal_loss(out, labels)
            total_loss += loss.item() * raw.size(0)
            preds = out.argmax(dim=-1).cpu().numpy().ravel()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().ravel())
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    raw_acc    = (all_preds == all_labels).mean()
    if apply_smoothing:
        sm = smooth_predictions(all_preds)
        return total_loss/len(loader.dataset), raw_acc, (sm==all_labels).mean(), sm, all_labels
    else:
        return total_loss/len(loader.dataset), raw_acc, raw_acc, all_preds, all_labels

def main():
    # --- Logging Setup --- START ---
    script_name = os.path.basename(__file__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists
    log_filename = log_dir / f"{Path(script_name).stem}_{timestamp}.log"

    original_stdout = sys.stdout
    log_file = open(log_filename, 'w')
    sys.stdout = Tee(original_stdout, log_file)

    print(f"Starting training for {script_name} at {timestamp}")
    print(f"Log file: {log_filename}")
    print("-" * 30)
    # --- Logging Setup --- END ---

    try:
        # 1) Train Stage1 detector
        print("\n--- Stage 1: Training N1 Detector ---")
        ds1      = Stage1Dataset(PROCESSED_DATA_DIR)
        w        = np.bincount(ds1.y.numpy(), minlength=2)
        w1       = (w[0]/w[1])*(DESIRED_N1/(1-DESIRED_N1))
        sw       = np.where(ds1.y.numpy()==1, w1, 1.0)
        loader1  = DataLoader(ds1, BATCH_SIZE,
                             sampler=WeightedRandomSampler(sw,len(sw),True))
        detector = Stage1Detector().to(device)
        opt1     = optim.Adam(detector.parameters(), lr=LR1)
        for ep in range(10):
            detector.train()
            tot, ls = 0, 0.0
            for X,y in loader1:
                X,y = X.to(device), y.to(device)
                opt1.zero_grad()
                loss = F.cross_entropy(detector(X), y)
                loss.backward()
                opt1.step()
                ls += loss.item()*X.size(0); tot += X.size(0)
            print(f"[Stage1] Ep{ep+1}: loss={ls/tot:.4f}")
        detector.eval()
        for p in detector.parameters():
            p.requires_grad = False

        # 2) 5-fold CV
        subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
        subs     = list(subj_map.keys())
        np.random.seed(SEED)
        np.random.shuffle(subs)
        folds    = np.array_split(subs, 5)

        summary = {'acc':[], 'f1_w':[], 'f1_n1':[], 'f1_n2':[], 'f1_n3':[], 'f1_r':[]}
        all_cm  = np.zeros((5,5), dtype=int)

        for k in range(5):
            test_subs  = folds[k]
            train_subs = [s for i,f in enumerate(folds) if i!=k for s in f]
            train_ids  = [rid for s in train_subs for rid in subj_map[s]]
            test_ids   = [rid for s in test_subs  for rid in subj_map[s]]

            tr_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, train_ids)
            te_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, test_ids)

            flat = tr_ds.labels.view(-1).numpy()
            cnts = np.bincount(flat, minlength=5)
            wts  = 1.0 / np.sqrt(cnts + 1e-6)
            wts[1] *= 1.5
            seq_w = np.array([wts[tr_ds.labels[i].item()] for i in range(len(tr_ds))])
            samp  = WeightedRandomSampler(seq_w, len(seq_w), True)

            tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE,
                               sampler=samp, num_workers=4, pin_memory=True)
            te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=4, pin_memory=True)

            model = HybridSleepTransformer(tr_ds.c22.size(-1)).to(device)
            opt2  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt2, 
                        T_max=NUM_EPOCHS * len(tr_ld),
                        eta_min=LEARNING_RATE / 100)

            best_n1=0.0; patience=7; wait=0
            for ep in range(NUM_EPOCHS):
                tl, ta = train_epoch(model, tr_ld, opt2, sched, 0.2, detector)
                vl, ra, sa, preds, labs = eval_epoch(model, te_ld, True, detector)
                rpt = classification_report(labs, preds,
                            target_names=["W","N1","N2","N3","REM"], output_dict=True)
                f1n1 = rpt["N1"]["f1-score"]
                print(f"Fold{k+1} Ep{ep+1}: train_loss={tl:.4f} N1_f1={f1n1:.4f}")
                if f1n1 > best_n1:
                    best_n1, wait = f1n1, 0
                    torch.save(model.state_dict(), RESULTS_DIR/f"models/best_fold{k+1}.pth")
                else:
                    wait += 1
                    if wait >= patience:
                        break

            model.load_state_dict(torch.load(RESULTS_DIR/f"models/best_fold{k+1}.pth"))
            _, _, _, preds, labs = eval_epoch(model, te_ld, True, detector)
            cm  = confusion_matrix(labs, preds)
            rpt = classification_report(labs, preds,
                        target_names=["W","N1","N2","N3","REM"], output_dict=True)
            all_cm += cm
            summary['acc'].append((preds==labs).mean())
            summary['f1_w'].append(rpt["W"]["f1-score"])
            summary['f1_n1'].append(rpt["N1"]["f1-score"])
            summary['f1_n2'].append(rpt["N2"]["f1-score"])
            summary['f1_n3'].append(rpt["N3"]["f1-score"])
            summary['f1_r'].append(rpt["REM"]["f1-score"])
            np.savez(RESULTS_DIR/f"metrics/fold{k+1}.npz", cm=cm, report=rpt)

        print("CV Accuracies:", summary['acc'])
        for key in ['f1_w','f1_n1','f1_n2','f1_n3','f1_r']:
            arr = np.array(summary[key])
            print(f"{key}: {arr.mean():.4f} ± {arr.std():.4f}")
        print("Overall Confusion Matrix:\n", all_cm)

    finally:
        # --- Restore stdout and close log file --- START ---
        print("-" * 30)
        print(f"Training finished at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        sys.stdout = original_stdout # Restore standard output
        log_file.close()
        print(f"\nTraining log saved to {log_filename}") # Notify user in terminal
        # --- Restore stdout and close log file --- END ---

if __name__ == "__main__":
    main()
