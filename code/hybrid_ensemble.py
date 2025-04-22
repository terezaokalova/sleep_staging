#!/usr/bin/env python3
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

# paths
BASE               = Path("/users/okalova/sleep/STAT-4830-GOALZ-project/data")
PROCESSED_DATA_DIR = BASE/"processed_sleepedf"
CATCH22_DATA_DIR   = BASE/"c22_processed_sleepedf"
RESULTS_DIR        = BASE/"hybrid_ensemble_results"
for d in ("plots","models","metrics"):
    (RESULTS_DIR/d).mkdir(parents=True, exist_ok=True)

# reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# hyperparams
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 2e-4
TRAIN_RATIO   = 0.8
SEQ_LENGTH    = 30
SEQ_STRIDE    = 5

# stage1 settings
LR1         = 1e-4
DESIRED_N1  = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage 1: N1 vs not
class Stage1Dataset(Dataset):
    def __init__(self, raw_dir):
        files, seqs, labs = glob.glob(str(raw_dir/"*_sequences.npz")), [], []
        for f in files:
            d = np.load(f)
            s, lab = d["sequences"], d["seq_labels"]
            if s.ndim==4:
                N,S,C,T = s.shape
                s       = s.reshape(N*S, C, T)
                lab     = lab.reshape(N*S)
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

# original hybrid dataset, model, loss, etc.
def get_true_subject_id(filename):
    b = Path(filename).stem
    return b[:5] if b.startswith(("SC4","ST7")) else b[:6]

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
    train_subs, test_subs = subs[:n_train], subs[n_train:]
    train_ids = [rid for s in train_subs for rid in subj_map[s]]
    test_ids  = [rid for s in test_subs  for rid in subj_map[s]]
    return train_ids, test_ids, train_subs, test_subs

class HybridSleepDataset(Dataset):
    def __init__(self, raw_dir, c22_dir, recording_ids=None):
        all_raw = glob.glob(str(raw_dir/"*_sequences.npz"))
        all_c22 = glob.glob(str(c22_dir/"*_c22.csv"))
        if recording_ids:
            all_raw = [p for p in all_raw if any(rid in p for rid in recording_ids)]
            all_c22 = [p for p in all_c22 if any(rid in p for rid in recording_ids)]
        raw_map = {Path(p).stem.split("_")[0]:p for p in all_raw}
        c22_map = {Path(p).stem.split("_")[0]:p for p in all_c22}
        ids = sorted(raw_map.keys() & c22_map.keys())
        seqs, c22s, labels = [], [], []
        for rid in ids:
            d = np.load(raw_map[rid])
            s, lab = d["sequences"], d["seq_labels"]
            if s.ndim==4:
                N,S,C,T = s.shape
                s       = s.reshape(N*S,C,T)
                lab     = lab.reshape(N*S)
            df = pd.read_csv(c22_map[rid])
            f  = df.drop(columns="label").values
            nS = s.shape[0]; feat_dim = f.shape[1]; exp = nS*SEQ_LENGTH
            if f.shape[0]!=exp:
                newf = np.zeros((nS,SEQ_LENGTH,feat_dim),np.float32)
                for i in range(nS):
                    st, en = i*SEQ_STRIDE, i*SEQ_STRIDE+SEQ_LENGTH
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
                f = f.reshape(nS,SEQ_LENGTH,feat_dim).astype(np.float32)
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
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2,32,3,padding=1); self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,3,padding=1); self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,128,3,padding=1);self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128,128,3,padding=1);self.bn4 = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(128,32,1), nn.ReLU(),
            nn.Conv1d(32,128,1), nn.Sigmoid()
        )
        self.adaptive_pool= nn.AdaptiveAvgPool1d(48)
        self.fc    = nn.Linear(128*48, embedding_dim)
        self.ln    = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self,x):
        B,S,C,T = x.shape
        x = x.view(B*S,C,T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x * self.channel_attn(x)
        x = self.adaptive_pool(x)
        x = x.view(B*S,-1)
        x = self.ln(self.dropout(F.relu(self.fc(x))))
        return x.view(B,S,-1)

class C22Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.ln0 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim,256); self.ln1=nn.LayerNorm(256)
        self.fc2 = nn.Linear(256,128);        self.ln2=nn.LayerNorm(128)
        self.fc3 = nn.Linear(128,embedding_dim); self.ln3=nn.LayerNorm(embedding_dim)
        self.dropout=nn.Dropout(0.2)
    def forward(self,x):
        B,S,D = x.shape
        h = x.view(B*S,D)
        h = self.dropout(F.relu(self.ln1(self.fc1(self.ln0(h)))))
        h = self.dropout(F.relu(self.ln2(self.fc2(h))))
        h = self.dropout(F.relu(self.ln3(self.fc3(h))))
        return h.view(B,S,-1)

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_emb=128, c22_emb=64,
                 num_classes=5, num_layers=3, num_heads=8,
                 dropout=0.2, seq_length=SEQ_LENGTH):
        super().__init__()
        self.epoch_enc = EpochEncoder(raw_emb)
        self.c22_enc   = C22Encoder(c22_dim, c22_emb)
        D = raw_emb + c22_emb
        self.fusion   = nn.Sequential(
            nn.Linear(D,D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(D,D)
        )
        self.ln_fusion= nn.LayerNorm(D)
        self.pos_encoder = nn.Parameter(torch.randn(1,seq_length,D))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=num_heads,
            dim_feedforward=8*D,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer,num_layers)
        self.aux_raw_classifier= nn.Sequential(
            nn.Linear(raw_emb,128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,num_classes)
        )
        self.aux_c22_classifier= nn.Sequential(
            nn.Linear(c22_emb,128), nn.LayerNorm(128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128,num_classes)
        )
        self.fc_shared = nn.Linear(D,256); self.ln_shared=nn.LayerNorm(256)
        self.dropout   = nn.Dropout(dropout)
        self.fc_classes= nn.ModuleList([nn.Linear(256,1) for _ in range(num_classes)])
        # N1 detector
        self.n1_detector= nn.Sequential(
            nn.Linear(D,128), nn.LayerNorm(128), nn.ReLU()
        )
        self.n1_lstm   = nn.LSTM(128,64,batch_first=True,bidirectional=True)
        self.n1_attn   = nn.MultiheadAttention(128,4,batch_first=True)
        self.n1_output = nn.Linear(128,1)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias,0)
    def forward(self,raw,c22):
        r = self.epoch_enc(raw)
        c = self.c22_enc(c22)
        B,S,Dr = r.shape
        _,Sc,Dc = c.shape
        if Sr:=r.shape[1] != Sc:
            m = min(Sr,Sc)
            r = r[:,:m]; c = c[:,:m]
        aux1 = self.aux_raw_classifier(r)
        aux2 = self.aux_c22_classifier(c)
        x = torch.cat([r,c],dim=2)
        x = F.relu(self.ln_fusion(self.fusion(x.view(B*S,-1)))).view(B,S,-1)
        x = x + self.pos_encoder[:,:x.size(1),:]
        class_tokens = self.pos_encoder.new_empty(B,1,x.size(2))
        class_tokens.normal_()
        x = torch.cat([x,class_tokens],dim=1)
        out = self.transformer(x)
        seq_out, class_out = out[:,:S,:], out[:,S:,:]
        nf = self.n1_detector(seq_out)
        h,_ = self.n1_lstm(nf); a,_ = self.n1_attn(h,h,h)
        n1_score = self.n1_output(a)
        shared = self.dropout(F.relu(self.ln_shared(self.fc_shared(seq_out))))
        class_outputs = []
        for i,fc in enumerate(self.fc_classes):
            if i==1:
                cs = fc(shared)+n1_score
            else:
                cs = fc(shared)
            class_outputs.append(cs)
        return torch.cat(class_outputs,dim=2), aux1, aux2

def focal_loss(inputs, targets,
               alpha_general=0.25, alpha_n1=0.9, gamma=2.5):
    B,S,C = inputs.shape
    logits = inputs.view(-1,C); tgt = targets.view(-1)
    logp, p = F.log_softmax(logits,1), torch.exp(F.log_softmax(logits,1)).clamp(min=1e-7)
    ce = F.nll_loss(logp,tgt,reduction='none')
    pt = p.gather(1,tgt.unsqueeze(1)).squeeze(1)
    n1m = (tgt==1).float()
    alpha = alpha_general*(1-n1m)+alpha_n1*n1m
    return (alpha*((1-pt)**gamma)*ce).mean()

def mixup_batch(raw, c22, labels, alpha=0.2):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1
    B = raw.size(0)
    idx = torch.randperm(B).to(raw.device)
    return lam*raw + (1-lam)*raw[idx], lam*c22 + (1-lam)*c22[idx], labels, labels[idx], lam

def train_epoch(model, loader, optimizer, scheduler=None,
                mixup_alpha=0.2, detector=None):
    model.train()
    running_loss=correct=total=0
    for raw,c22,labels in loader:
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        c22 = torch.nan_to_num(c22, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
        labels = labels.to(device)
        if np.random.rand()<0.5:
            raw2,c222,la,lb,lam = mixup_batch(raw,c22,labels,mixup_alpha)
            use_mixup=True
        else:
            raw2, c222, la, lb, lam = raw, c22, labels, labels, 1
            use_mixup=False
        optimizer.zero_grad()
        # stage1 scores
        with torch.no_grad():
            B,S,C,T = raw2.shape
            det_logits = detector(raw2.view(B*S,C,T))
            det_score  = det_logits[:,1].view(B,S)
        main_out, aux1, aux2 = model(raw2, c222)
        main_out[:,:,1] += det_score
        main_loss = focal_loss(main_out, la)
        aux1_loss = focal_loss(aux1, la)
        aux2_loss = focal_loss(aux2, la)
        loss = main_loss + 0.3*aux1_loss + 0.3*aux2_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if scheduler and isinstance(scheduler,
           (optim.lr_scheduler.OneCycleLR,optim.lr_scheduler.CyclicLR)):
            scheduler.step()
        running_loss += loss.item()*raw.size(0)
        if not use_mixup:
            preds = main_out.argmax(-1)
            correct += (preds==labels).sum().item()
            total   += labels.numel()
    return running_loss/len(loader.dataset), correct/total

def smooth_predictions(predictions, window_size=5):
    if window_size%2==0: window_size+=1
    sm = predictions.copy()
    pad = np.pad(predictions,(window_size//2,)*2,mode='edge')
    for i in range(len(sm)):
        w = pad[i:i+window_size]
        if 1 in w:
            c = (w==1).sum()
            sm[i] = 1 if (c>1 or w[window_size//2]==1) else np.argmax(np.bincount(w, minlength=5, weights=[0 if x==1 and c==1 else 1 for x in w]))
        else:
            sm[i] = np.argmax(np.bincount(w, minlength=5))
    for i in range(1,len(sm)-1):
        if sm[i] in (0,4) and sm[i-1]==sm[i+1]!=sm[i]:
            sm[i] = sm[i-1]
    return sm

def eval_epoch(model, loader, apply_smoothing=True, detector=None):
    model.eval()
    running_loss=0
    all_raw, all_lbl = [], []
    with torch.no_grad():
        for raw,c22,labels in loader:
            raw = torch.nan_to_num(raw, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
            c22 = torch.nan_to_num(c22, nan=0.0, posinf=1e5, neginf=-1e5).to(device)
            labels = labels.to(device)
            B,S,C,T=raw.shape
            det_logits = detector(raw.view(B*S,C,T))
            det_score  = det_logits[:,1].view(B,S)
            out, aux1, aux2 = model(raw, c22)
            out[:,:,1] += det_score
            loss = focal_loss(out, labels)
            running_loss += loss.item()*raw.size(0)
            preds = out.argmax(-1).cpu().numpy().ravel()
            all_raw.append(preds)
            all_lbl.append(labels.cpu().numpy().ravel())
    all_raw = np.concatenate(all_raw)
    all_lbl = np.concatenate(all_lbl)
    raw_acc = (all_raw==all_lbl).mean()
    if apply_smoothing:
        sm = smooth_predictions(all_raw)
        return running_loss/len(loader.dataset), raw_acc, (sm==all_lbl).mean(), sm, all_lbl
    else:
        return running_loss/len(loader.dataset), raw_acc, raw_acc, all_raw, all_lbl

def main():
    torch.autograd.set_detect_anomaly(True)
    # Stage 1 training
    ds1      = Stage1Dataset(PROCESSED_DATA_DIR)
    w        = np.bincount(ds1.y.numpy(),minlength=2)
    w1       = (w[0]/w[1])*(DESIRED_N1/(1-DESIRED_N1))
    sw       = np.where(ds1.y.numpy()==1, w1, 1.0)
    loader1  = DataLoader(ds1, BATCH_SIZE, sampler=WeightedRandomSampler(sw,len(sw),True))
    detector = Stage1Detector().to(device)
    opt1     = optim.Adam(detector.parameters(), lr=LR1)
    for ep in range(10):
        detector.train()
        tot, ls = 0,0
        for X,y in loader1:
            X,y = X.to(device), y.to(device)
            opt1.zero_grad()
            l = F.cross_entropy(detector(X), y)
            l.backward(); opt1.step()
            ls += l.item()*X.size(0); tot += X.size(0)
        print(f"[Stage1] Ep{ep+1}: loss={ls/tot:.4f}")
    detector.eval()
    for p in detector.parameters(): p.requires_grad=False

    # CV folds
    subj_map = group_by_true_subjects(PROCESSED_DATA_DIR)
    subs     = list(subj_map.keys())
    np.random.seed(SEED); np.random.shuffle(subs)
    folds    = np.array_split(subs, 5)

    fold_results = {'accuracy':[], 'f1_n1':[], 'f1_n2':[],
                    'f1_n3':[], 'f1_rem':[], 'f1_wake':[]}
    all_conf = np.zeros((5,5)); all_f1 = np.zeros((5,5))

    for k in range(5):
        test_subs  = folds[k]
        train_subs = [s for i,f in enumerate(folds) if i!=k for s in f]
        train_ids = [rid for s in train_subs for rid in subj_map[s]]
        test_ids  = [rid for s in test_subs  for rid in subj_map[s]]

        train_ds = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, train_ids)
        test_ds  = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, test_ids)

        flat_lbls = train_ds.labels.numpy().ravel()
        cnts = np.bincount(flat_lbls, minlength=5)
        weights = 1.0/np.sqrt(cnts+1e-6)
        weights[1] *= 1.5
        seq_w = np.array([weights[train_ds.labels[i].item()] for i in range(len(train_ds))])
        sampler = WeightedRandomSampler(seq_w, len(seq_w), True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  sampler=sampler, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=4, pin_memory=True)

        model = HybridSleepTransformer(train_ds.c22.size(-1)).to(device)
        opt2  = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        sched = optim.lr_scheduler.OneCycleLR(opt2, max_lr=LEARNING_RATE,
                    epochs=NUM_EPOCHS, steps_per_epoch=len(train_loader),
                    pct_start=0.3, div_factor=25, final_div_factor=1000)

        best_f1_n1=0; patience=7; p_cnt=0; best_epoch=0
        for ep in range(NUM_EPOCHS):
            tl, ta = train_epoch(model, train_loader, opt2, sched, 0.2, detector)
            vl, ra, sa, vp, vlbl = eval_epoch(model, test_loader, True, detector)
            rpt = classification_report(vlbl, vp, target_names=["W","N1","N2","N3","REM"], output_dict=True)
            f1    = rpt["N1"]["f1-score"]
            print(f"Fold{k+1} Ep{ep+1}: train_loss={tl:.4f} val_f1_N1={f1:.4f}")
            if f1>best_f1_n1:
                best_f1_n1, best_epoch, p_cnt = f1, ep, 0
                torch.save(model.state_dict(), RESULTS_DIR/f"models/best_fold{k+1}.pth")
            else:
                p_cnt+=1
                if p_cnt>=patience: break

        # final eval
        model.load_state_dict(torch.load(RESULTS_DIR/f"models/best_fold{k+1}.pth"))
        _,_,_,fp,fl = eval_epoch(model, test_loader, True, detector)
        cm = confusion_matrix(fl,fp)
        rpt = classification_report(fl,fp,target_names=["W","N1","N2","N3","REM"],output_dict=True)
        all_conf += cm
        for i,cls in enumerate(["W","N1","N2","N3","REM"]):
            all_f1[k,i] = rpt[cls]["f1-score"]
        fold_results['accuracy'].append((fp==fl).mean())
        fold_results['f1_wake'].append(rpt["W"]["f1-score"])
        fold_results['f1_n1'].append(rpt["N1"]["f1-score"])
        fold_results['f1_n2'].append(rpt["N2"]["f1-score"])
        fold_results['f1_n3'].append(rpt["N3"]["f1-score"])
        fold_results['f1_rem'].append(rpt["REM"]["f1-score"])
        np.savez(RESULTS_DIR/f"metrics/fold{k+1}_results.npz",
                 cm=cm, report=rpt)

    # summary
    print("CV Accuracy:", fold_results['accuracy'])
    print("Mean Accuracy:", np.mean(fold_results['accuracy']))
    for cls,key in zip(["W","N1","N2","N3","REM"],
                       ['f1_wake','f1_n1','f1_n2','f1_n3','f1_rem']):
        print(f"{cls}: {np.mean(fold_results[key]):.4f} ± {np.std(fold_results[key]):.4f}")
    print("Overall Confusion Matrix:\n", all_conf)
    print("Overall F1 scores per fold:\n", all_f1)

if __name__ == "__main__":
    main()
