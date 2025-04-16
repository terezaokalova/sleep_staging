#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import mne
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from scipy.signal import butter, filtfilt, find_peaks, sosfiltfilt
from scipy.ndimage import median_filter

# Check for GPU availability
try:
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
except ImportError:
    use_gpu = False
    device = torch.device("cpu")

# Paths and settings (modify these paths as required)
BASE_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'
DATA_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/processed_sleepedf'
RESULTS_DIR = '/Users/tereza/spring_4830/STAT-4830-GOALZ-project/results/sleepedf_res'
for d in [DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry']

USE_MULTIPLE_CHANNELS = True
CHANNELS_TO_LOAD = ["EEG Fpz-Cz", "EOG horizontal"] if USE_MULTIPLE_CHANNELS else ["EEG Fpz-Cz"]

TARGET_SFREQ = 100.0
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
EPOCH_LENGTH = 30.0
SEQ_LENGTH = 20
SEQ_STRIDE = 10

ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # collapse stage 3 and 4 into N3
    "Sleep stage R": 4
}

# =================== Data Processing Functions ===================

def detect_spindles(epoch_data, spindle_band=(11, 16), threshold_factor=2.5):
    """
    Detect spindles in each epoch using the following steps:
      1. Bandpass filter the epoch in the spindle frequency range (11–16 Hz).
      2. Compute the envelope (absolute value) of the filtered signal.
      3. Set a threshold at threshold_factor times the mean envelope.
      4. Count peaks above the threshold as spindles.
    Returns a (n_epochs, 1) array of spindle counts.
    """
    from scipy.signal import sosfiltfilt
    sos = butter(4, spindle_band, btype='bandpass', fs=TARGET_SFREQ, output='sos')
    spindle_counts = []
    for epoch in epoch_data:  # epoch shape: (n_channels, n_times)
        # Assume first channel is EEG Fpz-Cz
        eeg = epoch[0]
        filtered = sosfiltfilt(sos, eeg)
        envelope = np.abs(filtered)
        threshold = threshold_factor * np.mean(envelope)
        peaks, _ = find_peaks(envelope, height=threshold)
        spindle_counts.append(len(peaks))
    return np.array(spindle_counts).reshape(-1, 1)

def process_record(psg_path, hyp_path, channels, target_sfreq, low_freq, high_freq, epoch_length):
    raw = mne.io.read_raw_edf(psg_path, include=channels, preload=True, verbose=False)
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    raw.filter(l_freq=low_freq, h_freq=high_freq, picks=picks, verbose=False)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)
    events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_MAP, chunk_duration=epoch_length)
    tmin = 0.0
    tmax = epoch_length - 1/raw.info['sfreq']
    epochs = mne.Epochs(raw, events=events, event_id=ANNOTATION_MAP, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]
    # Normalize each channel in each epoch
    if use_gpu:
        data_tensor = torch.tensor(data, dtype=torch.float32, device=device)
        for ch in range(data_tensor.size(1)):
            m_val = torch.mean(data_tensor[:, ch, :])
            s_val = torch.std(data_tensor[:, ch, :])
            if s_val.item() == 0:
                s_val = torch.tensor(1.0, device=device)
            data_tensor[:, ch, :] = (data_tensor[:, ch, :] - m_val) / s_val
        data = data_tensor.cpu().numpy()
    else:
        for ch in range(data.shape[1]):
            m_val = np.mean(data[:, ch, :])
            s_val = np.std(data[:, ch, :]) if np.std(data[:, ch, :]) != 0 else 1.0
            data[:, ch, :] = (data[:, ch, :] - m_val) / s_val
    return data, labels, raw.ch_names

def create_sequences(data, labels, seq_length, seq_stride):
    sequences = []
    seq_labels = []
    n_epochs = data.shape[0]
    for start in range(0, n_epochs - seq_length + 1, seq_stride):
        sequences.append(data[start:start + seq_length])
        seq_labels.append(labels[start:start + seq_length])
    return np.array(sequences), np.array(seq_labels)

def find_hypnogram(psg_file):
    subject_id = os.path.basename(psg_file)[:6]
    pattern = os.path.join(os.path.dirname(psg_file), f"{subject_id}*Hypnogram.edf")
    hyp_files = glob.glob(pattern)
    if len(hyp_files) >= 1:
        return hyp_files[0]
    return None

def process_and_save(psg_file, output_dir, channels):
    hyp_file = find_hypnogram(psg_file)
    if not hyp_file:
        print(f"Hypnogram not found for {psg_file}, skipping.")
        return
    try:
        data, labels, ch_names = process_record(psg_file, hyp_file, channels,
                                                  TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)
    except Exception as e:
        print(f"Error processing {psg_file}: {e}")
        return
    rec_id = os.path.basename(psg_file).replace('-PSG.edf', '')
    spindle_feats = detect_spindles(data)
    npz_path = os.path.join(output_dir, f"{rec_id}_epochs.npz")
    np.savez_compressed(npz_path,
                        data=data.astype('float32'),
                        labels=labels.astype('int8'),
                        spindle_feats=spindle_feats.astype('float32'))
    sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
    np.savez_compressed(os.path.join(output_dir, f"{rec_id}_sequences.npz"),
                        sequences=sequences.astype('float32'),
                        seq_labels=seq_labels.astype('int8'))
    print(f"Processed {rec_id}: epochs {data.shape[0]}, sequences {sequences.shape[0]}, channels: {ch_names}")

def main_processing():
    psg_files = []
    for sub in SUBFOLDERS:
        psg_files.extend(glob.glob(os.path.join(BASE_DIR, sub, '*-PSG.edf')))
    print(f"Found {len(psg_files)} PSG files. GPU Enabled: {use_gpu}")
    Parallel(n_jobs=2)(delayed(process_and_save)(f, DATA_DIR, CHANNELS_TO_LOAD) for f in psg_files)
    print("EDF to NPZ conversion complete.")

# =================== PyTorch Dataset Classes ===================

class SleepDataset(Dataset):
    def __init__(self, npz_dir):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "*_epochs.npz")))
        data_list = []
        label_list = []
        for file in self.npz_files:
            loaded = np.load(file)
            data_list.append(loaded['data'])
            label_list.append(loaded['labels'])
        self.data = np.concatenate(data_list, axis=0)  # shape: (total_epochs, n_channels, n_times)
        self.labels = np.concatenate(label_list, axis=0)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(1)  # final shape: (n_channels, 1, n_times)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class SelfSupervisedSleepDataset(Dataset):
    def __init__(self, npz_dir):
        self.npz_files = sorted(glob.glob(os.path.join(npz_dir, "*_epochs.npz")))
        data_list = []
        for file in self.npz_files:
            loaded = np.load(file)
            data_list.append(loaded['data'])
        self.data = np.concatenate(data_list, axis=0)  # (total_epochs, n_channels, n_times)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        epoch = self.data[idx]
        # Add a singleton channel dimension if needed so that shape becomes (n_channels, 1, n_times)
        if len(epoch.shape) == 2:
            epoch = np.expand_dims(epoch, axis=0)
        aug1 = augment_epoch(epoch)
        aug2 = augment_epoch(epoch)
        return torch.tensor(aug1, dtype=torch.float32), torch.tensor(aug2, dtype=torch.float32)

def augment_epoch(epoch, noise_std=0.05):
    """Simple augmentation: add random Gaussian noise."""
    noise = np.random.randn(*epoch.shape) * noise_std
    return epoch + noise

# =================== Model Definitions ===================

# ResidualBlock from previous code
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# Encoder (CNN) for both self-supervised and supervised models
class Encoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.resblock = ResidualBlock(16, 16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, embedding_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.resblock(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        embedding = self.fc(x)
        return embedding

# Projection Head for Self-Supervised Learning
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, projection_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(projection_dim, projection_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Self-Supervised Model (Encoder + Projection Head)
class SelfSupervisedModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.projection_head = ProjectionHead()
    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return projection

# Domain Adaptation: Gradient Reversal Layer and Domain Classifier
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim=128, num_domains=10):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_domains)
    def forward(self, x):
        return self.fc(x)

# Supervised Model with Domain Adaptation
class SupervisedModel(nn.Module):
    def __init__(self, in_channels, num_classes=5, num_domains=10, use_domain_adaptation=True):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.classifier = nn.Linear(128, num_classes)
        self.use_domain_adaptation = use_domain_adaptation
        if use_domain_adaptation:
            self.domain_classifier = DomainClassifier(128, num_domains)
    def forward(self, x, lambda_=0.1):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        if self.use_domain_adaptation:
            reversed_embedding = grad_reverse(embedding, lambda_)
            domain_logits = self.domain_classifier(reversed_embedding)
            return logits, domain_logits
        return logits, None

# =================== Loss Functions ===================
def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * ((1 - pt) ** gamma) * ce_loss
    return loss.mean()

def nt_xent_loss(z1, z2, temperature=0.5):
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1_norm, z2_norm], dim=0)
    sim_matrix = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2*batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# =================== Evaluation Functions ===================
def median_smoothing(predictions, kernel_size=3):
    return median_filter(predictions, size=kernel_size)

def viterbi_decode(log_probs, transition_matrix):
    T, num_classes = log_probs.shape
    viterbi = np.zeros((T, num_classes))
    backpointer = np.zeros((T, num_classes), dtype=np.int)
    viterbi[0] = log_probs[0]
    for t in range(1, T):
        for j in range(num_classes):
            trans_scores = viterbi[t-1] + np.log(transition_matrix[:, j] + 1e-8)
            best_prev = np.argmax(trans_scores)
            viterbi[t, j] = trans_scores[best_prev] + log_probs[t, j]
            backpointer[t, j] = best_prev
    best_last_state = np.argmax(viterbi[-1])
    best_path = [best_last_state]
    for t in range(T-1, 0, -1):
        best_last_state = backpointer[t, best_last_state]
        best_path.insert(0, best_last_state)
    return best_path

def evaluate_model(model, dataloader, transition_matrix=None, use_median_smoothing=True):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            logits, _ = model(inputs, lambda_=0.1)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    if use_median_smoothing:
        all_preds = median_smoothing(all_preds, kernel_size=3)
    print("Final Evaluation Report:")
    print(classification_report(all_targets, all_preds))
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# =================== Training Functions ===================
def train_self_supervised(epochs=5, batch_size=64):
    dataset = SelfSupervisedSleepDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = SelfSupervisedModel(in_channels=1)  # assuming single EEG channel input
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for aug1, aug2 in dataloader:
            aug1, aug2 = aug1.to(device), aug2.to(device)
            z1 = model(aug1)
            z2 = model(aug2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * aug1.size(0)
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        print(f"Self-Supervised Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    torch.save(model.encoder.state_dict(), os.path.join(RESULTS_DIR, "self_supervised_encoder.pth"))
    plt.figure()
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Pretraining Loss")
    plt.title("Self-Supervised Pretraining Loss")
    plt.grid(True)
    plt.show()
    print("Self-supervised pretraining complete.")

def train_supervised(epochs=10, batch_size=32, use_domain_adaptation=True):
    dataset = SleepDataset(DATA_DIR)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    sample, _ = dataset[0]
    n_channels = sample.shape[0]
    
    model = SupervisedModel(in_channels=n_channels, num_classes=5, num_domains=10, use_domain_adaptation=use_domain_adaptation)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_train = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, domain_logits = model(inputs, lambda_=0.1)
            loss = focal_loss(logits, targets)
            if domain_logits is not None:
                # Dummy domain labels (all zeros); replace with actual subject IDs if available
                dummy_domain = torch.zeros(targets.size(0), dtype=torch.long, device=device)
                domain_loss = F.cross_entropy(domain_logits, dummy_domain)
                loss = loss + 0.1 * domain_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits, 1)
            total_train += targets.size(0)
            total_correct += (predicted == targets).sum().item()
        avg_train_loss = total_loss / total_train
        train_acc = total_correct / total_train * 100
        train_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs, lambda_=0.1)
                loss = focal_loss(logits, targets)
                total_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(logits, 1)
                correct_val += (predicted == targets).sum().item()
                total_val += targets.size(0)
                all_val_preds.append(predicted.cpu().numpy())
                all_val_targets.append(targets.cpu().numpy())
        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val * 100
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)
    print("Final Validation Report:")
    print(classification_report(all_val_targets, all_val_preds))
    cm = confusion_matrix(all_val_targets, all_val_preds)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "supervised_model.pth"))
    print("Supervised training complete.")

# =================== Main Entry Point ===================

def main():
    parser = argparse.ArgumentParser(description="Sleep-EDF Preprocessing and Training Pipeline")
    parser.add_argument("--mode", type=str, choices=["process", "pretrain", "train"], required=True,
                        help="Mode: 'process' for EDF-to-NPZ conversion, 'pretrain' for self-supervised pretraining, 'train' for supervised training with domain adaptation")
    args = parser.parse_args()
    if args.mode == "process":
        main_processing()
    elif args.mode == "pretrain":
        train_self_supervised()
    elif args.mode == "train":
        train_supervised()

if __name__ == '__main__':
    main()