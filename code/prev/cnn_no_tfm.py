#!/usr/bin/env python3
import os, sys, glob, argparse, numpy as np, mne
from joblib import Parallel, delayed
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import median_filter
from sklearn.metrics import classification_report, confusion_matrix

# BASE_DIR = os.path.join("/mnt/sauce/littlab/users/okalova/sleep/sleep_staging/data/sleep-edf-database-expanded-1.0.0")
BASE_DIR = "/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0"
SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry']

# process data fully in memory (no saving to disk).
# Use multiple channels flag; if False, only use Fpz‑Cz.
USE_MULTIPLE_CHANNELS = True
CHANNELS_TO_LOAD = ["EEG Fpz-Cz", "EOG horizontal"] if USE_MULTIPLE_CHANNELS else ["EEG Fpz-Cz"]

TARGET_SFREQ = 100.0
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
EPOCH_LENGTH = 30.0   # seconds per epoch
SEQ_LENGTH = 20
SEQ_STRIDE = 10

ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # collapse stage 4 into stage 3 (N3)
    "Sleep stage R": 4
}

def check_data_path():
    if not os.path.exists(BASE_DIR):
        print(f"ERROR: Data directory not found: {BASE_DIR}")
        return False
    for sub in SUBFOLDERS:
        subfolder = os.path.join(BASE_DIR, sub)
        if not os.path.exists(subfolder):
            print(f"ERROR: Expected subfolder not found: {subfolder}")
            return False
    found = False
    for sub in SUBFOLDERS:
        files = glob.glob(os.path.join(BASE_DIR, sub, '*-PSG.edf'))
        if files:
            print(f"Found {len(files)} PSG files in {os.path.join(BASE_DIR, sub)}")
            found = True
    if not found:
        print(f"ERROR: No PSG files found in any subfolder of {BASE_DIR}")
        return False
    return True

try:
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available; using CPU")
except Exception as e:
    use_gpu = False
    device = torch.device("cpu")
    print(f"Error checking GPU: {e}. Using CPU.")

def process_record(psg_path, hyp_path, channels, target_sfreq, low_freq, high_freq, epoch_length):
    print(f"Loading {os.path.basename(psg_path)}...")
    raw = mne.io.read_raw_edf(psg_path, include=channels, preload=True, verbose=False)
    print(f"Resampling from {raw.info['sfreq']} Hz to {target_sfreq} Hz...")
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, npad="auto", verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, eog=True)
    raw.filter(l_freq=low_freq, h_freq=high_freq, picks=picks, verbose=False)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)
    events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_MAP, chunk_duration=epoch_length)
    tmin = 0.0
    tmax = epoch_length - 1 / raw.info['sfreq']
    epochs = mne.Epochs(raw, events=events, event_id=ANNOTATION_MAP, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]
    # Normalize each channel over all epochs.
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

def process_and_print(psg_file, channels):
    hyp_file = find_hypnogram(psg_file)
    if not hyp_file:
        print(f"WARNING: Hypnogram not found for {psg_file}, skipping.")
        return
    try:
        data, labels, ch_names = process_record(psg_file, hyp_file, channels,
                                                  TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)
    except Exception as e:
        print(f"Error processing {psg_file}: {e}")
        return
    rec_id = os.path.basename(psg_file).replace('-PSG.edf', '')
    spindle_feats = detect_spindles(data)
    sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
    print(f"Processed {rec_id}:")
    print(f"  Epochs shape: {data.shape}")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Spindle features shape: {spindle_feats.shape}")
    print(f"  Channels used: {ch_names}")

def main_processing():
    if not check_data_path():
        sys.exit(1)
    psg_files = []
    for sub in SUBFOLDERS:
        psg_files.extend(glob.glob(os.path.join(BASE_DIR, sub, '*-PSG.edf')))
    print(f"Found {len(psg_files)} PSG files. GPU Enabled: {use_gpu}")
    Parallel(n_jobs=2)(delayed(process_and_print)(f, CHANNELS_TO_LOAD) for f in psg_files)
    print("EDF processing complete. Results printed to terminal.")

class RawSleepDataset(Dataset):
    """
    Loads raw EDF data from BASE_DIR by processing all PSG files and concatenating epochs.
    """
    def __init__(self, base_dir):
        if not check_data_path():
            sys.exit(1)
        all_data = []
        all_labels = []
        found_files = 0
        for sub in SUBFOLDERS:
            subfolder = os.path.join(base_dir, sub)
            if not os.path.exists(subfolder):
                print(f"Warning: Subfolder {subfolder} not found")
                continue
            psg_files = glob.glob(os.path.join(subfolder, '*-PSG.edf'))
            print(f"Found {len(psg_files)} PSG files in {subfolder}")
            for psg_file in psg_files:
                hyp_file = find_hypnogram(psg_file)
                if hyp_file is None:
                    print(f"Warning: No hypnogram found for {psg_file}")
                    continue
                try:
                    print(f"Processing {os.path.basename(psg_file)}...")
                    data, labels, _ = process_record(psg_file, hyp_file, CHANNELS_TO_LOAD,
                                                     TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)
                    all_data.append(data)
                    all_labels.append(labels)
                    found_files += 1
                except Exception as e:
                    print(f"Error processing {psg_file}: {e}")
        if len(all_data) == 0:
            error_msg = (f"No raw data was loaded; please check your BASE_DIR and file structure: {base_dir}\n" +
                         f"Found {found_files} processable files.")
            raise ValueError(error_msg)
        self.data = np.concatenate(all_data, axis=0)
        self.labels = np.concatenate(all_labels, axis=0)
        print(f"Successfully loaded {len(self.data)} epochs from {found_files} files")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        # Ensure consistent shape: add a singleton dimension to get (n_channels, 1, n_times)
        x = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

class SelfSupervisedSleepDataset(Dataset):
    """
    For self-supervised pretraining: loads raw EDF epochs from RawSleepDataset and returns
    two augmented versions of each epoch.
    """
    def __init__(self, base_dir):
        raw_dataset = RawSleepDataset(base_dir)
        self.data = raw_dataset.data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        epoch = self.data[idx]
        # Check that epoch has shape (n_channels, n_times)
        if len(epoch.shape) != 2:
            raise ValueError(f"Unexpected epoch shape: {epoch.shape}, expected (n_channels, n_times)")
        # Convert epoch to tensor and add singleton dimension so that shape becomes (n_channels, 1, n_times)
        epoch_tensor = torch.tensor(epoch, dtype=torch.float32).unsqueeze(1)
        aug1 = augment_epoch(epoch_tensor)
        aug2 = augment_epoch(epoch_tensor)
        return aug1, aug2

def augment_epoch(epoch, noise_std=0.05):
    # Ensure the epoch is a torch.Tensor; add Gaussian noise
    if not isinstance(epoch, torch.Tensor):
        epoch = torch.tensor(epoch, dtype=torch.float32)
    noise = torch.randn_like(epoch) * noise_std
    return epoch + noise

# -------------------- Model Definitions --------------------
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
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblock(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=128, projection_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, projection_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(projection_dim, projection_dim)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class SelfSupervisedModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.projection_head = ProjectionHead()
    def forward(self, x):
        embedding = self.encoder(x)
        projection = self.projection_head(embedding)
        return projection

class SupervisedModel(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        return logits

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
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    labels = torch.arange(batch_size, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)
    return F.cross_entropy(sim_matrix, labels)

def median_smoothing(predictions, kernel_size=3):
    return median_filter(predictions, size=kernel_size)

def viterbi_decode(log_probs, transition_matrix):
    T, num_classes = log_probs.shape
    viterbi = np.zeros((T, num_classes))
    backpointer = np.zeros((T, num_classes), dtype=int)
    viterbi[0] = log_probs[0]
    for t in range(1, T):
        for j in range(num_classes):
            trans_scores = viterbi[t - 1] + np.log(transition_matrix[:, j] + 1e-8)
            best_prev = np.argmax(trans_scores)
            viterbi[t, j] = trans_scores[best_prev] + log_probs[t, j]
            backpointer[t, j] = best_prev
    best_last_state = np.argmax(viterbi[-1])
    best_path = [best_last_state]
    for t in range(T - 1, 0, -1):
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
            logits = model(inputs)
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

def train_self_supervised(epochs=5, batch_size=64):
    if not check_data_path():
        sys.exit(1)
    dataset = SelfSupervisedSleepDataset(BASE_DIR)
    # Determine in_channels from the first sample (should be (n_channels, 1, n_times))
    sample, _ = dataset[0]  # self-supervised returns (aug1, aug2) so sample is aug1
    in_channels = sample.shape[0]
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = SelfSupervisedModel(in_channels=in_channels)
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
    plt.figure()
    plt.plot(range(1, epochs+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Self-Supervised Pretraining Loss")
    plt.grid(True)
    plt.show()
    print("Self-supervised pretraining complete.")

def train_supervised(epochs=10, batch_size=32):
    if not check_data_path():
        print("Cannot start training - data path issues. Please fix BASE_DIR.")
        sys.exit(1)
    dataset = RawSleepDataset(BASE_DIR)
    total_samples = len(dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    sample, _ = dataset[0]
    n_channels = sample.shape[0]
    
    model = SupervisedModel(in_channels=n_channels, num_classes=5)
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
            logits = model(inputs)
            loss = focal_loss(logits, targets)
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
                logits = model(inputs)
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
    
    # Post-processing evaluation using median smoothing.
    evaluate_model(model, val_loader, use_median_smoothing=True)
    
    print("Supervised training complete.")

def main():
    parser = argparse.ArgumentParser(description="Sleep-EDF Processing and Training Pipeline (No Disk-Save Mode)")
    parser.add_argument("--mode", type=str, choices=["process", "pretrain", "train"], required=True,
                        help="Mode to run: 'process' to process EDF files and print stats, 'pretrain' for self-supervised pretraining, 'train' for supervised training")
    args = parser.parse_args()
    if args.mode == "process":
        main_processing()
    elif args.mode == "pretrain":
        train_self_supervised()
    elif args.mode == "train":
        train_supervised()

if __name__ == '__main__':
    main()
