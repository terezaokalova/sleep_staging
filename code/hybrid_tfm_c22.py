#!/usr/bin/env python3
import os, sys, glob, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib
from datetime import datetime

# Data paths - updated to match server paths
PROCESSED_DATA_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/processed_sleepedf'
CATCH22_DATA_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/c22_processed_sleepedf'
RESULTS_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/hybrid_model_results'

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "metrics"), exist_ok=True)

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model parameters
BATCH_SIZE = 32
NUM_EPOCHS = 35
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
SEQ_LENGTH = 20

# Check for GPU availability
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

def get_true_subject_id(filename):
    """Extract true subject ID ignoring the night number"""
    basename = os.path.basename(filename).split('_')[0]
    if basename.startswith('SC4'):
        return basename[:5]  # SC4xx - first 5 chars for Sleep Cassette
    elif basename.startswith('ST7'):
        return basename[:5]  # ST7xx - first 5 chars for Sleep Telemetry
    else:
        return basename[:6]  # Fallback to original logic

def group_by_true_subjects(data_dir):
    """Map true subject IDs to their recording IDs"""
    all_files = glob.glob(os.path.join(data_dir, '*_sequences.npz'))
    
    true_subject_map = {}
    for f in all_files:
        recording_id = os.path.basename(f).split('_')[0]
        true_subject = get_true_subject_id(f)
        
        if true_subject not in true_subject_map:
            true_subject_map[true_subject] = []
        true_subject_map[true_subject].append(recording_id)
    
    print(f"Found {len(true_subject_map)} unique subjects")
    
    return true_subject_map

def split_true_subjects(data_dir, train_ratio=0.8, random_state=42):
    """Split data by true subject IDs (not by recording/night)"""
    true_subject_map = group_by_true_subjects(data_dir)
    
    true_subjects = list(true_subject_map.keys())
    
    np.random.seed(random_state)
    np.random.shuffle(true_subjects)
    
    n_train = int(len(true_subjects) * train_ratio)
    train_true_subjects = true_subjects[:n_train]
    test_true_subjects = true_subjects[n_train:]
    
    train_recording_ids = []
    for subject in train_true_subjects:
        train_recording_ids.extend(true_subject_map[subject])
    
    test_recording_ids = []
    for subject in test_true_subjects:
        test_recording_ids.extend(true_subject_map[subject])
    
    print(f"Training on {len(train_recording_ids)} recordings from {len(train_true_subjects)} subjects")
    print(f"Testing on {len(test_recording_ids)} recordings from {len(test_true_subjects)} subjects")
    
    return train_recording_ids, test_recording_ids, train_true_subjects, test_true_subjects

class HybridSleepDataset(Dataset):
    def __init__(self, raw_data_dir, c22_data_dir, patient_ids=None):
        """
        Dataset that combines raw signal sequences with Catch22 features
        Args:
            raw_data_dir: Directory containing the preprocessed NPZ sequence files
            c22_data_dir: Directory containing the Catch22 feature CSV files
            patient_ids: List of patient IDs to include (e.g., ['SC4252', 'SC4231'])
        """
        # Get all sequence files and Catch22 files
        all_raw_files = glob.glob(os.path.join(raw_data_dir, '*_sequences.npz'))
        all_c22_files = glob.glob(os.path.join(c22_data_dir, '*_c22.csv'))
        
        # Filter by patient IDs if specified
        if patient_ids is not None:
            self.raw_files = [f for f in all_raw_files if any(pid in os.path.basename(f) for pid in patient_ids)]
            self.c22_files = [f for f in all_c22_files if any(pid in os.path.basename(f) for pid in patient_ids)]
        else:
            self.raw_files = all_raw_files
            self.c22_files = all_c22_files
        
        # Map recording IDs to file paths for easy lookup
        self.raw_file_map = {os.path.basename(f).split('_')[0]: f for f in self.raw_files}
        self.c22_file_map = {os.path.basename(f).split('_')[0]: f for f in self.c22_files}
        
        # Get common recording IDs between raw and C22 datasets
        raw_ids = set(self.raw_file_map.keys())
        c22_ids = set(self.c22_file_map.keys())
        common_ids = raw_ids.intersection(c22_ids)
        
        # Check if we found matching files
        if len(common_ids) == 0:
            raise ValueError("No matching recordings found between raw data and Catch22 features!")
            
        # Only keep files for recordings that have both raw and C22 data
        self.recording_ids = sorted(list(common_ids))
        
        # Dictionary to store metadata and indices for quick lookup
        self.recording_data = {}
        
        # Lists to store data after loading
        sequences_list = []
        c22_features_list = []
        labels_list = []
        self.patient_ids = []
        self.true_subject_ids = []
        
        total_sequences = 0
        c22_feature_dim = None
        
        # Process each recording
        for recording_id in self.recording_ids:
            # Load raw sequence data
            raw_path = self.raw_file_map[recording_id]
            raw_data = np.load(raw_path)
            sequences = raw_data['sequences']  # shape: (n_sequences, seq_len, channels, samples)
            seq_labels = raw_data['seq_labels']  # shape: (n_sequences, seq_len)
            
            # Load Catch22 features
            c22_path = self.c22_file_map[recording_id]
            c22_df = pd.read_csv(c22_path)
            
            # Get the true subject ID
            true_subject = get_true_subject_id(recording_id)
            
            # Store start and end indices for this recording
            start_idx = total_sequences
            n_sequences = sequences.shape[0]
            end_idx = start_idx + n_sequences
            
            # Store metadata for this recording
            self.recording_data[recording_id] = {
                'true_subject': true_subject,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'n_sequences': n_sequences
            }
            
            # Check if sequence length matches number of epochs in C22 features
            if n_sequences * SEQ_LENGTH != len(c22_df):
                print(f"Warning: Mismatch in recording {recording_id}!")
                print(f"  Sequences: {n_sequences} x {SEQ_LENGTH} = {n_sequences * SEQ_LENGTH}")
                print(f"  C22 epochs: {len(c22_df)}")
                
                # Try to find if this is solvable
                if len(c22_df) >= n_sequences * SEQ_LENGTH:
                    # If we have extra C22 data, keep only what we need
                    c22_df = c22_df.iloc[:n_sequences * SEQ_LENGTH]
                    print(f"  Truncated C22 data to {len(c22_df)} epochs")
                else:
                    # Skip this recording if there's a mismatch we can't fix
                    print(f"  Skipping recording {recording_id} due to data mismatch")
                    continue
            
            # Reshape C22 features to match sequence format
            c22_features = c22_df.drop(columns=['label']).values
            if c22_feature_dim is None:
                c22_feature_dim = c22_features.shape[1]
                
            # Reshape to (n_sequences, seq_length, c22_features)
            c22_features = c22_features.reshape(n_sequences, SEQ_LENGTH, -1)
            
            # Append data to lists
            sequences_list.append(sequences)
            c22_features_list.append(c22_features)
            labels_list.append(seq_labels)
            
            # Add metadata
            self.patient_ids.extend([recording_id] * n_sequences)
            self.true_subject_ids.extend([true_subject] * n_sequences)
            
            # Update total count
            total_sequences += n_sequences
        
        # Concatenate all data
        self.sequences = np.concatenate(sequences_list, axis=0)
        self.c22_features = np.concatenate(c22_features_list, axis=0)
        self.seq_labels = np.concatenate(labels_list, axis=0)
        
        # Convert to pytorch tensors
        self.sequences = torch.from_numpy(self.sequences).float()
        self.c22_features = torch.from_numpy(self.c22_features).float()
        self.seq_labels = torch.from_numpy(self.seq_labels).long()
        
        print(f"Loaded {len(self.recording_ids)} recordings with both raw sequences and Catch22 features")
        print(f"Total sequences: {len(self.sequences)}")
        print(f"Raw sequence shape: {self.sequences.shape}")
        print(f"Catch22 feature shape: {self.c22_features.shape}")
        
        # Print class distribution
        unique, counts = np.unique(self.seq_labels.numpy().flatten(), return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label} ({'W N1 N2 N3 REM'.split()[label]}): {count} samples ({count/len(self.seq_labels.flatten())*100:.2f}%)")

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.c22_features[idx], self.seq_labels[idx]

def create_balanced_sampler(dataset):
    """
    Create a weighted sampler to balance class distributions
    """
    # Get all labels (we'll use the first label of each sequence since sequences are contiguous)
    all_labels = dataset.seq_labels[:, 0].numpy()  # Only take first label of each sequence
    
    # Compute class weights
    classes = np.unique(all_labels)
    class_weights = {}
    total_samples = len(all_labels)
    for c in classes:
        class_weights[c] = float(total_samples) / (len(classes) * np.sum(all_labels == c))
    
    # Create sample weights
    sample_weights = np.array([class_weights[label] for label in all_labels])
    
    # Create sampler with length equal to dataset
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),  # Use actual dataset length
        replacement=True
    )
    
    return sampler, torch.FloatTensor([class_weights[c] for c in sorted(class_weights.keys())])

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

class EpochEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # Let's calculate the correct dimension
        self.calculate_fc_input_dim = None  # Will be set in forward pass
        self.fc = None  # Will be initialized in first forward pass
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, time_points)
        batch_size, seq_len, channels, time_points = x.shape
        
        # Process each sequence element independently
        x = x.view(batch_size * seq_len, channels, time_points)
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Initialize fc layer if not done yet
        if self.fc is None:
            self.calculate_fc_input_dim = x.shape[1] * x.shape[2]
            self.fc = nn.Linear(self.calculate_fc_input_dim, self.embedding_dim).to(x.device)
            print(f"Initialized fc layer with input dim: {self.calculate_fc_input_dim}")
        
        x = x.view(batch_size * seq_len, -1)  # Flatten
        x = self.dropout(torch.relu(self.fc(x)))
        
        # Reshape back to sequence form
        x = x.view(batch_size, seq_len, -1)
        return x

class C22Encoder(nn.Module):
    """
    Encoder for Catch22 features
    """
    def __init__(self, input_dim, embedding_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, c22_features)
        batch_size, seq_len, input_dim = x.shape
        
        # Process each sequence element independently
        x = x.view(batch_size * seq_len, input_dim)
        
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        
        # Reshape back to sequence form
        x = x.view(batch_size, seq_len, -1)
        return x

class HybridSleepTransformer(nn.Module):
    def __init__(self, c22_dim, raw_embedding_dim=128, c22_embedding_dim=64, 
                 num_classes=5, num_layers=2, num_heads=4, dropout=0.1, seq_length=20):
        super().__init__()
        
        # Encoders for different modalities
        self.epoch_encoder = EpochEncoder(raw_embedding_dim)
        self.c22_encoder = C22Encoder(c22_dim, c22_embedding_dim)
        
        # Combined embedding dimension
        self.combined_dim = raw_embedding_dim + c22_embedding_dim
        
        # Fusion layer
        self.fusion = nn.Linear(self.combined_dim, self.combined_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, self.combined_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.combined_dim,
            nhead=num_heads,
            dim_feedforward=4*self.combined_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(self.combined_dim, num_classes)
        
    def forward(self, raw_signals, c22_features):
        # raw_signals shape: (batch, seq_len, channels, time_points)
        # c22_features shape: (batch, seq_len, c22_dim)
        
        # Get embeddings for each modality
        raw_embeddings = self.epoch_encoder(raw_signals)
        c22_embeddings = self.c22_encoder(c22_features)
        
        # Concatenate embeddings along feature dimension
        combined = torch.cat([raw_embeddings, c22_embeddings], dim=2)
        
        # Apply fusion layer
        combined = torch.relu(self.fusion(combined))
        
        # Add positional encoding
        combined = combined + self.pos_encoder
        
        # Pass through transformer
        encoded = self.transformer_encoder(combined)
        
        # Get predictions for each time step
        logits = self.fc_out(encoded)
        
        return logits

def focal_loss_with_n1_focus(inputs, targets, alpha_general=0.25, alpha_n1=0.75, gamma=2):
    """
    Focal Loss with specific focus on N1 class (class index 1)
    - alpha_general: weight for all non-N1 classes
    - alpha_n1: higher weight specifically for N1 class
    - gamma: focusing parameter - same as standard focal loss
    """
    # Get class dimension
    num_classes = inputs.size(-1)
    
    # Calculate standard cross entropy (per element)
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    
    # Create a mask for N1 instances (where target == 1)
    n1_mask = (targets == 1).float()
    
    # Apply different alpha values for N1 vs other classes
    alphas = alpha_general * (1 - n1_mask) + alpha_n1 * n1_mask
    
    # Calculate the full focal loss with the appropriate alpha per sample
    loss = alphas * ((1 - pt) ** gamma) * ce_loss
    
    return loss.mean()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for raw_seq, c22_seq, labels in dataloader:
        raw_seq, c22_seq, labels = raw_seq.to(device), c22_seq.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(raw_seq, c22_seq)
        loss = criterion(logits.view(-1, 5), labels.view(-1))  # 5 classes
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * raw_seq.size(0)
        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu().detach().numpy())
        all_labels.append(labels.cpu().detach().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    acc = (all_preds == all_labels).mean()
    
    return epoch_loss, acc

def eval_epoch(model, dataloader, criterion, device):
    """Evaluate model on validation data"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for raw_seq, c22_seq, labels in dataloader:
            raw_seq, c22_seq, labels = raw_seq.to(device), c22_seq.to(device), labels.to(device)
            logits = model(raw_seq, c22_seq)
            loss = criterion(logits.view(-1, 5), labels.view(-1))  # 5 classes
            
            running_loss += loss.item() * raw_seq.size(0)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_probs = np.concatenate(all_probs).reshape(-1, 5)  # 5 classes
    acc = (all_preds == all_labels).mean()
    
    return epoch_loss, acc, all_preds, all_labels, all_probs

def subject_based_kfold_cv(data_dir, n_folds=5, random_state=42):
    """Perform k-fold cross-validation with subject-based splitting"""
    # Get subject mapping
    true_subject_map = group_by_true_subjects(data_dir)
    true_subjects = list(true_subject_map.keys())
    
    # Shuffle subjects
    np.random.seed(random_state)
    np.random.shuffle(true_subjects)
    
    # Create folds
    subject_folds = np.array_split(true_subjects, n_folds)
    
    # For each fold
    results = []
    for fold_idx in range(n_folds):
        # Use current fold as test set
        test_subjects = subject_folds[fold_idx]
        # Use all other folds as train set
        train_subjects = [s for i, fold in enumerate(subject_folds) if i != fold_idx for s in fold]
        
        # Get recording IDs for train and test
        train_recordings = []
        for subject in train_subjects:
            train_recordings.extend(true_subject_map[subject])
        
        test_recordings = []
        for subject in test_subjects:
            test_recordings.extend(true_subject_map[subject])
        
        fold_data = {
            'fold': fold_idx,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
            'train_recordings': train_recordings,
            'test_recordings': test_recordings
        }
        results.append(fold_data)
    
    return results

def plot_curves(train_losses, test_losses, train_accs, test_accs):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss Curves')
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(test_accs, label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'training_curves.png'))
    plt.close()

def median_smoothing(predictions, kernel_size=3):
    """Apply median smoothing to predictions"""
    return median_filter(predictions, size=kernel_size)

def compute_metrics(y_true, y_pred, class_names=None):
    """Compute classification metrics"""
    if class_names is None:
        class_names = ["W", "N1", "N2", "N3", "REM"]
        
    # Overall metrics
    accuracy = (y_true == y_pred).mean()
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Extract F1 scores for each class
    f1_scores = {}
    for i, name in enumerate(class_names):
        f1_scores[name] = report[name]['f1-score']
    
    # Overall F1 scores
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_scores': f1_scores,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'classification_report': report,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", normalize=False, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_to_plot = cm_norm
        fmt = '.2f'
    else:
        cm_to_plot = cm
        fmt = 'd'
    
    plt.imshow(cm_to_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm_to_plot.max() / 2.
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            plt.text(j, i, format(cm_to_plot[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_to_plot[i, j] > thresh else "black")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(RESULTS_DIR, f'training_log_{timestamp}.txt')
    
    # Set up logging to file
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        with open(log_file, 'a') as f:
            print(*args, file=f, **kwargs)
    
    log_print(f"Starting hybrid model training at {timestamp}")
    log_print(f"Using device: {device}")
    
    # Implement k-fold cross-validation
    n_folds = 5
    log_print(f"\n=== Performing {n_folds}-fold Cross-Validation ===")
    
    # Get subject mapping
    true_subject_map = group_by_true_subjects(PROCESSED_DATA_DIR)
    true_subjects = list(true_subject_map.keys())
    
    # Shuffle subjects
    np.random.seed(SEED)
    np.random.shuffle(true_subjects)
    
    # Create folds
    subject_folds = np.array_split(true_subjects, n_folds)
    
    # Define class names
    class_names = ["Wake", "N1", "N2", "N3", "REM"]
    
    # Track results across folds
    fold_results = []
    
    # Create directories for each fold's results
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx+1}")
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, "metrics"), exist_ok=True)
    
    # For each fold
    for fold_idx in range(n_folds):
        fold_dir = os.path.join(RESULTS_DIR, f"fold_{fold_idx+1}")
        
        log_print(f"\n\n{'='*50}")
        log_print(f"=== FOLD {fold_idx+1}/{n_folds} ===")
        log_print(f"{'='*50}\n")
        
        # Use current fold as test set
        test_subjects = subject_folds[fold_idx]
        # Use all other folds as train set
        train_subjects = [s for i, fold in enumerate(subject_folds) if i != fold_idx for s in fold]
        
        log_print(f"Train subjects: {len(train_subjects)}")
        log_print(f"Test subjects: {len(test_subjects)}")
        
        # Get recording IDs for train and test
        train_patients = []
        for subject in train_subjects:
            train_patients.extend(true_subject_map[subject])
        
        test_patients = []
        for subject in test_subjects:
            test_patients.extend(true_subject_map[subject])
        
        log_print(f"Train recordings: {len(train_patients)}")
        log_print(f"Test recordings: {len(test_patients)}")
        
        # Save fold split information
        split_info = {
            'fold': fold_idx,
            'train_patients': train_patients,
            'test_patients': test_patients,
            'train_subjects': train_subjects,
            'test_subjects': test_subjects,
        }
        
        with open(os.path.join(fold_dir, 'metrics', 'subject_split.txt'), 'w') as f:
            f.write(f"Fold {fold_idx+1}/{n_folds}\n\n")
            f.write("Train subjects:\n")
            f.write(", ".join(train_subjects))
            f.write("\n\nTest subjects:\n")
            f.write(", ".join(test_subjects))
        
        # Create hybrid datasets with both raw sequences and Catch22 features
        log_print("\n=== Loading Datasets ===")
        train_dataset = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, patient_ids=train_patients)
        test_dataset = HybridSleepDataset(PROCESSED_DATA_DIR, CATCH22_DATA_DIR, patient_ids=test_patients)
        
        # Create balanced sampler and get class weights
        train_sampler, class_weights = create_balanced_sampler(train_dataset)
        
        # Create data loaders
        log_print("\n=== Creating DataLoaders ===")
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        log_print(f"Train loader: {len(train_loader)} batches, {len(train_dataset)} samples")
        log_print(f"Test loader: {len(test_loader)} batches, {len(test_dataset)} samples")
        
        # Get C22 feature dimension
        c22_dim = train_dataset.c22_features.shape[2]
        log_print(f"Catch22 feature dimension: {c22_dim}")
        
        # Initialize model
        log_print("\n=== Building Model ===")
        model = HybridSleepTransformer(
            c22_dim=c22_dim,
            raw_embedding_dim=128,
            c22_embedding_dim=64,
            num_classes=5,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            seq_length=SEQ_LENGTH
        )
        model.to(device)
        
        # Only log model architecture for the first fold
        if fold_idx == 0:
            log_print("Model Architecture:")
            log_print(str(model))
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_print(f"Total parameters: {total_params:,}")
        log_print(f"Trainable parameters: {trainable_params:,}")
        
        # Define loss function and optimizer
        log_print("\n=== Setting Up Training ===")
        class_weights = class_weights.to(device)
        criterion = lambda x, y: focal_loss_with_n1_focus(x, y, alpha_general=0.25, alpha_n1=0.75, gamma=2)
        
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        log_print("\n=== Starting Training ===")
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(NUM_EPOCHS):
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Evaluate
            val_loss, val_acc, val_preds, val_labels, val_probs = eval_epoch(
                model, test_loader, criterion, device
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save metrics
            train_losses.append(train_loss)
            test_losses.append(val_loss)
            train_accs.append(train_acc)
            test_accs.append(val_acc)
            
            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                best_epoch = epoch
                
                # Calculate metrics for best model so far
                metrics = compute_metrics(val_labels, val_preds, class_names)
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': val_acc,
                    'metrics': metrics,
                }, os.path.join(fold_dir, 'models', 'best_model.pth'))
                
                # Save confusion matrix for best model
                plot_confusion_matrix(
                    metrics['confusion_matrix'], 
                    class_names, 
                    title=f"Fold {fold_idx+1} - Confusion Matrix (Epoch {epoch+1})",
                    save_path=os.path.join(fold_dir, 'plots', f'confusion_matrix_epoch_{epoch+1}.png')
                )
                
                # Also save normalized version
                plot_confusion_matrix(
                    metrics['confusion_matrix'], 
                    class_names, 
                    title=f"Fold {fold_idx+1} - Normalized Confusion Matrix (Epoch {epoch+1})",
                    normalize=True,
                    save_path=os.path.join(fold_dir, 'plots', f'norm_confusion_matrix_epoch_{epoch+1}.png')
                )
            
            log_print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            log_print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            log_print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Print detailed metrics every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
                metrics = compute_metrics(val_labels, val_preds, class_names)
                
                log_print("\nDetailed Metrics:")
                log_print(f"Accuracy: {metrics['accuracy']:.4f}")
                log_print(f"Macro F1: {metrics['macro_f1']:.4f}")
                log_print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
                
                log_print("\nPer-class F1 scores:")
                for cls, f1 in metrics['f1_scores'].items():
                    log_print(f"  {cls}: {f1:.4f}")
                
                # Also try median smoothing
                smoothed_preds = median_smoothing(val_preds, kernel_size=5)
                smoothed_metrics = compute_metrics(val_labels, smoothed_preds, class_names)
                
                log_print("\nWith median smoothing (kernel=5):")
                log_print(f"Accuracy: {smoothed_metrics['accuracy']:.4f}")
                log_print(f"Macro F1: {smoothed_metrics['macro_f1']:.4f}")
                
                log_print("\nPer-class F1 scores (smoothed):")
                for cls, f1 in smoothed_metrics['f1_scores'].items():
                    log_print(f"  {cls}: {f1:.4f}")
        
        # Save training curves
        plot_curves(train_losses, test_losses, train_accs, test_accs)
        plt.savefig(os.path.join(fold_dir, 'plots', 'training_curves.png'))
        
        # Final evaluation with best model
        log_print("\n=== Final Evaluation for Fold ===")
        log_print(f"Loading best model from epoch {best_epoch+1}")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        test_loss, test_acc, test_preds, test_labels, test_probs = eval_epoch(
            model, test_loader, criterion, device
        )
        
        # Calculate metrics
        metrics = compute_metrics(test_labels, test_preds, class_names)
        
        # Also apply median smoothing
        smoothed_preds = median_smoothing(test_preds, kernel_size=5)
        smoothed_metrics = compute_metrics(test_labels, smoothed_preds, class_names)
        
        # Save final confusion matrices
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            class_names, 
            title=f"Fold {fold_idx+1} - Final Confusion Matrix",
            save_path=os.path.join(fold_dir, 'plots', 'final_confusion_matrix.png')
        )
        
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            class_names, 
            title=f"Fold {fold_idx+1} - Normalized Final Confusion Matrix",
            normalize=True,
            save_path=os.path.join(fold_dir, 'plots', 'final_norm_confusion_matrix.png')
        )
        
        plot_confusion_matrix(
            smoothed_metrics['confusion_matrix'], 
            class_names, 
            title=f"Fold {fold_idx+1} - Final Confusion Matrix (Smoothed)",
            save_path=os.path.join(fold_dir, 'plots', 'final_smoothed_confusion_matrix.png')
        )
        
        # Log detailed final results
        log_print("\n=== FOLD RESULTS ===")
        log_print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        log_print(f"Test Macro F1: {metrics['macro_f1']:.4f}")
        log_print(f"Test Weighted F1: {metrics['weighted_f1']:.4f}")
        
        log_print("\nPer-class F1 scores:")
        for cls, f1 in metrics['f1_scores'].items():
            log_print(f"  {cls}: {f1:.4f}")
        
        log_print("\nClassification Report:")
        log_print(classification_report(test_labels, test_preds, target_names=class_names))
        
        log_print("\nWith median smoothing (kernel=5):")
        log_print(f"Smoothed Accuracy: {smoothed_metrics['accuracy']:.4f}")
        log_print(f"Smoothed Macro F1: {smoothed_metrics['macro_f1']:.4f}")
        
        log_print("\nPer-class F1 scores (smoothed):")
        for cls, f1 in smoothed_metrics['f1_scores'].items():
            log_print(f"  {cls}: {f1:.4f}")
        
        # Save fold results
        fold_results.append({
            'fold': fold_idx,
            'metrics': metrics,
            'smoothed_metrics': smoothed_metrics,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'best_epoch': best_epoch,
            'predictions': test_preds,
            'smoothed_predictions': smoothed_preds,
            'true_labels': test_labels,
            'subject_split': split_info
        })
        
        # Save fold metrics
        joblib.dump(fold_results[-1], os.path.join(fold_dir, 'metrics', 'fold_results.pkl'))
    
    # Compute cross-validation summary
    log_print("\n\n" + "="*50)
    log_print("=== CROSS-VALIDATION SUMMARY ===")
    log_print("="*50 + "\n")
    
    # Average metrics across folds
    avg_accuracy = np.mean([r['metrics']['accuracy'] for r in fold_results])
    avg_macro_f1 = np.mean([r['metrics']['macro_f1'] for r in fold_results])
    avg_weighted_f1 = np.mean([r['metrics']['weighted_f1'] for r in fold_results])
    
    # Standard deviation of metrics
    std_accuracy = np.std([r['metrics']['accuracy'] for r in fold_results])
    std_macro_f1 = np.std([r['metrics']['macro_f1'] for r in fold_results])
    std_weighted_f1 = np.std([r['metrics']['weighted_f1'] for r in fold_results])
    
    # Smoothed metrics
    avg_smoothed_accuracy = np.mean([r['smoothed_metrics']['accuracy'] for r in fold_results])
    avg_smoothed_macro_f1 = np.mean([r['smoothed_metrics']['macro_f1'] for r in fold_results])
    std_smoothed_accuracy = np.std([r['smoothed_metrics']['accuracy'] for r in fold_results])
    std_smoothed_macro_f1 = np.std([r['smoothed_metrics']['macro_f1'] for r in fold_results])
    
    log_print("Raw Predictions:")
    log_print(f"Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    log_print(f"Average Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
    log_print(f"Average Weighted F1: {avg_weighted_f1:.4f} ± {std_weighted_f1:.4f}")
    
    log_print("\nWith Median Smoothing:")
    log_print(f"Average Accuracy: {avg_smoothed_accuracy:.4f} ± {std_smoothed_accuracy:.4f}")
    log_print(f"Average Macro F1: {avg_smoothed_macro_f1:.4f} ± {std_smoothed_macro_f1:.4f}")
    
    # Per-class F1 scores across folds
    log_print("\nPer-class F1 scores (averaged across folds):")
    for cls in class_names:
        f1_values = [r['metrics']['f1_scores'][cls] for r in fold_results]
        avg_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        log_print(f"  {cls}: {avg_f1:.4f} ± {std_f1:.4f}")
    
    log_print("\nPer-class F1 scores with smoothing (averaged across folds):")
    for cls in class_names:
        f1_values = [r['smoothed_metrics']['f1_scores'][cls] for r in fold_results]
        avg_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        log_print(f"  {cls}: {avg_f1:.4f} ± {std_f1:.4f}")
    
    # Save CV summary
    cv_summary = {
        'fold_results': fold_results,
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'avg_macro_f1': avg_macro_f1,
        'std_macro_f1': std_macro_f1,
        'avg_weighted_f1': avg_weighted_f1,
        'std_weighted_f1': std_weighted_f1,
        'avg_smoothed_accuracy': avg_smoothed_accuracy,
        'std_smoothed_accuracy': std_smoothed_accuracy,
        'avg_smoothed_macro_f1': avg_smoothed_macro_f1,
        'std_smoothed_macro_f1': std_smoothed_macro_f1
    }
    
    joblib.dump(cv_summary, os.path.join(RESULTS_DIR, 'metrics', 'cv_summary.pkl'))
    
    # Save a comprehensive text summary
    with open(os.path.join(RESULTS_DIR, 'metrics', 'cv_summary.txt'), 'w') as f:
        f.write("Hybrid CNN-Transformer with Catch22 Features - Cross-Validation Results\n")
        f.write("=================================================================\n\n")
        f.write(f"Cross-validation completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of folds: {n_folds}\n\n")
        
        f.write("Average Performance Metrics:\n")
        f.write(f"- Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        f.write(f"- Macro F1: {avg_macro_f1:.4f} ± {std_macro_f1:.4f}\n")
        f.write(f"- Weighted F1: {avg_weighted_f1:.4f} ± {std_weighted_f1:.4f}\n\n")
        
        f.write("With Median Smoothing:\n")
        f.write(f"- Accuracy: {avg_smoothed_accuracy:.4f} ± {std_smoothed_accuracy:.4f}\n")
        f.write(f"- Macro F1: {avg_smoothed_macro_f1:.4f} ± {std_smoothed_macro_f1:.4f}\n\n")
        
        f.write("Per-class F1 scores:\n")
        for cls in class_names:
            f1_values = [r['metrics']['f1_scores'][cls] for r in fold_results]
            avg_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values)
            f.write(f"  {cls}: {avg_f1:.4f} ± {std_f1:.4f}\n")
        
        f.write("\nPer-class F1 scores with smoothing:\n")
        for cls in class_names:
            f1_values = [r['smoothed_metrics']['f1_scores'][cls] for r in fold_results]
            avg_f1 = np.mean(f1_values)
            std_f1 = np.std(f1_values)
            f.write(f"  {cls}: {avg_f1:.4f} ± {std_f1:.4f}\n")
        
        f.write("\nResults by fold:\n")
        for i, fold in enumerate(fold_results):
            f.write(f"\nFold {i+1}:\n")
            f.write(f"  Accuracy: {fold['metrics']['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {fold['metrics']['macro_f1']:.4f}\n")
            f.write(f"  Best epoch: {fold['best_epoch']+1}\n")
            f.write(f"  Test subjects: {len(fold['subject_split']['test_subjects'])}\n")
    
    log_print("\nCross-validation complete!")
    log_print(f"Results saved to: {RESULTS_DIR}")