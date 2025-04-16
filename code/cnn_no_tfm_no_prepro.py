#!/usr/bin/env python3
import os, sys, glob, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from sklearn.metrics import classification_report, confusion_matrix

# PROCESSED_DATA_DIR = '/content/drive/MyDrive/spring_2025/STAT_4830/4830_project/sleepedf_data/processed_sleepedf'
PROCESSED_DATA_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/processed_sleepedf'

# Model parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 20

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
    
    return train_recording_ids, test_recording_ids

class SleepSequenceDataset(Dataset):
    def __init__(self, data_dir, patient_ids=None):
        """
        Args:
            data_dir: Directory containing the preprocessed NPZ files
            patient_ids: List of patient IDs to include (e.g., ['SC4252', 'SC4231'])
        """
        # Get all sequence files
        all_files = glob.glob(os.path.join(data_dir, '*_sequences.npz'))
        
        # Filter by patient IDs if specified
        if patient_ids is not None:
            self.files = [f for f in all_files if any(pid in os.path.basename(f) for pid in patient_ids)]
        else:
            self.files = all_files
            
        sequences_list = []
        labels_list = []
        self.patient_ids = []
        self.true_subject_ids = []
        
        for f in self.files:
            loaded = np.load(f)
            sequences_list.append(loaded['sequences'])
            labels_list.append(loaded['seq_labels'])
            recording_id = os.path.basename(f).split('_')[0]
            true_subject = get_true_subject_id(f)
            
            self.patient_ids.append(recording_id)
            self.true_subject_ids.append(true_subject)
            
        self.sequences = np.concatenate(sequences_list, axis=0)
        self.seq_labels = np.concatenate(labels_list, axis=0)
        self.sequences = torch.from_numpy(self.sequences).float()
        self.seq_labels = torch.from_numpy(self.seq_labels).long()
        
        print(f"Loaded {len(self.files)} files from {len(set(self.true_subject_ids))} unique subjects")
        print(f"Total sequences: {len(self.sequences)}")
        
        # Print class distribution
        unique, counts = np.unique(self.seq_labels.numpy().flatten(), return_counts=True)
        print("\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"Class {label} ({'W N1 N2 N3 REM'.split()[label]}): {count} samples")

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx]

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

class SleepTransformer(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=5, num_layers=2, num_heads=4, dropout=0.1, seq_length=20):
        super().__init__()
        
        self.epoch_encoder = EpochEncoder(embedding_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_length, embedding_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4*embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, time_points)
        
        # Get embeddings for each epoch
        x = self.epoch_encoder(x)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get predictions for each time step
        x = self.fc_out(x)
        
        return x

def focal_loss(inputs, targets, alpha=0.25, gamma=2):
    """
    Focal Loss for handling class imbalance
    """
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * ((1 - pt) ** gamma) * ce_loss
    return loss.mean()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for seq, labels in dataloader:
        seq, labels = seq.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seq)
        loss = criterion(logits.view(-1, 5), labels.view(-1))  # 5 classes
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * seq.size(0)
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
        for seq, labels in dataloader:
            seq, labels = seq.to(device), labels.to(device)
            logits = model(seq)
            loss = criterion(logits.view(-1, 5), labels.view(-1))  # 5 classes
            
            running_loss += loss.item() * seq.size(0)
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
    plt.savefig('training_curves.png')
    plt.close()

def median_smoothing(predictions, kernel_size=3):
    """Apply median smoothing to predictions"""
    return median_filter(predictions, size=kernel_size)

def compute_metrics(y_true, y_pred):
    """Compute classification metrics"""
    return {
        'classification_report': classification_report(y_true, y_pred, target_names=["W", "N1", "N2", "N3", "REM"]),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, ["W", "N1", "N2", "N3", "REM"])
    plt.yticks(tick_marks, ["W", "N1", "N2", "N3", "REM"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Split patients into train and test sets
    train_patients, test_patients = split_true_subjects(PROCESSED_DATA_DIR, train_ratio=TRAIN_RATIO)
    
    # Create datasets with specific patients
    train_dataset = SleepSequenceDataset(PROCESSED_DATA_DIR, patient_ids=train_patients)
    test_dataset = SleepSequenceDataset(PROCESSED_DATA_DIR, patient_ids=test_patients)
    
    # Create balanced sampler and get class weights
    train_sampler, class_weights = create_balanced_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Initialize model
    model = SleepTransformer(
        embedding_dim=128,
        num_classes=5,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        seq_length=SEQ_LENGTH
    )
    model.to(device)
    
    # Use focal loss with class weights
    class_weights = class_weights.to(device)
    criterion = lambda x, y: focal_loss(x, y, alpha=0.25, gamma=2)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_val_loss = float('inf')
    best_model_state = None
    
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
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Print detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            metrics = compute_metrics(val_labels, val_preds)
            print("\nDetailed Metrics:")
            print(metrics['classification_report'])
            
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    val_loss, val_acc, val_preds, val_labels, val_probs = eval_epoch(
        model, test_loader, criterion, device
    )
    
    # Apply median smoothing to final predictions
    smoothed_preds = median_smoothing(val_preds, kernel_size=5)
    
    # Compute and print final metrics
    print("\nFinal Metrics (with smoothing):")
    final_metrics = compute_metrics(val_labels, smoothed_preds)
    print(final_metrics['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(final_metrics['confusion_matrix'])
    
    # Plot training curves
    plot_curves(train_losses, test_losses, train_accs, test_accs)
    
    # Save the model
    torch.save({
        'model_state_dict': best_model_state,
        'train_patients': train_patients,
        'test_patients': test_patients,
        'final_metrics': final_metrics,
    }, 'sleep_transformer_model.pth')
    
    print("Training complete. Model and results saved.")

if __name__ == "__main__":
    main()