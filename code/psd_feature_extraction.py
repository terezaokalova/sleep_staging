#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from scipy import signal
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt

from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# 1) Root of all your project data
DATA_ROOT = Path.home() / 'spring_2025' / 'STAT_4830' / 'STAT-4830-GOALZ-project' / 'data'

# 2) Raw EDF files live here:
BASE_DIR = DATA_ROOT / 'sleep-edf-database-expanded-1.0.0'

# 3) Your preprocessing must have written epoch .npz files here:
PROCESSED_DIR = DATA_ROOT / 'processed_sleepedf'

# 4) We'll dump PSD features & plots here:
OUTPUT_DIR = DATA_ROOT / 'psd_features_sleepedf'

# 5) Sanity‐check that the input directories actually exist:
if not BASE_DIR.exists():
    raise FileNotFoundError(f"Raw EDF folder not found: {BASE_DIR}")
if not PROCESSED_DIR.exists():
    raise FileNotFoundError(f"Preprocessed epochs folder not found: {PROCESSED_DIR}")

# 6) Create the output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'plots').mkdir(exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────────

# Sampling rate for Sleep-EDF is 100Hz
FS = 100

# sanity checks
if not BASE_DIR.exists():
    raise FileNotFoundError(f"Raw EDF folder not found: {BASE_DIR}")
if not PROCESSED_DIR.exists():
    raise FileNotFoundError(f"Preprocessed epochs folder not found: {PROCESSED_DIR}")

# 5) create output dirs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'plots').mkdir(exist_ok=True)


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'plots'), exist_ok=True)


# Define frequency bands of interest
FREQ_BANDS = {
    'delta': (0.5, 4),   # Delta: 0.5-4 Hz
    'theta': (4, 8),     # Theta: 4-8 Hz
    'alpha': (8, 13),    # Alpha: 8-13 Hz
    'sigma': (12, 16),   # Sigma/Sleep spindles: 12-16 Hz
    'beta': (16, 30),    # Beta: 16-30 Hz
    'gamma': (30, 49)    # Gamma: 30-49 Hz (limited by Nyquist freq)
}

# Additional parameters
WELCH_WINDOW = 4 * FS  # 4-second window
WELCH_OVERLAP = 2 * FS  # 2-second overlap (50% overlap)
NFFT = 8 * FS  # 8-second segment for FFT
    
def get_true_subject_id(filename):
    """Extract true subject ID ignoring the night number."""
    basename = os.path.basename(filename).split('_')[0]
    if basename.startswith('SC4'):
        return basename[:5]  # SC4xx - first 5 chars for Sleep Cassette
    elif basename.startswith('ST7'):
        return basename[:5]  # ST7xx - first 5 chars for Sleep Telemetry
    else:
        return basename[:6]  # Fallback to original logic

def apply_bandpass_filter(data, lowcut=0.5, highcut=45, fs=100, order=5):
    """Apply bandpass filter to the data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def compute_psd(data, fs=100):
    """
    Compute power spectral density using Welch's method.
    
    Args:
        data: np.ndarray, shape (n_samples,)
        fs: int, sampling frequency
        
    Returns:
        freqs: np.ndarray, frequencies
        psd: np.ndarray, power spectral density
    """
    freqs, psd = signal.welch(data, fs=fs, nperseg=WELCH_WINDOW, 
                             noverlap=WELCH_OVERLAP, nfft=NFFT)
    return freqs, psd

def extract_band_power(freqs, psd, freq_range):
    """
    Extract power in a specific frequency band.
    
    Args:
        freqs: np.ndarray, frequencies
        psd: np.ndarray, power spectral density
        freq_range: tuple, (low_freq, high_freq)
        
    Returns:
        float: Power in the specified band
    """
    low, high = freq_range
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx])

def compute_spectral_entropy(psd):
    """Compute spectral entropy."""
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # Avoid log(0)
    return -np.sum(psd_norm * np.log2(psd_norm))

def compute_spectral_edge_frequency(freqs, psd, percentage=0.95):
    """Compute spectral edge frequency."""
    total_power = np.sum(psd)
    target_power = total_power * percentage
    cumulative_power = np.cumsum(psd)
    idx = np.argmax(cumulative_power >= target_power)
    return freqs[idx]

def extract_psd_features(data_chunk, channel_idx=0, include_spectral_metrics=True, plot=False, recording_id=None, epoch_idx=None):
    """
    Extract power spectral density features from one epoch.
    
    Args:
        data_chunk: np.ndarray, shape (n_channels, n_samples)
        channel_idx: int, which channel to use (0=EEG, 1=EOG)
        include_spectral_metrics: bool, include additional spectral metrics
        plot: bool, whether to generate and save plots
        recording_id: str, recording ID for plot title
        epoch_idx: int, epoch index for plot title
        
    Returns:
        dict: Features extracted from the epoch
    """
    channel_data = data_chunk[channel_idx, :]
    
    # Apply bandpass filter (0.5-45 Hz)
    filtered_data = apply_bandpass_filter(channel_data)
    
    # Compute PSD
    freqs, psd = compute_psd(filtered_data, fs=FS)
    
    # Extract absolute band powers
    band_powers = {}
    for band_name, freq_range in FREQ_BANDS.items():
        band_powers[f'abs_{band_name}'] = extract_band_power(freqs, psd, freq_range)
    
    # Compute total power (0.5-45 Hz)
    total_power = extract_band_power(freqs, psd, (0.5, 45))
    
    # Extract relative band powers
    for band_name, freq_range in FREQ_BANDS.items():
        band_powers[f'rel_{band_name}'] = band_powers[f'abs_{band_name}'] / total_power if total_power > 0 else 0
    
    # Calculate band power ratios
    band_powers['theta_alpha_ratio'] = band_powers['abs_theta'] / band_powers['abs_alpha'] if band_powers['abs_alpha'] > 0 else 0
    band_powers['delta_beta_ratio'] = band_powers['abs_delta'] / band_powers['abs_beta'] if band_powers['abs_beta'] > 0 else 0
    band_powers['theta_beta_ratio'] = band_powers['abs_theta'] / band_powers['abs_beta'] if band_powers['abs_beta'] > 0 else 0
    band_powers['delta_theta_ratio'] = band_powers['abs_delta'] / band_powers['abs_theta'] if band_powers['abs_theta'] > 0 else 0
    
    # Additional spectral metrics
    if include_spectral_metrics:
        # Spectral entropy
        band_powers['spectral_entropy'] = compute_spectral_entropy(psd)
        
        # Spectral edge frequencies
        band_powers['spectral_edge_50'] = compute_spectral_edge_frequency(freqs, psd, 0.5)
        band_powers['spectral_edge_90'] = compute_spectral_edge_frequency(freqs, psd, 0.9)
        band_powers['spectral_edge_95'] = compute_spectral_edge_frequency(freqs, psd, 0.95)
        
        # Spectral moments
        # Mean frequency
        band_powers['spectral_mean_freq'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        # Variance of frequency
        mean_freq = band_powers['spectral_mean_freq']
        band_powers['spectral_var_freq'] = np.sum(((freqs - mean_freq) ** 2) * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        # Spectral skewness
        if band_powers['spectral_var_freq'] > 0:
            std_freq = np.sqrt(band_powers['spectral_var_freq'])
            band_powers['spectral_skew_freq'] = np.sum(((freqs - mean_freq) ** 3) * psd) / (np.sum(psd) * (std_freq ** 3)) if std_freq > 0 else 0
        else:
            band_powers['spectral_skew_freq'] = 0
        
    # Generate PSD plot if requested
    if plot and recording_id and epoch_idx is not None:
        plt.figure(figsize=(10, 6))
        plt.semilogy(freqs, psd)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.title(f'PSD - {recording_id} - Epoch {epoch_idx} - {"EEG" if channel_idx==0 else "EOG"}')
        plt.grid(True)
        
        # Add colored bands
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'magenta']
        for (band_name, freq_range), color in zip(FREQ_BANDS.items(), colors):
            plt.axvspan(freq_range[0], freq_range[1], alpha=0.3, color=color, label=band_name)
        
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(OUTPUT_DIR, 'plots', f'{recording_id}_epoch{epoch_idx:04d}_{channel_idx}.png'))
        plt.close()
    
    return band_powers

def process_file(file_path, plot_samples=False, max_plots=5):
    """Process one epochs-NPZ file and save its PSD features."""
    start_time = time.time()
    try:
        data = np.load(file_path)
        epochs = data['data']     # (n_epochs, n_channels, n_samples)
        labels = data['labels']   # (n_epochs,)

        n_epochs, n_channels, _ = epochs.shape
        
        # Choose random epochs to plot (if requested)
        if plot_samples and n_epochs > 0:
            n_plots = min(max_plots, n_epochs)
            plot_indices = np.random.choice(n_epochs, n_plots, replace=False)
        else:
            plot_indices = []
        
        recording_id = os.path.basename(file_path).replace('_epochs.npz', '')
        
        eeg_features = []
        eog_features = []
        
        batch_size = 100
        for i in range(0, n_epochs, batch_size):
            batch = epochs[i:i+batch_size]
            batch_indices = list(range(i, min(i+batch_size, n_epochs)))
            
            # Process EEG (channel 0)
            eeg_batch = Parallel(n_jobs=-1)(
                delayed(extract_psd_features)(
                    batch[j-i], 
                    channel_idx=0,
                    plot=(j in plot_indices),
                    recording_id=recording_id,
                    epoch_idx=j
                )
                for j in batch_indices
            )
            eeg_features.extend(eeg_batch)
            
            # Process EOG (channel 1)
            eog_batch = Parallel(n_jobs=-1)(
                delayed(extract_psd_features)(
                    batch[j-i], 
                    channel_idx=1,
                    plot=(j in plot_indices),
                    recording_id=recording_id,
                    epoch_idx=j
                )
                for j in batch_indices
            )
            eog_features.extend(eog_batch)
            
        # Convert to DataFrame
        eeg_df = pd.DataFrame(eeg_features)
        eeg_df.columns = [f'eeg_{col}' for col in eeg_df.columns]
        
        eog_df = pd.DataFrame(eog_features)
        eog_df.columns = [f'eog_{col}' for col in eog_df.columns]
        
        # Combine features
        all_df = pd.concat([eeg_df, eog_df], axis=1)
        all_df['label'] = labels
        
        # Extract subject ID
        subj_id = get_true_subject_id(recording_id)
        
        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{recording_id}_psd.csv")
        all_df.to_csv(csv_path, index=False)
        
        # Also save as compressed NPZ
        features = all_df.drop(columns=['label']).values.astype('float32')
        npz_path = os.path.join(OUTPUT_DIR, f"{recording_id}_psd.npz")
        np.savez_compressed(
            npz_path,
            features=features,
            labels=labels.astype('int8'),
            subject_id=subj_id,
            recording_id=recording_id
        )
        
        elapsed = time.time() - start_time
        print(f"Processed {recording_id} (subj {subj_id}): {n_epochs} epochs in {elapsed:.1f}s")
        
        return subj_id, recording_id
        
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None, None

def main():
    # Find all epoch files
    epoch_files = glob.glob(os.path.join(PROCESSED_DIR, '*_epochs.npz'))
    print(f"Found {len(epoch_files)} files.")
    
    if not epoch_files:
        print("No epoch files—run preprocessing first.")
        print(f"Looked in: {PROCESSED_DIR}")
        print("Checking if directory exists:", os.path.exists(PROCESSED_DIR))
        print("Files in directory:", os.listdir(PROCESSED_DIR) if os.path.exists(PROCESSED_DIR) else "Directory doesn't exist")
        return
    
    # Group by subject
    subj_map = {}
    for fp in epoch_files:
        rec = os.path.basename(fp).replace('_epochs.npz', '')
        subj = get_true_subject_id(rec)
        subj_map.setdefault(subj, []).append(fp)
    
    print(f"{len(subj_map)} unique subjects.")
    
    # Process in parallel
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() // 2)
    print(f"Using {n_jobs} parallel jobs for file processing.")
    
    # Process first file to generate sample plots for verification
    if len(epoch_files) > 0:
        print("Processing first file with sample plots...")
        process_file(epoch_files[0], plot_samples=True, max_plots=5)
    
    # Process remaining files
    remaining_files = epoch_files[1:] if len(epoch_files) > 1 else []
    
    if remaining_files:
        print(f"Processing {len(remaining_files)} remaining files...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_file)(f)
            for f in remaining_files
        )
        # Filter out failed results
        results = [r for r in results if r[0] is not None]
        
        # Add the first file's result if it was processed
        if len(epoch_files) > 0:
            first_rec = os.path.basename(epoch_files[0]).replace('_epochs.npz', '')
            first_subj = get_true_subject_id(first_rec)
            results.append((first_subj, first_rec))
    
        # Create the mapping file
        mapping = [(s, r) for s, r in results]
        map_df = pd.DataFrame(mapping, columns=['subject_id', 'recording_id'])
        map_df.to_csv(os.path.join(OUTPUT_DIR, 'subject_recording_mapping.csv'), index=False)
    
    # Generate feature distribution plots
    print("Generating feature distribution plots...")
    generate_feature_distribution_plots()
    
    print("All done!")

def generate_feature_distribution_plots():
    """Generate plots of feature distributions across all files."""
    # Get all CSV files
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*_psd.csv'))
    
    if not csv_files:
        print("No processed files found for plotting.")
        return
    
    # Sample a subset of files (max 10) to avoid memory issues
    if len(csv_files) > 10:
        csv_files = np.random.choice(csv_files, 10, replace=False)
    
    # Load and concatenate data
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Map labels to sleep stages
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    combined_df['sleep_stage'] = combined_df['label'].map(lambda x: stage_names[x])
    
    # Create plot directory
    os.makedirs(os.path.join(OUTPUT_DIR, 'distribution_plots'), exist_ok=True)
    
    # Plot relative band powers by sleep stage for EEG
    rel_band_cols = [col for col in combined_df.columns if col.startswith('eeg_rel_')]
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(rel_band_cols):
        plt.subplot(2, 3, i+1)
        # Use boxplot instead of seaborn if not available
        plt.boxplot([combined_df[combined_df['sleep_stage']==stage][col].dropna() for stage in stage_names], 
                   labels=stage_names)
        plt.title(col.replace('eeg_rel_', 'EEG Relative ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'eeg_relative_bands.png'))
    plt.close()
    
    # Plot band ratios for EEG
    ratio_cols = [col for col in combined_df.columns if 'ratio' in col and col.startswith('eeg_')]
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(ratio_cols):
        plt.subplot(2, 2, i+1)
        plt.boxplot([combined_df[combined_df['sleep_stage']==stage][col].dropna() for stage in stage_names], 
                   labels=stage_names)
        plt.title(col.replace('eeg_', 'EEG ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'eeg_band_ratios.png'))
    plt.close()
    
    # Plot spectral metrics for EEG
    spectral_cols = [col for col in combined_df.columns if 'spectral' in col and col.startswith('eeg_')]
    
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(spectral_cols):
        plt.subplot(2, 3, i+1)
        plt.boxplot([combined_df[combined_df['sleep_stage']==stage][col].dropna() for stage in stage_names], 
                   labels=stage_names)
        plt.title(col.replace('eeg_', 'EEG ').title())
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'eeg_spectral_metrics.png'))
    plt.close()
    
    # Create correlation matrix
    plt.figure(figsize=(16, 14))
    feature_cols = [col for col in combined_df.columns if col.startswith('eeg_') or col.startswith('eog_')]
    corr = combined_df[feature_cols].corr()
    
    # Plot correlation matrix
    plt.matshow(corr, fignum=1, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    
    # Too many features for labels to be readable
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'feature_correlation.png'))
    plt.close()
    
    # Print summary statistics
    stats_file = os.path.join(OUTPUT_DIR, 'feature_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("PSD Feature Statistics\n")
        f.write("====================\n\n")
        
        f.write(f"Sample size: {len(combined_df)} epochs from {len(csv_files)} recordings\n\n")
        
        f.write("Class distribution:\n")
        stage_counts = combined_df['sleep_stage'].value_counts()
        for stage, count in stage_counts.items():
            f.write(f"  {stage}: {count} epochs ({count/len(combined_df)*100:.1f}%)\n")
        
        f.write("\nFeature statistics:\n")
        stats = combined_df[feature_cols].describe().T
        f.write(stats.to_string())
    
    print(f"Generated distribution plots and statistics in {OUTPUT_DIR}/distribution_plots/")

if __name__ == "__main__":
    # Import seaborn if available, but don't require it
    try:
        import seaborn as sns
        print("Using seaborn for enhanced plots")
        
        def generate_feature_distribution_plots():
            """Generate plots of feature distributions across all files (with seaborn)."""
            # Get all CSV files
            csv_files = glob.glob(os.path.join(OUTPUT_DIR, '*_psd.csv'))
            
            if not csv_files:
                print("No processed files found for plotting.")
                return
            
            # Sample a subset of files (max 10) to avoid memory issues
            if len(csv_files) > 10:
                csv_files = np.random.choice(csv_files, 10, replace=False)
            
            # Load and concatenate data
            dfs = []
            for file in csv_files:
                df = pd.read_csv(file)
                dfs.append(df)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            
            # Map labels to sleep stages
            stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
            combined_df['sleep_stage'] = combined_df['label'].map(lambda x: stage_names[x])
            
            # Create plot directory
            os.makedirs(os.path.join(OUTPUT_DIR, 'distribution_plots'), exist_ok=True)
            
            # Plot relative band powers by sleep stage for EEG
            rel_band_cols = [col for col in combined_df.columns if col.startswith('eeg_rel_')]
            
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(rel_band_cols):
                plt.subplot(2, 3, i+1)
                sns.boxplot(x='sleep_stage', y=col, data=combined_df)
                plt.title(col.replace('eeg_rel_', 'EEG Relative ').title())
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'eeg_relative_bands.png'))
            plt.close()
            
            # Plot band ratios for EEG
            ratio_cols = [col for col in combined_df.columns if 'ratio' in col and col.startswith('eeg_')]
            
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(ratio_cols):
                plt.subplot(2, 2, i+1)
                sns.boxplot(x='sleep_stage', y=col, data=combined_df)
                plt.title(col.replace('eeg_', 'EEG ').title())
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'eeg_band_ratios.png'))
            plt.close()
            
            # Plot spectral metrics for EEG
            spectral_cols = [col for col in combined_df.columns if 'spectral' in col and col.startswith('eeg_')]
            
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(spectral_cols):
                plt.subplot(2, 3, i+1)
                sns.boxplot(x='sleep_stage', y=col, data=combined_df)
                plt.title(col.replace('eeg_', 'EEG ').title())
                plt.xticks(rotation=45)
                plt.tight_layout()
            
            plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'eeg_spectral_metrics.png'))
            plt.close()
            
            # Create correlation heatmap
            plt.figure(figsize=(16, 14))
            feature_cols = [col for col in combined_df.columns if col.startswith('eeg_') or col.startswith('eog_')]
            corr = combined_df[feature_cols].corr()
            
            # Use seaborn for better heatmap
            sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.5, annot=False)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'distribution_plots', 'feature_correlation.png'))
            plt.close()
            
            # Print summary statistics
            stats_file = os.path.join(OUTPUT_DIR, 'feature_statistics.txt')
            with open(stats_file, 'w') as f:
                f.write("PSD Feature Statistics\n")
                f.write("====================\n\n")
                
                f.write(f"Sample size: {len(combined_df)} epochs from {len(csv_files)} recordings\n\n")
                
                f.write("Class distribution:\n")
                stage_counts = combined_df['sleep_stage'].value_counts()
                for stage, count in stage_counts.items():
                    f.write(f"  {stage}: {count} epochs ({count/len(combined_df)*100:.1f}%)\n")
                
                f.write("\nFeature statistics:\n")
                stats = combined_df[feature_cols].describe().T
                f.write(stats.to_string())
            
            print(f"Generated distribution plots and statistics in {OUTPUT_DIR}/distribution_plots/")
            
    except ImportError:
        print("Seaborn not available, using matplotlib for plots")
    
    main()