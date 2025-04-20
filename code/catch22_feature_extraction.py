#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import catch22  # The catch22 library for extracting features

# Configuration
BASE_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'
PROCESSED_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/processed_sleepedf'
OUTPUT_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/catch22_feats_sleepedf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_true_subject_id(filename):
    """Extract true subject ID ignoring the night number"""
    basename = os.path.basename(filename).split('_')[0]
    if basename.startswith('SC4'):
        return basename[:5]  # SC4xx - first 5 chars for Sleep Cassette
    elif basename.startswith('ST7'):
        return basename[:5]  # ST7xx - first 5 chars for Sleep Telemetry
    else:
        return basename[:6]  # Fallback to original logic

def extract_catch22_features(data_chunk, channel_idx=0):
    """
    Extract catch22 features from a single data chunk
    
    Args:
        data_chunk: 2D array with shape (channels, time_points)
        channel_idx: Index of the channel to use (0 for EEG, 1 for EOG)
        
    Returns:
        List of 22 catch22 features
    """
    # Select the channel
    channel_data = data_chunk[channel_idx, :]
    
    # Calculate all 22 catch22 features
    features = catch22.catch22_all(channel_data)
    
    # Return just the feature values (index 1 of each tuple)
    return [feat[1] for feat in features['values']]

def process_file(file_path):
    """Process a single epochs file to extract catch22 features"""
    start_time = time.time()
    
    # Load the epochs file
    data = np.load(file_path)
    epochs = data['data']  # Shape: (n_epochs, n_channels, n_samples)
    labels = data['labels']  # Shape: (n_epochs,)
    
    n_epochs, n_channels, _ = epochs.shape
    
    # Extract catch22 features for each epoch, for each channel
    eeg_features = []
    eog_features = []
    
    # Process in batches for memory efficiency
    batch_size = 100
    for i in range(0, n_epochs, batch_size):
        batch_epochs = epochs[i:min(i+batch_size, n_epochs)]
        
        # Extract features in parallel for each epoch
        eeg_batch_features = Parallel(n_jobs=-1)(
            delayed(extract_catch22_features)(batch_epochs[j], channel_idx=0) 
            for j in range(len(batch_epochs))
        )
        
        eog_batch_features = Parallel(n_jobs=-1)(
            delayed(extract_catch22_features)(batch_epochs[j], channel_idx=1) 
            for j in range(len(batch_epochs))
        )
        
        eeg_features.extend(eeg_batch_features)
        eog_features.extend(eog_batch_features)
    
    # Create dataframes for EEG and EOG features
    feature_names = [
        'DN_HistogramMode_5', 'DN_HistogramMode_10', 'SB_BinaryStats_mean_longstretch1',
        'DN_OutlierInclude_p_001_mdrmd', 'DN_OutlierInclude_n_001_mdrmd', 'CO_f1ecac',
        'CO_FirstMin_ac', 'SP_Summaries_welch_rect_area_5_1', 'SP_Summaries_welch_rect_centroid',
        'FC_LocalSimple_mean3_stderr', 'CO_trev_1_num', 'CO_HistogramAMI_even_2_5',
        'IN_AutoMutualInfoStats_40_gaussian_fmmi', 'MD_hrv_classic_pnn40', 'SB_BinaryStats_diff_longstretch0',
        'SB_MotifThree_quantile_hh', 'FC_LocalSimple_mean1_tauresrat', 'CO_Embed2_Dist_tau_d_expfit_meandiff',
        'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1', 'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
        'SB_TransitionMatrix_3ac_sumdiagcov', 'PD_PeriodicityWang_th0_01'
    ]
    
    # Create full feature dataframe
    eeg_df = pd.DataFrame(eeg_features, columns=[f'eeg_{name}' for name in feature_names])
    eog_df = pd.DataFrame(eog_features, columns=[f'eog_{name}' for name in feature_names])
    
    # Combine features from both channels
    all_features_df = pd.concat([eeg_df, eog_df], axis=1)
    all_features_df['label'] = labels
    
    # Get subject ID and recording ID
    rec_id = os.path.basename(file_path).replace('_epochs.npz', '')
    subj_id = get_true_subject_id(rec_id)
    
    # Save features to a file
    output_path = os.path.join(OUTPUT_DIR, f"{rec_id}_catch22_features.csv")
    all_features_df.to_csv(output_path, index=False)
    
    # Also save them to a numpy file for easier loading
    output_path_np = os.path.join(OUTPUT_DIR, f"{rec_id}_catch22_features.npz")
    features_array = all_features_df.drop(columns=['label']).values
    np.savez_compressed(output_path_np, 
                       features=features_array.astype('float32'), 
                       labels=labels.astype('int8'),
                       subject_id=subj_id,
                       recording_id=rec_id)
    
    end_time = time.time()
    print(f"Processed {rec_id} (subject {subj_id}): {n_epochs} epochs, took {end_time - start_time:.2f} seconds")
    
    return subj_id, rec_id

def main():
    # Get all epoch files
    epoch_files = glob.glob(os.path.join(PROCESSED_DIR, '*_epochs.npz'))
    print(f"Found {len(epoch_files)} epoch files.")
    
    # Group files by subject ID
    subject_files = {}
    for file in epoch_files:
        rec_id = os.path.basename(file).replace('_epochs.npz', '')
        subj_id = get_true_subject_id(rec_id)
        if subj_id not in subject_files:
            subject_files[subj_id] = []
        subject_files[subj_id].append(file)
    
    print(f"Found {len(subject_files)} unique subjects.")
    
    # Process all files in parallel, grouped by subject
    results = Parallel(n_jobs=4)(
        delayed(process_file)(file) 
        for files in subject_files.values() 
        for file in files
    )
    
    # Create a mapping file of subject IDs to recording IDs
    subject_mapping = {}
    for subj_id, rec_id in results:
        if subj_id not in subject_mapping:
            subject_mapping[subj_id] = []
        subject_mapping[subj_id].append(rec_id)
    
    # Save the mapping to a file
    mapping_df = pd.DataFrame([(subj, rec) for subj, recs in subject_mapping.items() for rec in recs],
                            columns=['subject_id', 'recording_id'])
    mapping_df.to_csv(os.path.join(OUTPUT_DIR, 'subject_recording_mapping.csv'), index=False)
    
    print("Feature extraction complete!")
    print(f"Features saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()