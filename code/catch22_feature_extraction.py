#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import pycatch22 as catch22

# Configuration for the Borel server
BASE_DIR      = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'
PROCESSED_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/processed_sleepedf'
OUTPUT_DIR    = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/c22_processed_sleepedf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_true_subject_id(filename):
    """Extract true subject ID ignoring the night number."""
    basename = os.path.basename(filename).split('_')[0]
    if basename.startswith('SC4'):
        return basename[:5]  # SC4xx
    elif basename.startswith('ST7'):
        return basename[:5]  # ST7xx
    else:
        return basename[:6]

def extract_catch22_features(data_chunk, channel_idx=0):
    """
    Extract the 22 catch22 features from one epoch.

    Args:
        data_chunk: np.ndarray, shape (n_channels, n_samples)
        channel_idx: int, which channel to use (0=EEG, 1=EOG)

    Returns:
        List[float]: 22 catch22 feature values.
    """
    channel_data = data_chunk[channel_idx, :]
    feats = catch22.catch22_all(channel_data)
    # `feats['values']` is already a list of 22 floats
    return feats['values']

def process_file(file_path):
    """Process one epochs‐NPZ file and save its catch22 features."""
    start_time = time.time()
    try:
        data   = np.load(file_path)
        epochs = data['data']    # (n_epochs, n_channels, n_samples)
        labels = data['labels']  # (n_epochs,)

        n_epochs, _, _ = epochs.shape
        eeg_feats = []
        eog_feats = []

        batch_size = 100
        for i in range(0, n_epochs, batch_size):
            batch = epochs[i:i+batch_size]
            # parallel over each epoch in this batch
            eeg_batch = Parallel(n_jobs=-1)(
                delayed(extract_catch22_features)(batch[j], channel_idx=0)
                for j in range(batch.shape[0])
            )
            eog_batch = Parallel(n_jobs=-1)(
                delayed(extract_catch22_features)(batch[j], channel_idx=1)
                for j in range(batch.shape[0])
            )
            eeg_feats.extend(eeg_batch)
            eog_feats.extend(eog_batch)

        feature_names = [
            'DN_HistogramMode_5', 'DN_HistogramMode_10',
            'SB_BinaryStats_mean_longstretch1', 'DN_OutlierInclude_p_001_mdrmd',
            'DN_OutlierInclude_n_001_mdrmd', 'CO_f1ecac', 'CO_FirstMin_ac',
            'SP_Summaries_welch_rect_area_5_1', 'SP_Summaries_welch_rect_centroid',
            'FC_LocalSimple_mean3_stderr', 'CO_trev_1_num',
            'CO_HistogramAMI_even_2_5', 'IN_AutoMutualInfoStats_40_gaussian_fmmi',
            'MD_hrv_classic_pnn40', 'SB_BinaryStats_diff_longstretch0',
            'SB_MotifThree_quantile_hh', 'FC_LocalSimple_mean1_tauresrat',
            'CO_Embed2_Dist_tau_d_expfit_meandiff',
            'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',
            'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',
            'SB_TransitionMatrix_3ac_sumdiagcov', 'PD_PeriodicityWang_th0_01'
        ]

        eeg_df = pd.DataFrame(eeg_feats, columns=[f'eeg_{n}' for n in feature_names])
        eog_df = pd.DataFrame(eog_feats, columns=[f'eog_{n}' for n in feature_names])

        all_df = pd.concat([eeg_df, eog_df], axis=1)
        all_df['label'] = labels

        rec_id = os.path.basename(file_path).replace('_epochs.npz','')
        subj_id = get_true_subject_id(rec_id)

        # Save CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{rec_id}_c22.csv")
        all_df.to_csv(csv_path, index=False)

        # Save compressed NPZ
        arr = all_df.drop(columns=['label']).values.astype('float32')
        npz_path = os.path.join(OUTPUT_DIR, f"{rec_id}_c22.npz")
        np.savez_compressed(
            npz_path,
            features=arr,
            labels=labels.astype('int8'),
            subject_id=subj_id,
            recording_id=rec_id
        )

        elapsed = time.time() - start_time
        print(f"Processed {rec_id} (subj {subj_id}): {n_epochs} epochs in {elapsed:.1f}s")
        return subj_id, rec_id

    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None, None

def main():
    epoch_files = glob.glob(os.path.join(PROCESSED_DIR, '*_epochs.npz'))
    print(f"Found {len(epoch_files)} files.")

    if not epoch_files:
        print("No epoch files—run preprocessing first.")
        return

    # group by subject
    subj_map = {}
    for fp in epoch_files:
        rec = os.path.basename(fp).replace('_epochs.npz','')
        subj = get_true_subject_id(rec)
        subj_map.setdefault(subj, []).append(fp)

    print(f"{len(subj_map)} unique subjects.")
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() // 2)
    print(f"Using {n_jobs} parallel jobs.")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(f)
        for files in subj_map.values() for f in files
    )
    results = [r for r in results if r[0] is not None]

    # write mapping
    mapping = [(s, r) for s, rs in results for r in [rs]]
    map_df = pd.DataFrame(mapping, columns=['subject_id','recording_id'])
    map_df.to_csv(os.path.join(OUTPUT_DIR,'subject_recording_mapping.csv'), index=False)

    print("alles")

if __name__ == "__main__":
    main()
