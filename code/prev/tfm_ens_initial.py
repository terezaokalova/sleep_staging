#!/usr/bin/env python3
import os
import glob
import numpy as np
import mne
from joblib import Parallel, delayed

# Attempt to import torch for GPU-accelerated normalization
try:
    import torch
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
except ImportError:
    use_gpu = False

# Server paths
BASE_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'
DATA_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/processed_sleepedf'
SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry']
os.makedirs(DATA_DIR, exist_ok=True)

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
    "Sleep stage 4": 3,  # collapsed with stage 3
    "Sleep stage R": 4
}

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
    tmax = epoch_length - 1 / raw.info['sfreq']
    epochs = mne.Epochs(raw, events=events, event_id=ANNOTATION_MAP, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False)
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    # Normalize each channel; if GPU is available, perform on GPU
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
    np.savez_compressed(os.path.join(output_dir, f"{rec_id}_epochs.npz"),
                        data=data.astype('float32'), labels=labels.astype('int8'))
    sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
    np.savez_compressed(os.path.join(output_dir, f"{rec_id}_sequences.npz"),
                        sequences=sequences.astype('float32'), seq_labels=seq_labels.astype('int8'))
    print(f"Processed {rec_id}: epochs {data.shape[0]}, sequences {sequences.shape[0]}, channels: {ch_names}")

def main():
    psg_files = []
    for sub in SUBFOLDERS:
        psg_files.extend(glob.glob(os.path.join(BASE_DIR, sub, '*-PSG.edf')))
    print(f"Found {len(psg_files)} PSG files. GPU Enabled: {use_gpu}")
    Parallel(n_jobs=2)(delayed(process_and_save)(f, DATA_DIR, CHANNELS_TO_LOAD) for f in psg_files)

if __name__ == '__main__':
    main()
