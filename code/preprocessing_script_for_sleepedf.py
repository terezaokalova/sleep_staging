{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#!/usr/bin/env python3\n",
    "import os, glob, numpy as np, mne, pandas as pd\n",
    "from joblib import Parallel, delayed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Configuration\n",
    "BASE_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'\n",
    "SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry']\n",
    "OUTPUT_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/processed_sleepedf'\n",
    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
    "\n",
    "USE_MULTIPLE_CHANNELS = True\n",
    "if USE_MULTIPLE_CHANNELS:\n",
    "    CHANNELS_TO_LOAD = [\"EEG Fpz-Cz\", \"EOG horizontal\"]\n",
    "else:\n",
    "    CHANNELS_TO_LOAD = [\"EEG Fpz-Cz\"]\n",
    "\n",
    "TARGET_SFREQ = 100.0\n",
    "LOW_FREQ = 0.5\n",
    "HIGH_FREQ = 30.0\n",
    "EPOCH_LENGTH = 30.0\n",
    "SEQ_LENGTH = 20\n",
    "SEQ_STRIDE = 10\n",
    "\n",
    "ANNOTATION_MAP = {\n",
    "    \"Sleep stage W\": 0,\n",
    "    \"Sleep stage 1\": 1,\n",
    "    \"Sleep stage 2\": 2,\n",
    "    \"Sleep stage 3\": 3,\n",
    "    \"Sleep stage 4\": 3,\n",
    "    \"Sleep stage R\": 4\n",
    "}\n",
    "\n",
    "def process_record(psg_path, hyp_path, channels, target_sfreq, low_freq, high_freq, epoch_length):\n",
    "    raw = mne.io.read_raw_edf(psg_path, include=channels, preload=True, verbose=False)\n",
    "    if raw.info['sfreq'] != target_sfreq:\n",
    "        raw.resample(target_sfreq, npad=\"auto\", verbose=False)\n",
    "    picks = mne.pick_types(raw.info, eeg=True, eog=True)\n",
    "    raw.filter(l_freq=low_freq, h_freq=high_freq, picks=picks, verbose=False)\n",
    "    ann = mne.read_annotations(hyp_path)\n",
    "    raw.set_annotations(ann, emit_warning=False)\n",
    "    events, _ = mne.events_from_annotations(raw, event_id=ANNOTATION_MAP, chunk_duration=epoch_length)\n",
    "    tmin = 0.0; tmax = epoch_length - 1/raw.info['sfreq']\n",
    "    epochs = mne.Epochs(raw, events=events, event_id=ANNOTATION_MAP, tmin=tmin, tmax=tmax,\n",
    "                        baseline=None, preload=True, verbose=False)\n",
    "    data = epochs.get_data()\n",
    "    labels = epochs.events[:, -1]\n",
    "    for ch in range(data.shape[1]):\n",
    "        m = np.mean(data[:, ch, :])\n",
    "        s = np.std(data[:, ch, :]) if np.std(data[:, ch, :]) != 0 else 1.0\n",
    "        data[:, ch, :] = (data[:, ch, :] - m) / s\n",
    "    return data, labels, raw.ch_names\n",
    "\n",
    "def create_sequences(data, labels, seq_length, seq_stride):\n",
    "    n_epochs = data.shape[0]\n",
    "    sequences, seq_labels = [], []\n",
    "    for start in range(0, n_epochs - seq_length + 1, seq_stride):\n",
    "        sequences.append(data[start:start+seq_length])\n",
    "        seq_labels.append(labels[start:start+seq_length])\n",
    "    return np.array(sequences), np.array(seq_labels)\n",
    "\n",
    "def find_hypnogram(psg_file):\n",
    "    # Extract subject ID (assumes first 6 characters, e.g., \"SC4001\")\n",
    "    subject_id = os.path.basename(psg_file)[:6]\n",
    "    dir_path = os.path.dirname(psg_file)\n",
    "    pattern = os.path.join(dir_path, f\"{subject_id}*Hypnogram.edf\")\n",
    "    hyp_files = glob.glob(pattern)\n",
    "    if len(hyp_files) == 1:\n",
    "        return hyp_files[0]\n",
    "    elif len(hyp_files) > 1:\n",
    "        return hyp_files[0]  # or choose based on additional rules if needed\n",
    "    else:\n",
    "        return None\n",
    "\n",
    "def process_and_save(psg_file, output_dir, channels):\n",
    "    hyp_file = find_hypnogram(psg_file)\n",
    "    if not hyp_file:\n",
    "        print(f\"Hypnogram not found for {psg_file}, skipping.\")\n",
    "        return\n",
    "    try:\n",
    "        data, labels, ch_names = process_record(psg_file, hyp_file, channels,\n",
    "                                                  TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)\n",
    "    except Exception as e:\n",
    "        print(f\"Error processing {psg_file}: {e}\")\n",
    "        return\n",
    "    rec_id = os.path.basename(psg_file).replace('-PSG.edf', '')\n",
    "    np.savez_compressed(os.path.join(output_dir, f\"{rec_id}_epochs.npz\"),\n",
    "                        data=data.astype('float32'), labels=labels.astype('int8'))\n",
    "    sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)\n",
    "    np.savez_compressed(os.path.join(output_dir, f\"{rec_id}_sequences.npz\"),\n",
    "                        sequences=sequences.astype('float32'), seq_labels=seq_labels.astype('int8'))\n",
    "    print(f\"Processed {rec_id}: epochs {data.shape[0]}, sequences {sequences.shape[0]}, channels: {ch_names}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 197 PSG files.\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4201E0: epochs 2803, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4171E0: epochs 2741, sequences 273, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4421E0: epochs 2764, sequences 275, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4272F0: epochs 2870, sequences 286, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4751E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4452F0: epochs 2670, sequences 266, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4622E0: epochs 2856, sequences 284, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4592G0: epochs 2040, sequences 203, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4002E0: epochs 2829, sequences 281, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4112E0: epochs 2780, sequences 277, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4732E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4482F0: epochs 2880, sequences 287, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4442E0: epochs 2778, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4802G0: epochs 2810, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4641E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/tereza/anaconda3/envs/cnt_gen/lib/python3.12/site-packages/joblib/externals/loky/process_executor.py:752: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4542F0: epochs 2680, sequences 267, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4531E0: epochs 2602, sequences 259, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4362F0: epochs 2266, sequences 225, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4061E0: epochs 2770, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4311E0: epochs 2670, sequences 266, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4232E0: epochs 2635, sequences 262, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4142E0: epochs 2774, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4461F0: epochs 2695, sequences 268, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4762E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Processed SC4412E0: epochs 2765, sequences 275, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4082E0: epochs 2634, sequences 262, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4611E0: epochs 2644, sequences 263, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4381F0: epochs 2750, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4332F0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Processed SC4031E0: epochs 2820, sequences 281, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4121E0: epochs 2685, sequences 267, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4251E0: epochs 2760, sequences 275, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4772G0: epochs 2242, sequences 223, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4291G0: epochs 2756, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4701E0: epochs 2679, sequences 266, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4192E0: epochs 2605, sequences 259, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4571F0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4502E0: epochs 2770, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4052E0: epochs 2804, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4322E0: epochs 2616, sequences 260, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4351F0: epochs 2710, sequences 270, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4671G0: epochs 2780, sequences 277, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4602E0: epochs 2800, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4091E0: epochs 2721, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4022E0: epochs 2755, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4151E0: epochs 2616, sequences 260, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4221E0: epochs 2700, sequences 269, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4401E0: epochs 2630, sequences 262, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4472F0: epochs 2789, sequences 277, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4562F0: epochs 2800, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4511E0: epochs 2712, sequences 270, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4342F0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4661E0: epochs 2664, sequences 265, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4041E0: epochs 2569, sequences 255, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4242E0: epochs 2710, sequences 270, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4712E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4181E0: epochs 2756, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4282G0: epochs 2814, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4822G0: epochs 2810, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4631E0: epochs 2758, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4011E0: epochs 2802, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4581G0: epochs 2738, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4261F0: epochs 2800, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4162E0: epochs 2750, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4742E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4212E0: epochs 2694, sequences 268, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4432E0: epochs 2780, sequences 277, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4522E0: epochs 2774, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4652E0: epochs 2838, sequences 282, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4551F0: epochs 2772, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4302E0: epochs 2808, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4072E0: epochs 2770, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4371F0: epochs 2850, sequences 284, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4491G0: epochs 2800, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4721E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4101E0: epochs 2719, sequences 270, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4811G0: epochs 2402, sequences 239, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4222E0: epochs 2760, sequences 275, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4152E0: epochs 2859, sequences 284, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4471F0: epochs 2739, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4402E0: epochs 2790, sequences 278, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4092E0: epochs 2044, sequences 203, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4601E0: epochs 2730, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4672G0: epochs 2580, sequences 257, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4021E0: epochs 2804, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4821G0: epochs 2721, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4131E0: epochs 2814, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4241E0: epochs 2700, sequences 269, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4281G0: epochs 2788, sequences 277, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4711E0: epochs 2730, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4182E0: epochs 2842, sequences 283, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4662E0: epochs 2820, sequences 281, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4512E0: epochs 2750, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4561F0: epochs 2702, sequences 269, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4341F0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4042E0: epochs 2788, sequences 277, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4211E0: epochs 2805, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4161E0: epochs 2621, sequences 261, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4262F0: epochs 2720, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4741E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4431E0: epochs 2723, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4632E0: epochs 2848, sequences 283, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4582G0: epochs 2634, sequences 262, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4012E0: epochs 2848, sequences 283, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4812G0: epochs 2414, sequences 240, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4102E0: epochs 2857, sequences 284, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4492G0: epochs 2135, sequences 212, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4722E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4552F0: epochs 2810, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4651E0: epochs 2860, sequences 285, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4372F0: epochs 2840, sequences 283, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4071E0: epochs 2810, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4301E0: epochs 2644, sequences 263, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4621E0: epochs 2612, sequences 260, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4001E0: epochs 2650, sequences 264, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4591G0: epochs 2819, sequences 280, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4271F0: epochs 2440, sequences 243, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4202E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4172E0: epochs 2720, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4451F0: epochs 2790, sequences 278, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4752E0: epochs 2470, sequences 246, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4422E0: epochs 2682, sequences 267, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4642E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Processed SC4532E0: epochs 2744, sequences 273, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4541F0: epochs 2739, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4312E0: epochs 2700, sequences 269, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4062E0: epochs 2830, sequences 282, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4801G0: epochs 2774, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4111E0: epochs 2641, sequences 263, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4481F0: epochs 2880, sequences 287, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4731E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4441E0: epochs 2620, sequences 261, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4382F0: epochs 2770, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4612E0: epochs 2730, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4331F0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Processed SC4081E0: epochs 2796, sequences 278, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4032E0: epochs 2732, sequences 272, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4141E0: epochs 2756, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4231E0: epochs 2743, sequences 273, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4761E0: epochs 2610, sequences 260, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4411E0: epochs 2728, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4462F0: epochs 2854, sequences 284, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4501E0: epochs 2750, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4572F0: epochs 2873, sequences 286, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4321E0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4352F0: epochs 2530, sequences 252, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4051E0: epochs 2722, sequences 271, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4252E0: epochs 2665, sequences 265, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4122E0: epochs 2606, sequences 259, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4191E0: epochs 2774, sequences 276, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed SC4702E0: epochs 2624, sequences 261, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed SC4771G0: epochs 2756, sequences 274, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Processed SC4292G0: epochs 2807, sequences 279, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7172J0: epochs 1005, sequences 99, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7202J0: epochs 898, sequences 88, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7111J0: epochs 1017, sequences 100, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7062J0: epochs 952, sequences 94, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n",
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7141J0: epochs 858, sequences 84, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Processed ST7081J0: epochs 966, sequences 95, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7122J0: epochs 945, sequences 93, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Processed ST7191J0: epochs 955, sequences 94, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7051J0: epochs 1018, sequences 100, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7092J0: epochs 923, sequences 91, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Error processing /Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-telemetry/ST7222J0-PSG.edf: No matching events found for Sleep stage 3 (event id 3)\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7021J0: epochs 920, sequences 91, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7152J0: epochs 1045, sequences 103, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7042J0: epochs 1130, sequences 112, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7131J0: epochs 859, sequences 84, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7241J0: epochs 1021, sequences 101, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7182J0: epochs 967, sequences 95, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7012J0: epochs 1040, sequences 103, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7211J0: epochs 1077, sequences 106, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7071J0: epochs 821, sequences 81, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7161J0: epochs 1057, sequences 104, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7102J0: epochs 980, sequences 97, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7151J0: epochs 897, sequences 88, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7221J0: epochs 1033, sequences 102, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7091J0: epochs 943, sequences 93, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7022J0: epochs 944, sequences 93, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7242J0: epochs 939, sequences 92, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7132J0: epochs 852, sequences 84, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Processed ST7181J0: epochs 959, sequences 94, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7041J0: epochs 1007, sequences 99, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Processed ST7162J0: epochs 988, sequences 97, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7212J0: epochs 1010, sequences 100, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7011J0: epochs 1092, sequences 108, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7101J0: epochs 1042, sequences 103, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7072J0: epochs 914, sequences 90, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7201J0: epochs 921, sequences 91, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7171J0: epochs 964, sequences 95, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7061J0: epochs 1008, sequences 99, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7112J0: epochs 978, sequences 96, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7082J0: epochs 931, sequences 92, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n",
      "Processed ST7142J0: epochs 873, sequences 86, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7052J0: epochs 1034, sequences 102, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Used Annotations descriptions: [np.str_('Sleep stage 1'), np.str_('Sleep stage 2'), np.str_('Sleep stage 3'), np.str_('Sleep stage 4'), np.str_('Sleep stage R'), np.str_('Sleep stage W')]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/var/folders/g6/qyrrm6017k55qbd8_wywqxgw0000gn/T/ipykernel_26652/2945860575.py:41: FutureWarning: The current default of copy=False will change to copy=True in 1.7. Set the value of copy explicitly to avoid this warning\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processed ST7121J0: epochs 1026, sequences 101, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n",
      "Processed ST7192J0: epochs 931, sequences 92, channels: ['EEG Fpz-Cz', 'EOG horizontal']\n"
     ]
    }
   ],
   "source": [
    "def main():\n",
    "    psg_files = []\n",
    "    for sub in SUBFOLDERS:\n",
    "        psg_files.extend(glob.glob(os.path.join(BASE_DIR, sub, '*-PSG.edf')))\n",
    "    print(f\"Found {len(psg_files)} PSG files.\")\n",
    "    Parallel(n_jobs=2)(delayed(process_and_save)(f, OUTPUT_DIR, CHANNELS_TO_LOAD) for f in psg_files)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch-level data shape: (1026, 2, 3000)\n",
      "Epoch-level labels shape: (1026,)\n",
      "First 10 labels: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 1 0 0 1 1 1 1 1 1 1 1 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n",
      " 2 2 2 3 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1]\n",
      "Sequence-level data shape: (101, 20, 2, 3000)\n",
      "Sequence-level labels shape: (101, 20)\n",
      "First sequence labels: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "OUTPUT_DIR = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/processed_sleepedf'\n",
    "rec_id = 'ST7121J0'  # Change this to any recording ID you want to inspect\n",
    "\n",
    "# Load epoch-level data\n",
    "epoch_file = os.path.join(OUTPUT_DIR, f\"{rec_id}_epochs.npz\")\n",
    "epoch_data = np.load(epoch_file)\n",
    "data = epoch_data['data']      # shape: (n_epochs, n_channels, n_samples)\n",
    "labels = epoch_data['labels']  # shape: (n_epochs,)\n",
    "\n",
    "print(f\"Epoch-level data shape: {data.shape}\")\n",
    "print(f\"Epoch-level labels shape: {labels.shape}\")\n",
    "print(\"First 10 labels:\", labels[:100])\n",
    "\n",
    "# Load sequence-level data\n",
    "seq_file = os.path.join(OUTPUT_DIR, f\"{rec_id}_sequences.npz\")\n",
    "seq_data = np.load(seq_file)\n",
    "sequences = seq_data['sequences']       # shape: (n_sequences, SEQ_LENGTH, n_channels, n_samples)\n",
    "seq_labels = seq_data['seq_labels']     # shape: (n_sequences, SEQ_LENGTH)\n",
    "\n",
    "print(f\"Sequence-level data shape: {sequences.shape}\")\n",
    "print(f\"Sequence-level labels shape: {seq_labels.shape}\")\n",
    "print(\"First sequence labels:\", seq_labels[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Annotation descriptions: ['Sleep stage W' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 1'\n",
      " 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage 4' 'Sleep stage 3'\n",
      " 'Sleep stage 4' 'Sleep stage 3' 'Sleep stage 4' 'Sleep stage 3'\n",
      " 'Sleep stage 2' 'Sleep stage R' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage 2'\n",
      " 'Sleep stage 3' 'Sleep stage 4' 'Sleep stage 3' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage W' 'Sleep stage 1'\n",
      " 'Sleep stage 2' 'Sleep stage W' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage W' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage R'\n",
      " 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage 2' 'Sleep stage 3'\n",
      " 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage 2' 'Sleep stage 3'\n",
      " 'Sleep stage 2' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 3'\n",
      " 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage 2' 'Sleep stage 3'\n",
      " 'Sleep stage 4' 'Sleep stage 3' 'Sleep stage 4' 'Sleep stage 3'\n",
      " 'Sleep stage 4' 'Sleep stage 3' 'Sleep stage 4' 'Sleep stage 3'\n",
      " 'Sleep stage 4' 'Sleep stage W' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage R' 'Sleep stage 2' 'Sleep stage W' 'Sleep stage 1'\n",
      " 'Sleep stage W' 'Sleep stage 1' 'Sleep stage W' 'Sleep stage 1'\n",
      " 'Sleep stage 2' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 3'\n",
      " 'Sleep stage 2' 'Sleep stage 3' 'Sleep stage W' 'Sleep stage 1'\n",
      " 'Sleep stage W' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 1'\n",
      " 'Sleep stage 2' 'Sleep stage R' 'Sleep stage 2' 'Sleep stage W'\n",
      " 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage W' 'Sleep stage 1'\n",
      " 'Sleep stage W' 'Sleep stage 1' 'Sleep stage 2' 'Sleep stage 1'\n",
      " 'Sleep stage 2' 'Sleep stage W' 'Sleep stage 1' 'Sleep stage R'\n",
      " 'Sleep stage W' 'Sleep stage R' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage W' 'Sleep stage 1' 'Sleep stage 2'\n",
      " 'Sleep stage 1' 'Sleep stage W' 'Sleep stage 1' 'Sleep stage R'\n",
      " 'Sleep stage W' 'Sleep stage R' 'Sleep stage W' 'Sleep stage ?'\n",
      " 'Sleep stage ?' 'Sleep stage ?' 'Sleep stage ?']\n"
     ]
    }
   ],
   "source": [
    "hyp_path = '/Users/tereza/spring_2025/STAT_4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0/sleep-cassette/SC4121EC-Hypnogram.edf'\n",
    "ann = mne.read_annotations(hyp_path)\n",
    "print(\"Annotation descriptions:\", ann.description)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "cnt_gen",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
