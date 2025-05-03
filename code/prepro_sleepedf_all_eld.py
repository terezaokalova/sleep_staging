import os
import glob
import mne
import numpy as np
import concurrent.futures
import functools # Import functools for partial
import time # Optional: for timing the script

# --- Configuration ---
# Paths specific to the remote server 'pioneer'
BASE_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'
SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry'] # Subdirectories containing the raw data within BASE_DIR
OUTPUT_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleepedf_prepro_all_eld' # Output directory for processed files

# Ensure output directory exists
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")
except OSError as e:
    print(f"Error creating output directory {OUTPUT_DIR}: {e}")
    # Exit if we can't create the output directory
    exit(1)


# Signal Processing Parameters
TARGET_SFREQ = 100.0 # Hz
LOW_FREQ = 0.5 # Hz
HIGH_FREQ = 30.0 # Hz
EPOCH_LENGTH = 30.0 # seconds

# Sequence Parameters (for potential sequential modeling)
SEQ_LENGTH = 20 # Number of consecutive epochs in a sequence
SEQ_STRIDE = 10 # Step size between start of consecutive sequences

# Annotation mapping (Sleep stage 4 merged into 3 as is common)
ANNOTATION_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3, # Merging stage 4 into 3
    "Sleep stage R": 4
}
# --- End Configuration ---

def find_hypnogram(psg_file):
    """Finds the corresponding Hypnogram EDF file for a given PSG file."""
    base_name = os.path.basename(psg_file)
    # Assumes format like SC4XXXE0-PSG.edf or ST7XXXJ0-PSG.edf
    identifier = base_name.split('-')[0]
    dir_path = os.path.dirname(psg_file)

    pattern = os.path.join(dir_path, f"{identifier}*Hypnogram.edf")
    hyp_files = glob.glob(pattern)

    if len(hyp_files) == 1:
        return hyp_files[0]
    elif len(hyp_files) > 1:
        print(f"Warning: Multiple hypnogram files found for {psg_file}. Using {hyp_files[0]}")
        return hyp_files[0]
    else:
        print(f"Warning: No hypnogram file found matching pattern {pattern} for {psg_file}")
        return None

def process_record(psg_path, hyp_path, target_sfreq, low_freq, high_freq, epoch_length):
    """
    Loads, preprocesses, and epochs a single PSG recording.
    Loads all channels initially, then picks only EEG and EOG.
    """
    try:
        # Load raw data, preload=True makes subsequent operations easier if memory allows
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error loading {psg_path}: {e}")
        return None, None, None

    # Select only EEG and EOG channels
    try:
        picks = mne.pick_types(raw.info, eeg=True, eog=True, meg=False, stim=False, exclude='bads')
        if not picks.size:
             print(f"Skipping {os.path.basename(psg_path)}: No EEG or EOG channels found.")
             del raw # Clean up memory
             return None, None, None
        raw.pick(picks) # Keep only selected channels
    except Exception as e:
         print(f"Error picking EEG/EOG channels for {os.path.basename(psg_path)}: {e}")
         del raw
         return None, None, None

    initial_sfreq = raw.info['sfreq']
    # Resample if necessary
    if initial_sfreq != target_sfreq:
        try:
            raw.resample(target_sfreq, npad="auto", verbose=False)
        except Exception as e:
            print(f"Error resampling {os.path.basename(psg_path)} from {initial_sfreq} to {target_sfreq}: {e}")
            del raw
            return None, None, None

    # Apply band-pass filter
    try:
        raw.filter(l_freq=low_freq, h_freq=high_freq, picks='all', fir_design='firwin', skip_by_annotation='edge', verbose=False)
    except Exception as e:
        print(f"Error filtering {os.path.basename(psg_path)}: {e}")
        del raw
        return None, None, None

    # Load annotations
    try:
        ann = mne.read_annotations(hyp_path)
        raw.set_annotations(ann, emit_warning=False)
        # Create events based on annotations and epoch length
        events, event_ids = mne.events_from_annotations(
            raw, event_id=ANNOTATION_MAP, chunk_duration=epoch_length, verbose=False
        )
        # Filter out event IDs not in our map (might happen with 'edge' annotations etc.)
        valid_event_mask = np.isin(events[:, 2], list(event_ids.values()))
        events = events[valid_event_mask]
        if events.shape[0] == 0:
            print(f"No valid sleep stage annotations found in {os.path.basename(psg_path)} after mapping.")
            del raw
            return None, None, None

    except Exception as e:
        print(f"Error processing annotations for {os.path.basename(psg_path)} ({os.path.basename(hyp_path)}): {e}")
        del raw
        return None, None, None

    # Create epochs
    tmin = 0.0
    tmax = epoch_length - 1/raw.info['sfreq'] # Duration minus one sample period
    try:
        epochs = mne.Epochs(
            raw, events=events, event_id=event_ids, tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False
        )
    except Exception as e:
        print(f"Error creating epochs for {os.path.basename(psg_path)}: {e}")
        del raw
        return None, None, None

    # Check if any epochs were created
    if len(epochs) == 0:
        print(f"No epochs created for {os.path.basename(psg_path)}, likely due to event timing or data gaps.")
        del raw, epochs
        return None, None, None

    # Get data (epochs x channels x time) and labels
    data = epochs.get_data(copy=False) # Use copy=False if memory is tight, but be careful
    labels = epochs.events[:, -1]

    # Z-score normalization per channel across all epochs for this recording
    # Note: Normalizing across all epochs assumes stationarity across the recording for mean/std
    for ch in range(data.shape[1]):
        channel_data = data[:, ch, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std < 1e-6: # Check for near-zero std deviation
             print(f"Warning: Channel {epochs.ch_names[ch]} in {os.path.basename(psg_path)} has std near zero. Setting to 0.")
             data[:, ch, :] = 0.0
        else:
             data[:, ch, :] = (channel_data - mean) / std

    ch_names = epochs.ch_names

    # Make a copy before deleting MNE object if get_data(copy=False) was used
    data_copy = data.copy()
    labels_copy = labels.copy()
    ch_names_copy = list(ch_names) # Keep a copy of channel names

    # Clean up MNE objects explicitly to free memory
    del epochs
    del raw

    return data_copy, labels_copy, ch_names_copy


def create_sequences(data, labels, seq_length, seq_stride):
    """Creates sequences of consecutive epochs."""
    n_epochs, n_channels, n_times = data.shape

    sequences = []
    seq_labels = []

    # Iterate using stride to create overlapping sequences
    for start in range(0, n_epochs - seq_length + 1, seq_stride):
        end = start + seq_length
        sequences.append(data[start:end, :, :]) # Shape: (seq_length, n_channels, n_times)
        seq_labels.append(labels[start:end]) # Shape: (seq_length,)

    if not sequences:
        # Return empty arrays with correct dimensions if no sequences could be formed
        return np.empty((0, seq_length, n_channels, n_times)), np.empty((0, seq_length))

    # Stack sequences into numpy arrays
    # Output shape: (num_sequences, seq_length, n_channels, n_times)
    # Output labels shape: (num_sequences, seq_length)
    return np.array(sequences), np.array(seq_labels)


def process_and_save(psg_file, output_dir):
    """Processes a single PSG file and saves epochs and sequences."""
    psg_basename = os.path.basename(psg_file)
    print(f"Starting processing for: {psg_basename}")
    hyp_file = find_hypnogram(psg_file)
    if not hyp_file:
        # Error already printed in find_hypnogram
        return f"FAILED_HYPNO: {psg_basename}" # Return status

    try:
        # Process the record - loads all channels, picks EEG/EOG
        data, labels, ch_names = process_record(psg_file, hyp_file,
                                                TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)

        # Check if processing returned valid data
        if data is None or labels is None or ch_names is None:
            print(f"Skipping saving for {psg_basename} due to processing errors or lack of data.")
            return f"FAILED_PROCESS: {psg_basename}" # Return status

    except Exception as e:
        # Catch any unexpected errors during processing
        print(f"Unhandled error during process_record call for {psg_basename}: {e}")
        return f"FAILED_UNHANDLED: {psg_basename}" # Return status

    # Proceed if data is valid
    rec_id = psg_basename.replace('-PSG.edf', '')
    save_error = False

    # --- Save Epochs ---
    epochs_filename = os.path.join(output_dir, f"{rec_id}_epochs.npz")
    try:
        np.savez_compressed(epochs_filename,
                            data=data.astype('float32'),
                            labels=labels.astype('int8'),
                            ch_names=np.array(ch_names, dtype=object), # Ensure correct dtype for names
                            sfreq=np.array([TARGET_SFREQ]))
        print(f"  Saved epochs for {rec_id}: {data.shape} -> {epochs_filename}")
    except Exception as e:
        print(f"  Error saving epochs for {rec_id}: {e}")
        save_error = True # Mark that saving failed

    # --- Create and Save Sequences ---
    # Check if data exists before creating sequences (might be None if epochs saving failed and we decided to stop)
    if data is not None and labels is not None:
        sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
        sequences_filename = os.path.join(output_dir, f"{rec_id}_sequences.npz")

        if sequences.size == 0:
             print(f"  No sequences generated for {rec_id} (recording might be too short or stride too large).")
        else:
            try:
                np.savez_compressed(sequences_filename,
                                    sequences=sequences.astype('float32'),
                                    seq_labels=seq_labels.astype('int8'),
                                    ch_names=np.array(ch_names, dtype=object), # Ensure correct dtype
                                    sfreq=np.array([TARGET_SFREQ]))
                print(f"  Saved sequences for {rec_id}: {sequences.shape} -> {sequences_filename}")
            except Exception as e:
                print(f"  Error saving sequences for {rec_id}: {e}")
                save_error = True # Mark that saving failed
        # Clean up sequence variables
        del sequences
        del seq_labels

    # Clean up epoch variables
    del data
    del labels
    del ch_names

    print(f"Finished processing for: {psg_basename}")
    return f"SUCCESS: {psg_basename}" if not save_error else f"FAILED_SAVE: {psg_basename}"


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    print("-" * 50)
    print("Starting Sleep EDF Preprocessing Script")
    print(f"Data Source Base Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Processing Subfolders: {SUBFOLDERS}")
    print("-" * 50)
    print(f"Parameters: Target SFreq={TARGET_SFREQ}Hz, Filter={LOW_FREQ}-{HIGH_FREQ}Hz, Epoch={EPOCH_LENGTH}s")
    print(f"Sequence: Length={SEQ_LENGTH} epochs, Stride={SEQ_STRIDE} epochs")
    print(f"Annotation Map: {ANNOTATION_MAP}")
    print("-" * 50)


    all_psg_files = []
    for subfolder in SUBFOLDERS:
        folder_path = os.path.join(BASE_DIR, subfolder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Subfolder '{folder_path}' not found. Skipping.")
            continue
        # Find files matching the typical PSG naming convention
        psg_pattern = os.path.join(folder_path, '*PSG.edf')
        found_files = sorted(glob.glob(psg_pattern)) # Sort for consistent processing order
        all_psg_files.extend(found_files)
        print(f"Found {len(found_files)} PSG files in '{folder_path}'")

    if not all_psg_files:
        print("\nError: No PSG files found in the specified subfolders. Please check paths:")
        print(f"  - BASE_DIR: {BASE_DIR}")
        print(f"  - SUBFOLDERS: {SUBFOLDERS}")
        exit(1) # Exit if no files to process
    else:
        print(f"\nFound a total of {len(all_psg_files)} PSG files to process.")
        print("Starting parallel processing using ProcessPoolExecutor...")
        print("(Number of workers defaults to the number of CPU cores)")
        print("-" * 50)

        # Create a partial function with the output directory fixed
        # This makes it easy to use with executor.map which passes only one argument (psg_file)
        process_func = functools.partial(process_and_save, output_dir=OUTPUT_DIR)

        success_count = 0
        fail_count = 0
        results_summary = []

        # Use ProcessPoolExecutor for parallel processing
        # Set max_workers=N to limit cores, e.g., max_workers=4
        # If None, it uses os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            # executor.map processes files concurrently and returns an iterator of results
            results = executor.map(process_func, all_psg_files)

            # Process results as they complete
            for result in results:
                results_summary.append(result) # Store result status
                if result.startswith("SUCCESS"):
                    success_count += 1
                else:
                    fail_count += 1
                # Optional: Print progress update periodically
                # print(f"  Processed {success_count + fail_count} / {len(all_psg_files)} files...")


    # --- Script Completion Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print("-" * 50)
    print("Preprocessing Complete")
    print(f"Total files processed: {len(all_psg_files)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if fail_count > 0:
        print("\nFailures occurred. Check logs above for details on specific files.")
        print("Failed file summaries:")
        for res in results_summary:
            if not res.startswith("SUCCESS"):
                print(f"  - {res}")
    print("-" * 50)