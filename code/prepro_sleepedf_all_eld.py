import os
import glob
import mne
import numpy as np
import concurrent.futures
import functools # Import functools for partial
import time
import logging # Import logging module
import traceback # Import traceback module
import sys # Import sys for exit

# --- Logging Configuration ---
LOG_FILE = 'preprocessing.log'
# Clear previous log file if it exists
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.INFO, # Log INFO level and above (INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s [%(levelname)s] %(processName)s: %(message)s', # Include timestamp, level, process name
    handlers=[
        logging.FileHandler(LOG_FILE), # Log to file
        logging.StreamHandler(sys.stdout) # Log to console (stdout)
    ]
)
# --- End Logging Configuration ---


# --- Configuration ---
# Paths specific to the remote server 'pioneer'
BASE_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0'
SUBFOLDERS = ['sleep-cassette', 'sleep-telemetry'] # Subdirectories containing the raw data within BASE_DIR
OUTPUT_DIR = '/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleepedf_prepro_all_eld' # Output directory for processed files

# Ensure output directory exists
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory '{OUTPUT_DIR}' ensured.")
except OSError as e:
    logging.error(f"Fatal: Error creating output directory {OUTPUT_DIR}: {e}", exc_info=True)
    sys.exit(1) # Exit if we can't create the output directory


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
    identifier = base_name.split('-')[0]
    dir_path = os.path.dirname(psg_file)
    pattern = os.path.join(dir_path, f"{identifier}*Hypnogram.edf")
    hyp_files = glob.glob(pattern)

    if len(hyp_files) == 1:
        logging.debug(f"Found hypnogram {os.path.basename(hyp_files[0])} for {base_name}")
        return hyp_files[0]
    elif len(hyp_files) > 1:
        logging.warning(f"Multiple hypnogram files found for {base_name}. Using {os.path.basename(hyp_files[0])}")
        return hyp_files[0]
    else:
        logging.warning(f"No hypnogram file found matching pattern {pattern} for {base_name}")
        return None

def process_record(psg_path, hyp_path, target_sfreq, low_freq, high_freq, epoch_length):
    """
    Loads, preprocesses, and epochs a single PSG recording.
    """
    psg_basename = os.path.basename(psg_path)
    hyp_basename = os.path.basename(hyp_path)
    logging.info(f"Processing record: {psg_basename}, Hypnogram: {hyp_basename}")
    raw = None # Initialize raw to None
    try:
        logging.debug(f"Loading EDF: {psg_basename}")
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        logging.info(f"Successfully loaded {psg_basename}. Initial channels: {len(raw.ch_names)}, sfreq: {raw.info['sfreq']} Hz")
    except Exception as e:
        logging.error(f"Error loading {psg_basename}: {e}", exc_info=True)
        return None, None, None

    # Select only EEG and EOG channels
    try:
        logging.debug(f"Picking EEG/EOG types for {psg_basename}")
        picks = mne.pick_types(raw.info, eeg=True, eog=True, meg=False, stim=False, exclude='bads')
        if not picks.size:
             logging.warning(f"Skipping {psg_basename}: No EEG or EOG channels found.")
             del raw
             return None, None, None
        raw.pick(picks)
        logging.info(f"Picked {len(raw.ch_names)} EEG/EOG channels for {psg_basename}: {raw.ch_names}")
    except Exception as e:
         logging.error(f"Error picking EEG/EOG channels for {psg_basename}: {e}", exc_info=True)
         if raw: del raw
         return None, None, None

    initial_sfreq = raw.info['sfreq']
    # Resample if necessary
    if initial_sfreq != target_sfreq:
        logging.info(f"Resampling {psg_basename} from {initial_sfreq} Hz to {target_sfreq} Hz...")
        try:
            raw.resample(target_sfreq, npad="auto", verbose=False)
            logging.debug(f"Resampling complete for {psg_basename}")
        except Exception as e:
            logging.error(f"Error resampling {psg_basename}: {e}", exc_info=True)
            del raw
            return None, None, None

    # Apply band-pass filter
    logging.info(f"Filtering {psg_basename} between {low_freq} Hz and {high_freq} Hz...")
    try:
        raw.filter(l_freq=low_freq, h_freq=high_freq, picks='all', fir_design='firwin', skip_by_annotation='edge', verbose=False)
        logging.debug(f"Filtering complete for {psg_basename}")
    except Exception as e:
        logging.error(f"Error filtering {psg_basename}: {e}", exc_info=True)
        del raw
        return None, None, None

    # Load annotations
    logging.info(f"Loading annotations from {hyp_basename} for {psg_basename}...")
    try:
        ann = mne.read_annotations(hyp_path)
        raw.set_annotations(ann, emit_warning=False)
        logging.debug(f"Annotations set for {psg_basename}. Creating events...")
        events, event_ids = mne.events_from_annotations(
            raw, event_id=ANNOTATION_MAP, chunk_duration=epoch_length, verbose=False
        )
        valid_event_mask = np.isin(events[:, 2], list(event_ids.values()))
        events = events[valid_event_mask]
        if events.shape[0] == 0:
            logging.warning(f"No valid sleep stage annotations found in {psg_basename} after mapping.")
            del raw
            return None, None, None
        logging.info(f"Found {events.shape[0]} valid annotation events for {psg_basename}")
    except Exception as e:
        logging.error(f"Error processing annotations for {psg_basename} ({hyp_basename}): {e}", exc_info=True)
        del raw
        return None, None, None

    # Create epochs
    logging.info(f"Creating epochs for {psg_basename}...")
    tmin = 0.0
    tmax = epoch_length - 1/raw.info['sfreq']
    try:
        epochs = mne.Epochs(
            raw, events=events, event_id=event_ids, tmin=tmin, tmax=tmax,
            baseline=None, preload=True, verbose=False
        )
        logging.info(f"Created {len(epochs)} epochs for {psg_basename}")
    except Exception as e:
        logging.error(f"Error creating epochs for {psg_basename}: {e}", exc_info=True)
        del raw
        return None, None, None

    # Check if any epochs were created
    if len(epochs) == 0:
        logging.warning(f"No epochs created for {psg_basename}, likely due to event timing or data gaps.")
        del raw, epochs
        return None, None, None

    # Get data and labels
    logging.debug(f"Getting data and labels from epochs for {psg_basename}")
    data = epochs.get_data(copy=False)
    labels = epochs.events[:, -1]
    logging.info(f"Data shape: {data.shape}, Labels shape: {labels.shape} for {psg_basename}")

    # Z-score normalization per channel
    logging.info(f"Normalizing channels for {psg_basename}...")
    for ch in range(data.shape[1]):
        channel_data = data[:, ch, :]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std < 1e-6:
             logging.warning(f"Channel {epochs.ch_names[ch]} in {psg_basename} has std near zero. Setting to 0.")
             data[:, ch, :] = 0.0
        else:
             data[:, ch, :] = (channel_data - mean) / std
    logging.debug(f"Normalization complete for {psg_basename}")

    ch_names = epochs.ch_names
    data_copy = data.copy()
    labels_copy = labels.copy()
    ch_names_copy = list(ch_names)

    logging.info(f"Finished processing record {psg_basename}. Cleaning up MNE objects.")
    del epochs, raw
    return data_copy, labels_copy, ch_names_copy


def create_sequences(data, labels, seq_length, seq_stride):
    """Creates sequences of consecutive epochs."""
    n_epochs, n_channels, n_times = data.shape
    logging.debug(f"Creating sequences: data_shape={data.shape}, seq_length={seq_length}, seq_stride={seq_stride}")

    sequences = []
    seq_labels = []

    for start in range(0, n_epochs - seq_length + 1, seq_stride):
        end = start + seq_length
        sequences.append(data[start:end, :, :])
        seq_labels.append(labels[start:end])

    if not sequences:
        logging.warning(f"No sequences created (n_epochs={n_epochs}, seq_length={seq_length})")
        return np.empty((0, seq_length, n_channels, n_times)), np.empty((0, seq_length))

    sequences_np = np.array(sequences)
    seq_labels_np = np.array(seq_labels)
    logging.info(f"Created {len(sequences_np)} sequences. Shape: {sequences_np.shape}")
    return sequences_np, seq_labels_np


def process_and_save(psg_file, output_dir):
    """Processes a single PSG file and saves epochs and sequences."""
    psg_basename = os.path.basename(psg_file)
    logging.info(f"--- Starting pipeline for: {psg_basename} ---")
    status = f"INIT: {psg_basename}" # Initial status

    # Wrap core logic in try-except to catch unexpected errors within this file's processing
    try:
        hyp_file = find_hypnogram(psg_file)
        if not hyp_file:
            # Warning already logged by find_hypnogram
            return f"FAILED_HYPNO: {psg_basename}"

        # Process the record
        data, labels, ch_names = process_record(psg_file, hyp_file,
                                                TARGET_SFREQ, LOW_FREQ, HIGH_FREQ, EPOCH_LENGTH)

        # Check if processing returned valid data
        if data is None or labels is None or ch_names is None:
            logging.warning(f"Processing record failed for {psg_basename}. Skipping saving.")
            return f"FAILED_PROCESS: {psg_basename}"

        logging.info(f"Record processing successful for {psg_basename}. Got data {data.shape}, {len(ch_names)} channels.")
        rec_id = psg_basename.replace('-PSG.edf', '')
        save_error = False

        # --- Save Epochs ---
        epochs_filename = os.path.join(output_dir, f"{rec_id}_epochs.npz")
        logging.info(f"Attempting to save epochs to {epochs_filename}")
        try:
            np.savez_compressed(epochs_filename,
                                data=data.astype('float32'),
                                labels=labels.astype('int8'),
                                ch_names=np.array(ch_names, dtype=object),
                                sfreq=np.array([TARGET_SFREQ]))
            logging.info(f"Successfully saved epochs for {rec_id}")
        except Exception as e:
            logging.error(f"Error saving epochs for {rec_id} to {epochs_filename}: {e}", exc_info=True)
            save_error = True

        # --- Create and Save Sequences ---
        logging.info(f"Attempting to create sequences for {rec_id}")
        sequences, seq_labels = create_sequences(data, labels, SEQ_LENGTH, SEQ_STRIDE)
        sequences_filename = os.path.join(output_dir, f"{rec_id}_sequences.npz")

        if sequences.size == 0:
             logging.warning(f"No sequences generated for {rec_id}.")
             # Decide if this constitutes a save error or not. Let's say no for now.
        else:
            logging.info(f"Attempting to save {sequences.shape[0]} sequences to {sequences_filename}")
            try:
                np.savez_compressed(sequences_filename,
                                    sequences=sequences.astype('float32'),
                                    seq_labels=seq_labels.astype('int8'),
                                    ch_names=np.array(ch_names, dtype=object),
                                    sfreq=np.array([TARGET_SFREQ]))
                logging.info(f"Successfully saved sequences for {rec_id}")
            except Exception as e:
                logging.error(f"Error saving sequences for {rec_id} to {sequences_filename}: {e}", exc_info=True)
                save_error = True
            del sequences, seq_labels # Clean up memory

        # Clean up epoch variables
        del data, labels, ch_names

        status = f"SUCCESS: {psg_basename}" if not save_error else f"FAILED_SAVE: {psg_basename}"

    except Exception as main_e:
        # Catch any unexpected error during the processing of this single file
        logging.error(f"--- Unhandled exception during processing pipeline for {psg_basename} ---", exc_info=True)
        status = f"FAILED_UNHANDLED: {psg_basename}"

    logging.info(f"--- Finished pipeline for: {psg_basename} with status: {status} ---")
    return status


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    logging.info("-" * 60)
    logging.info("Starting Sleep EDF Preprocessing Script")
    logging.info(f"Data Source Base Directory: {BASE_DIR}")
    logging.info(f"Output Directory: {OUTPUT_DIR}")
    logging.info(f"Processing Subfolders: {SUBFOLDERS}")
    logging.info(f"Log File: {LOG_FILE}")
    logging.info("-" * 60)
    logging.info(f"Parameters: Target SFreq={TARGET_SFREQ}Hz, Filter={LOW_FREQ}-{HIGH_FREQ}Hz, Epoch={EPOCH_LENGTH}s")
    logging.info(f"Sequence: Length={SEQ_LENGTH} epochs, Stride={SEQ_STRIDE} epochs")
    logging.info(f"Annotation Map: {ANNOTATION_MAP}")
    logging.info("-" * 60)

    all_psg_files = []
    for subfolder in SUBFOLDERS:
        folder_path = os.path.join(BASE_DIR, subfolder)
        if not os.path.isdir(folder_path):
            logging.warning(f"Subfolder '{folder_path}' not found. Skipping.")
            continue
        logging.info(f"Searching for *PSG.edf files in '{folder_path}'")
        psg_pattern = os.path.join(folder_path, '*PSG.edf')
        found_files = sorted(glob.glob(psg_pattern))
        all_psg_files.extend(found_files)
        logging.info(f"Found {len(found_files)} PSG files in '{folder_path}'")

    if not all_psg_files:
        logging.error("Fatal: No PSG files found in the specified subfolders. Please check paths:")
        logging.error(f"  - BASE_DIR: {BASE_DIR}")
        logging.error(f"  - SUBFOLDERS: {SUBFOLDERS}")
        sys.exit(1) # Exit if no files to process
    else:
        logging.info(f"Found a total of {len(all_psg_files)} PSG files to process.")
        num_workers = os.cpu_count() # Determine default number of workers
        logging.info(f"Detected {num_workers} CPU cores.")
        logging.info("Starting parallel processing using ProcessPoolExecutor (max_workers=None will use available cores)...")
        logging.info("-" * 60)

        process_func = functools.partial(process_and_save, output_dir=OUTPUT_DIR)

        success_count = 0
        fail_count = 0
        results_summary = []
        processed_count = 0

        # Using ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
            # Submit all jobs
            future_to_psg = {executor.submit(process_func, psg_file): psg_file for psg_file in all_psg_files}
            logging.info(f"Submitted {len(future_to_psg)} jobs to the executor.")

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_psg):
                psg_file_path = future_to_psg[future]
                psg_basename = os.path.basename(psg_file_path)
                processed_count += 1
                try:
                    result = future.result() # Get the result string from process_and_save
                    results_summary.append(result)
                    if result.startswith("SUCCESS"):
                        success_count += 1
                        logging.info(f"Job COMPLETED [{processed_count}/{len(all_psg_files)}] - SUCCESS: {psg_basename}")
                    else:
                        fail_count += 1
                        logging.warning(f"Job COMPLETED [{processed_count}/{len(all_psg_files)}] - FAILED ({result.split(':')[0]}): {psg_basename}")

                except Exception as exc:
                    # Catch potential errors from the future/executor itself, though handled errors should return a string
                    fail_count += 1
                    # Extract base filename for logging, even if the full path caused issues
                    try:
                        psg_basename_err = os.path.basename(psg_file_path)
                    except:
                        psg_basename_err = "unknown_file"
                    logging.error(f"Job FAILED [{processed_count}/{len(all_psg_files)}] - EXCEPTION during execution for {psg_basename_err}: {exc}", exc_info=True)
                    results_summary.append(f"FAILED_EXECUTOR: {psg_basename_err}")


    # --- Script Completion Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    logging.info("-" * 60)
    logging.info("Preprocessing Script Finished")
    logging.info(f"Total files attempted: {len(all_psg_files)}")
    logging.info(f"  Successfully processed (incl. saves): {success_count}")
    logging.info(f"  Failed (processing, saving, or other): {fail_count}")
    logging.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if fail_count > 0:
        logging.warning("\nFailures occurred. Check log file ('preprocessing.log') for detailed errors and tracebacks.")
        logging.warning("Failed file summaries reported by processes:")
        # Print only failures for summary
        failure_list = [res for res in results_summary if not res.startswith("SUCCESS")]
        # Limit printing long lists of failures to the log maybe?
        max_failures_to_print = 50
        for i, res in enumerate(failure_list):
             if i < max_failures_to_print:
                 logging.warning(f"  - {res}")
             elif i == max_failures_to_print:
                 logging.warning(f"  - ... (further failures logged in file)")
                 break

    logging.info("-" * 60)