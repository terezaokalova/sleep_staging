import os
import re
import glob
import logging
import time
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime

try:
    import pyedflib
    print(f"pyedflib is installed (version: {pyedflib.__version__})")
except ImportError:
    print("ERROR: pyedflib is NOT installed. Run 'pip install pyedflib'")
    sys.exit(1)

BASE_DATA_DIR = "/users/okalova/sleep/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0"
CASSETTE_DIR_NAME = "sleep-cassette"
TELEMETRY_DIR_NAME = "sleep-telemetry"
OUTPUT_DIR_NAME = "processed_output"
LOG_FILENAME = f"preprocessing_full_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
MAX_WORKERS = 16

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(processName)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILENAME, mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout) 
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

for handler in logger.handlers:
    handler.flush()

def find_hypnogram(psg_filepath):
    psg_filename = os.path.basename(psg_filepath)
    dir_path = os.path.dirname(psg_filepath)

    match = re.match(r"(SC|ST)(\d{4})", psg_filename)
    if not match:
        return None, f"Could not extract core ID from PSG filename: {psg_filename}"
    core_id = match.group(0)

    hypnogram_pattern = f"{core_id}*-Hypnogram.edf"
    search_pattern = os.path.join(dir_path, hypnogram_pattern)

    try:
        found_hypno_files = glob.glob(search_pattern)
    except Exception as e:
        return None, f"Error during glob search for {search_pattern}: {e}"

    if not found_hypno_files:
        return None, f"No hypnogram file found matching pattern {search_pattern}"
    elif len(found_hypno_files) > 1:
        file_list = ", ".join([os.path.basename(f) for f in found_hypno_files])
        return None, f"Multiple hypnograms found matching {search_pattern}: [{file_list}]. Skipping."
    else:
        return found_hypno_files[0], None 

def process_file_pair(psg_filepath, hypno_filepath):
    psg_filename = os.path.basename(psg_filepath)
    hypno_filename = os.path.basename(hypno_filepath)
    
    try:
        logging.info(f"Processing pair: PSG='{psg_filename}', Hypnogram='{hypno_filename}'")
        
        if not os.path.exists(psg_filepath):
            return False, f"PSG file disappeared: {psg_filename}"
        if not os.path.exists(hypno_filepath):
            return False, f"Hypno file disappeared: {hypno_filename}"
        
        base_dir = os.path.dirname(os.path.dirname(psg_filepath))
        output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
        os.makedirs(output_dir, exist_ok=True)
        
        match = re.match(r"(SC|ST)(\d{4})", psg_filename)
        if not match:
            return False, f"Regex extraction failed for output name: {psg_filename}"
        record_id = match.group(0)
        output_path = os.path.join(output_dir, f"{record_id}_processed.npz")
        
        file_size_mb = os.path.getsize(psg_filepath) / (1024 * 1024)
        logging.info(f"Reading PSG file: {psg_filename} ({file_size_mb:.2f} MB)")
        
        with pyedflib.EdfReader(psg_filepath) as psg_file:
            n_signals = psg_file.signals_in_file
            signal_labels = psg_file.getSignalLabels()
            logging.info(f"Found {n_signals} signals: {', '.join(signal_labels)}")
            
            signals = []
            signal_headers = []
            for i in range(n_signals):
                signals.append(psg_file.readSignal(i))
                signal_headers.append(psg_file.getSignalHeader(i))
            
            if signal_headers and len(signal_headers) > 0:
                logging.info(f"Header keys: {list(signal_headers[0].keys())}")
            
            # Try different possible keys for sampling frequency
            try:
                sampling_freqs = [header['sample_frequency'] for header in signal_headers]
            except KeyError:
                try:
                    sampling_freqs = [header['fs'] for header in signal_headers]
                except KeyError:
                    try:
                        sampling_freqs = [header['samplefrequency'] for header in signal_headers]
                    except KeyError:
                        try:
                            sampling_freqs = [header['sample_rate'] for header in signal_headers]
                        except KeyError:
                            logging.warning(f"Could not find sampling frequency in headers for {psg_filename}")
                            sampling_freqs = [100.0] * len(signal_headers)
            
            logging.info(f"Sampling frequencies: {sampling_freqs}")
        
        with pyedflib.EdfReader(hypno_filepath) as hypno_file:
            hypnogram = hypno_file.readSignal(0)
            hypno_header = hypno_file.getSignalHeader(0)
            
            try:
                hypno_annotations = hypno_file.readAnnotations()
                logging.info(f"Found {len(hypno_annotations[0])} annotations in hypnogram")
            except Exception as e:
                logging.warning(f"Could not read annotations: {e}")
                hypno_annotations = None
        
        logging.info(f"Saving processed data to: {output_path}")
        np.savez(output_path,
                 signals=signals,
                 signal_labels=signal_labels,
                 sampling_freqs=sampling_freqs,
                 hypnogram=hypnogram,
                 hypno_annotations=hypno_annotations)
        
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logging.info(f"Successfully saved: {output_path} ({output_size_mb:.2f} MB)")
        return True, "PROCESSED_OK"
        
    except Exception as e:
        logging.error(f"Error processing {psg_filename}: {e}", exc_info=True)
        return False, f"PROCESSING_ERROR: {e}"

def verify_output_file(output_path):
    try:
        data = np.load(output_path)
        info_str = f"Verification of {os.path.basename(output_path)}:\n"
        info_str += f"  - Found {len(data.files)} arrays: {', '.join(data.files)}\n"
        info_str += f"  - Signals shape: {data['signals'].shape}\n"
        info_str += f"  - Hypnogram length: {len(data['hypnogram'])}"
        logging.info(info_str)
        return True
    except Exception as e:
        logging.error(f"Verification failed for {output_path}: {e}")
        return False

def worker_task(psg_filepath):
    psg_filename = os.path.basename(psg_filepath)
    logging.info(f"--- Starting pipeline for: {psg_filename} ---")

    hypno_filepath, find_error_msg = find_hypnogram(psg_filepath)

    if hypno_filepath is None:
        logging.warning(f"{find_error_msg} for {psg_filename}")
        return (psg_filename, "FAILED_HYPNO", find_error_msg) 

    success, message_code = process_file_pair(psg_filepath, hypno_filepath)

    if success:
        logging.info(f"Processing successful for {psg_filename}")
        return (psg_filename, message_code, f"Successfully processed {psg_filename}")
    else:
        logging.warning(f"Processing failed for {psg_filename} with code {message_code}")
        return (psg_filename, message_code, f"Failed during processing step for {psg_filename}")

def main():
    start_time = time.time()
    logging.info(f"Base Data Directory: {BASE_DATA_DIR}")
    logging.info(f"Using up to {MAX_WORKERS} workers.")

    output_dir = os.path.join(BASE_DATA_DIR, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    cassette_path = os.path.join(BASE_DATA_DIR, CASSETTE_DIR_NAME)
    telemetry_path = os.path.join(BASE_DATA_DIR, TELEMETRY_DIR_NAME)

    try:
        psg_files_cassette = glob.glob(os.path.join(cassette_path, "SC*PSG.edf"))
        psg_files_telemetry = glob.glob(os.path.join(telemetry_path, "ST*PSG.edf"))
        all_psg_files = sorted(psg_files_cassette + psg_files_telemetry)
        total_files = len(all_psg_files)
    except Exception as e:
        logging.error(f"Fatal error finding PSG files: {e}", exc_info=True)
        return

    if total_files == 0:
        logging.warning("No PSG files found in the specified directories. Check paths:")
        logging.warning(f" - Cassette: {cassette_path}")
        logging.warning(f" - Telemetry: {telemetry_path}")
        logging.warning("Exiting.")
        return

    logging.info(f"Found {total_files} PSG files to process.")

    results = []
    processed_count = 0
    success_code = "PROCESSED_OK"
    success_count = 0
    failure_count = 0
    failed_files_summary = {}

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_task, psg_file): psg_file for psg_file in all_psg_files}

        for future in as_completed(futures):
            psg_filepath_orig = futures[future]
            processed_count += 1
            
            if processed_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                est_remaining = (total_files - processed_count) / rate if rate > 0 else "unknown"
                logging.info(f"Progress: {processed_count}/{total_files} files ({processed_count/total_files*100:.1f}%). " +
                           f"Rate: {rate:.2f} files/sec. Est. remaining: {est_remaining if isinstance(est_remaining, str) else f'{est_remaining:.1f} sec'}")
                
                for handler in logger.handlers:
                    handler.flush()
            
            try:
                result = future.result()
                results.append(result)
                psg_filename, status_code, details_message = result

                if status_code == success_code:
                    success_count += 1
                else:
                    failure_count += 1
                    logging.warning(f"Job COMPLETED [{processed_count}/{total_files}] - FAILED ({status_code}): {psg_filename}")
                    if status_code not in failed_files_summary:
                        failed_files_summary[status_code] = []
                    failed_files_summary[status_code].append((psg_filename, details_message))

            except Exception as exc:
                psg_filename = os.path.basename(psg_filepath_orig)
                failure_count += 1
                status_code = "EXCEPTION_IN_FUTURE"
                logging.error(f"Job COMPLETED [{processed_count}/{total_files}] - FAILED ({status_code}): Task for {psg_filename} generated an exception: {exc}", exc_info=True)
                results.append((psg_filename, status_code, str(exc)))
                if status_code not in failed_files_summary:
                     failed_files_summary[status_code] = []
                failed_files_summary[status_code].append((psg_filename, str(exc)))

    end_time = time.time()
    total_time = end_time - start_time

    logging.info("Preprocessing Script Finished")
    logging.info(f"Total files attempted: {total_files}")
    logging.info(f"   Successfully processed: {success_count}")
    logging.info(f"   Failed: {failure_count}")
    logging.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if failure_count > 0:
        logging.warning(" ")
        logging.warning(f"Failures occurred ({failure_count}). Check log file ('{LOG_FILENAME}') for details.")
        logging.warning("Failed file summaries (status: count):")
        for status, files_details in failed_files_summary.items():
             logging.warning(f"   - {status}: {len(files_details)} file(s)")

        max_details_to_print = 10
        for status, files_details in failed_files_summary.items():
            logging.warning(f"   --- Details for status '{status}' ---")
            for i, (filename, detail_msg) in enumerate(files_details):
                 if i < max_details_to_print:
                     logging.warning(f"     - {filename}: {detail_msg}")
                 elif i == max_details_to_print:
                     logging.warning(f"     - ... (further details for {status} logged in file)")
                     break

    if success_count > 0:
        output_files = glob.glob(os.path.join(output_dir, "*_processed.npz"))
        sample_count = min(2, len(output_files))
        if sample_count > 0:
            logging.info(f"Verifying {sample_count} output files for data integrity:")
            for i in range(sample_count):
                verify_output_file(output_files[i])

if __name__ == "__main__":
    main()