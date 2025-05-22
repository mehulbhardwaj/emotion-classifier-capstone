import sys
from pathlib import Path
import pandas as pd # Added
from tqdm import tqdm # Added
import subprocess # Added
import shutil # Added
import filetype # Added
# from configs.base_config import BaseConfig # Placeholder
from utils.utils import ensure_dir # Changed
# from common.script_utils import convert_mp4_to_wav_meld # Placeholder, load_meld_csvs removed
from .csv_loader import load_metadata # Added import

# import logging
# logger = logging.getLogger(__name__)

# Function moved from common/script_utils.py
def convert_mp4_to_wav_meld(
    mp4_base_dir: Path, 
    wav_output_base_dir: Path, 
    meld_csv_df: pd.DataFrame, 
    split_name: str, 
    ffmpeg_path: str = "ffmpeg", 
    force_conversion: bool = False
) -> pd.DataFrame:
    """
    Converts MP4 files from MELD dataset to WAV format (16kHz, mono, 16-bit PCM).
    Updates and returns the DataFrame with an 'audio_path' column pointing to the WAV files.
    (Adapted from old data_loader.py)
    """
    print(f"\nStarting MP4 to WAV conversion for {split_name} split...")
    print(f"  Source MP4s directory: {mp4_base_dir}")
    print(f"  Target WAVs directory: {wav_output_base_dir}")

    if not shutil.which(ffmpeg_path):
        print(f"ERROR: ffmpeg executable not found at '{ffmpeg_path}'. Please install ffmpeg or provide correct path.")
        if 'audio_path' not in meld_csv_df.columns:
            meld_csv_df['audio_path'] = pd.NA 
        return meld_csv_df

    ensure_dir(wav_output_base_dir) # Uses imported ensure_dir

    if 'audio_path' not in meld_csv_df.columns:
        meld_csv_df['audio_path'] = pd.NA

    new_audio_paths = [pd.NA] * len(meld_csv_df)
    total_files_to_process = len(meld_csv_df)
    converted_count = 0
    skipped_existing_wav_count = 0
    missing_mp4_count = 0
    conversion_errors = 0
    invalid_mp4_count = 0 # New counter for ffprobe failures

    with tqdm(total=total_files_to_process, desc=f"Converting MP4s for {split_name}") as pbar:
        for index, row in meld_csv_df.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            
            source_mp4_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
            source_mp4_path = mp4_base_dir / source_mp4_filename
            target_wav_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
            target_wav_path = wav_output_base_dir / target_wav_filename
            
            new_audio_paths[index] = str(target_wav_path)
            pbar.update(1)

            if target_wav_path.exists() and not force_conversion:
                skipped_existing_wav_count += 1
                continue 

            if not source_mp4_path.exists():
                if split_name == "test": # Specific MELD test set naming quirk
                    alt_source_mp4_filename = f"final_videos_testdia{dialogue_id}_utt{utterance_id}.mp4"
                    alt_source_mp4_path = mp4_base_dir / alt_source_mp4_filename
                    if alt_source_mp4_path.exists():
                        source_mp4_path = alt_source_mp4_path
                    else:
                        missing_mp4_count += 1
                        new_audio_paths[index] = pd.NA
                        continue
                else:
                    missing_mp4_count += 1
                    new_audio_paths[index] = pd.NA
                    continue
            
            # Use ffprobe to validate the MP4 file
            # Assuming ffprobe is in the same location or PATH as ffmpeg
            ffprobe_path_executable = shutil.which("ffprobe") or shutil.which(Path(ffmpeg_path).parent / "ffprobe")
            if not ffprobe_path_executable:
                # Fallback: if ffprobe specific check fails, try ffmpeg_path parent for ffprobe
                ffprobe_path_executable = Path(ffmpeg_path).parent / "ffprobe"
                if not shutil.which(str(ffprobe_path_executable)): # Check if it's executable
                    print(f"Warning: ffprobe not found. Skipping ffprobe validation for {source_mp4_path}. Will attempt ffmpeg directly.")
                    ffprobe_path_executable = None # proceed without ffprobe check

            if ffprobe_path_executable: # Only run ffprobe if found
                try:
                    ffprobe_command = [
                        str(ffprobe_path_executable),
                        "-v", "error",  # Only print errors
                        "-select_streams", "v:0", # Check for video stream
                        "-show_entries", "stream=codec_type", # Check codec type
                        "-of", "csv=p=0", # Output codec type directly
                        str(source_mp4_path)
                    ]
                    # Timeout for ffprobe, e.g., 10 seconds
                    probe_result = subprocess.run(ffprobe_command, capture_output=True, text=True, check=False, timeout=15)
                    
                    if probe_result.returncode != 0:
                        # ffprobe failed, log error and skip
                        # print(f"Warning: ffprobe failed for {source_mp4_path}. Return code: {probe_result.returncode}. Stderr: {probe_result.stderr.strip()}. Skipping.")
                        invalid_mp4_count += 1
                        new_audio_paths[index] = pd.NA
                        continue
                    # Basic check on output, e.g., if it found a video stream.
                    if "video" not in probe_result.stdout.strip().lower():
                        # print(f"Warning: ffprobe did not identify a video stream in {source_mp4_path}. Output: {probe_result.stdout.strip()}. Skipping.")
                        invalid_mp4_count += 1
                        new_audio_paths[index] = pd.NA
                        continue

                except subprocess.TimeoutExpired:
                    print(f"Warning: ffprobe timed out for {source_mp4_path}. Skipping.")
                    invalid_mp4_count += 1
                    new_audio_paths[index] = pd.NA
                    continue
                except Exception as e_probe: # Catch other potential ffprobe errors
                    print(f"Warning: An error occurred during ffprobe for {source_mp4_path}: {e_probe}. Skipping.")
                    invalid_mp4_count += 1
                    new_audio_paths[index] = pd.NA
                    continue
            
            # Removed filetype.guess() block as ffprobe is more robust.
            # If ffprobe is not found, we proceed directly to ffmpeg conversion attempt.

            try:
                command = [
                    str(ffmpeg_path), "-y", 
                    "-i", str(source_mp4_path),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(target_wav_path)
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    converted_count += 1
                else:
                    print(f"ERROR converting {source_mp4_path}. ffmpeg stderr: {result.stderr}")
                    conversion_errors += 1
                    new_audio_paths[index] = pd.NA
            except Exception as e:
                print(f"Exception during ffmpeg conversion for {source_mp4_path}: {e}")
                conversion_errors += 1
                new_audio_paths[index] = pd.NA
        
    meld_csv_df['audio_path'] = new_audio_paths

    print(f"\nMP4 to WAV conversion summary for {split_name}:")
    print(f"  Total utterances: {total_files_to_process}, Converted: {converted_count}, Skipped existing: {skipped_existing_wav_count}")
    print(f"  MP4s missing: {missing_mp4_count}, Invalid (ffprobe): {invalid_mp4_count}, Conversion errors (ffmpeg): {conversion_errors}")
    print(f"  DataFrame now has {meld_csv_df['audio_path'].notna().sum()} valid audio paths for {split_name} split.")
    return meld_csv_df 


def extract_meld_wavs_from_mp4_files(
    config, # Expects a BaseConfig-like object
    force_conversion: bool = False, 
    ffmpeg_path: str = "ffmpeg"
):
    """
    Converts MP4 video files from the MELD dataset to WAV audio files.
    This function is adapted from the original scripts/extract_meld_wavs_from_mp4s.py.

    Args:
        config: Configuration object (e.g., BaseConfig instance) containing dataset paths 
                (config.raw_data_dir, config.processed_audio_output_dir) and settings.
        force_conversion (bool): If True, overwrites existing WAV files.
        ffmpeg_path (str): Path to the ffmpeg executable.
    """
    # These paths should now come directly from the passed config object
    video_base_raw_data_dir = Path(config.raw_data_dir) 
    output_audio_dir = Path(config.processed_audio_output_dir)
    # load_meld_csvs should internally use config.raw_data_dir for consistency
    csv_data_dir_for_loading = Path(config.raw_data_dir)

    # logger.info(f"Starting MELD WAV extraction via audio_converter...")
    # logger.info(f"  MELD CSVs expected in: {csv_data_dir_for_loading}")
    # logger.info(f"  Video subfolders (train_videos, etc.) expected within: {video_base_raw_data_dir}")
    # logger.info(f"  Processed WAVs will be saved to subdirectories of: {output_audio_dir}")
    print(f"Starting MELD WAV extraction via audio_converter...")
    print(f"  MELD CSVs expected in: {csv_data_dir_for_loading}")
    print(f"  Video subfolders (train_videos, etc.) expected within: {video_base_raw_data_dir}")
    print(f"  Processed WAVs will be saved to subdirectories of: {output_audio_dir}")
    
    # Use a local ensure_dir or import the proper one
    # ensure_dir(output_audio_dir) # Assumes ensure_dir from utils.utils
    ensure_dir(output_audio_dir) # Changed


    # This import needs to be resolvable. 
    # If common.script_utils is not in sys.path or not part of the package, this will fail.
    # For now, assuming it can be imported.
    try:
        # from common.script_utils import load_meld_csvs, convert_mp4_to_wav_meld
        # from common.script_utils import convert_mp4_to_wav_meld # load_meld_csvs removed # This import is no longer needed
        pass # convert_mp4_to_wav_meld is now local
    except ImportError: # This except block might be removable if no other imports from common.script_utils are attempted
        # logger.critical("Could not import load_meld_csvs or convert_mp4_to_wav_meld from common.script_utils. WAV extraction cannot proceed.")
        # print("CRITICAL: Could not import from common.script_utils. Ensure it\'s in PYTHONPATH or installed.")
        # print("CRITICAL: Could not import convert_mp4_to_wav_meld from common.script_utils. Ensure it\\'s in PYTHONPATH or installed.")
        # return # Not returning here as the specific import was removed; if other critical imports fail, they'd raise their own error.
        print("Note: An import from common.script_utils was removed as the function is now local. If other errors occur, check other imports.")

    splits_to_process = getattr(config, 'default_splits_to_process', ['train', 'dev', 'test'])
    
    if not csv_data_dir_for_loading.exists(): # This check might still be useful for the general raw data dir
        # logger.error(f"CSV data directory not found: {csv_data_dir_for_loading}. Exiting WAV extraction.")
        print(f"ERROR: Raw data directory for CSVs not found: {csv_data_dir_for_loading}. Exiting WAV extraction.")
        return
        
    # all_meld_dfs = load_meld_csvs(csv_data_dir_for_loading) # Removed

    # if not all_meld_dfs or all(df.empty for df in all_meld_dfs.values()): # Removed
        # logger.warning(f"No CSV data loaded from {csv_data_dir_for_loading}. Check files. Exiting WAV extraction.")
        # print(f"Warning: No CSV data loaded from {csv_data_dir_for_loading}. Check files. Exiting WAV extraction.")
        # return

    for split in splits_to_process:
        # logger.info(f"\nProcessing WAV extraction for {split} split...")
        print(f"\nProcessing WAV extraction for {split} split...")
        
        # df = all_meld_dfs.get(split) # Replaced
        # Load metadata for the current split using the new loader
        df = load_metadata(split=split, config=config, limit_dialogues=None) # limit_dialogues can be added if needed for this script

        if df is None or df.empty:
            # logger.warning(f"No data found for {split} split in CSVs from {csv_data_dir_for_loading}. Skipping WAV extraction for this split.")
            # print(f"Warning: No data found for {split} split in CSVs from {csv_data_dir_for_loading}. Skipping WAV extraction for this split.")
            print(f"Warning: No metadata loaded for {split} split using load_metadata. Skipping WAV extraction for this split.")
            continue

        # MELD specific: MP4s are usually in <split>_videos folders (e.g., train_videos)
        # This logic was in the original script and seems standard for MELD.
        mp4_split_dir_name = f"{split}_videos" 
            
        mp4_split_dir = video_base_raw_data_dir / mp4_split_dir_name
        wav_output_split_dir = output_audio_dir / split # Output to train/, dev/, test/ directly
        
        # ensure_dir(wav_output_split_dir)
        ensure_dir(wav_output_split_dir) # Changed


        if not mp4_split_dir.is_dir():
            # logger.warning(f"MP4 directory for {split} split not found: {mp4_split_dir}. Skipping WAVs for this split.")
            print(f"WARNING: MP4 directory for {split} split not found: {mp4_split_dir}. Skipping WAVs for this split.")
            continue
        
        # logger.debug(f"  Input MP4s directory: {mp4_split_dir}")
        # logger.debug(f"  Output WAVs directory: {wav_output_split_dir}")
        print(f"  Input MP4s directory: {mp4_split_dir}")
        print(f"  Output WAVs directory: {wav_output_split_dir}")

        # Call the conversion utility (assumed to be imported from common.script_utils)
        conversion_results = convert_mp4_to_wav_meld(
            mp4_base_dir=mp4_split_dir, 
            wav_output_base_dir=wav_output_split_dir,
            meld_csv_df=df.copy(), # Pass the DataFrame for this split
            split_name=split, 
            ffmpeg_path=ffmpeg_path,
            force_conversion=force_conversion
        )
        # logger.info(f"Conversion for {split} reported: {conversion_results}") # Example of using results

    # logger.info("\nMELD WAV extraction process finished.")
    print("\nMELD WAV extraction process finished.")


if __name__ == "__main__":
    # This section is for testing audio_converter.py directly.
    # It requires BaseConfig and the common.script_utils to be importable.
    # And, of course, actual MELD data (CSVs and MP4s) in the configured paths.
    
    print("Testing audio_converter.py standalone...")

    # For standalone testing, we need a BaseConfig instance.
    # Adjust PROJECT_ROOT and sys.path if BaseConfig is in a different location relative to this script.
    # PROJECT_ROOT_FOR_TEST = Path(__file__).resolve().parent.parent.parent 
    # if str(PROJECT_ROOT_FOR_TEST) not in sys.path:
    #     sys.path.insert(0, str(PROJECT_ROOT_FOR_TEST))
    #     print(f"Added to sys.path for test: {PROJECT_ROOT_FOR_TEST}")

    try:
        from configs.base_config import BaseConfig
        # Create a dummy config for testing, assuming MELD data is at "data/meld_raw_test"
        # and output should go to "data/meld_processed_test/audio"
        
        # You MUST adjust these paths for your actual test MELD data location.
        # Create a temporary config file or use a BaseConfig that points to test data.
        class TestConfig(BaseConfig):
            def __init__(self):
                super().__init__(dataset_name="meld_test_audio_conversion")
                # Override paths for a specific test data location if necessary
                # self.data_root = Path("temp_meld_conversion_test_data") 
                # self._setup_paths() # Re-run path setup if data_root changes
                # print(f"TestConfig raw_data_dir: {self.raw_data_dir}")
                # print(f"TestConfig processed_audio_output_dir: {self.processed_audio_output_dir}")

                # Minimal dummy data setup for the test config to work
                # (self.data_root / "raw").mkdir(parents=True, exist_ok=True)
                # (self.data_root / "processed_audio_16kHz_mono" / "train").mkdir(parents=True, exist_ok=True)
                # Dummy CSV (must match expected format for load_meld_csvs)
                # import pandas as pd
                # dummy_train_csv = self.raw_data_dir / "train_sent_emo.csv"
                # if not dummy_train_csv.exists():
                #     pd.DataFrame({'Dialogue_ID': [0], 'Utterance_ID': [0], 'Speaker': ['S1'], 
                #                     'Utterance':['Test'],'Emotion':['neutral'],'Sentiment':['neutral']})
                #     .to_csv(dummy_train_csv, index=False)
                # Dummy MP4 dir
                # (self.raw_data_dir / "train_videos").mkdir(parents=True, exist_ok=True)
                # Create a tiny dummy MP4 if ffmpeg is available and you want to test actual conversion
                # Otherwise, convert_mp4_to_wav_meld might need to handle missing ffmpeg gracefully or mock conversion.

        test_cfg = TestConfig() # Create instance of your test config
        
        # If your BaseConfig is complex to instantiate, use a simpler mock or pre-existing test config.
        # For this example, let's assume BaseConfig() can be instantiated and points to some valid (even if empty) paths.
        # test_cfg = BaseConfig(dataset_name="meld") # Default MELD config

        print(f"Using config for test: raw_data_dir='{test_cfg.raw_data_dir}', processed_audio_output_dir='{test_cfg.processed_audio_output_dir}'")
        
        # Create dummy directories based on this test_cfg if they don't exist
        # This is crucial for the script to run without I/O errors if data is not present.
        # _ensure_dir_local(test_cfg.raw_data_dir) # Replaced
        ensure_dir(test_cfg.raw_data_dir) # Changed
        # _ensure_dir_local(test_cfg.processed_audio_output_dir) # Replaced
        ensure_dir(test_cfg.processed_audio_output_dir) # Changed
        # Create split subdirs for audio output
        for split in ['train', 'dev', 'test']:
            # _ensure_dir_local(test_cfg.processed_audio_output_dir / split) # Replaced
            ensure_dir(test_cfg.processed_audio_output_dir / split) # Changed
            # Also ensure dummy MP4 video dirs exist for common.script_utils.convert_mp4_to_wav_meld to check
            # _ensure_dir_local(test_cfg.raw_data_dir / f"{split}_videos")  # Replaced
            ensure_dir(test_cfg.raw_data_dir / f"{split}_videos") # Changed
        
        # You might need to create dummy CSV files in test_cfg.raw_data_dir for load_meld_csvs to work.
        # Example: (test_cfg.raw_data_dir / "train_sent_emo.csv").touch(exist_ok=True)
        # (test_cfg.raw_data_dir / "dev_sent_emo.csv").touch(exist_ok=True)
        # (test_cfg.raw_data_dir / "test_sent_emo.csv").touch(exist_ok=True)
        print("NOTE: This test requires dummy MELD CSVs (e.g., train_sent_emo.csv) in the raw_data_dir ")
        print("and dummy MP4 video folders (e.g., train_videos) for full execution.")
        print("The common.script_utils.convert_mp4_to_wav_meld must also be functional.")

        extract_meld_wavs_from_mp4_files(
            config=test_cfg, 
            force_conversion=False, # Set to True to test overwriting
            ffmpeg_path="ffmpeg"  # Ensure ffmpeg is in PATH or provide full path
        )
        print("\nStandalone test of audio_converter.py finished.")
        print(f"Check for WAV files (if any were converted) under: {test_cfg.processed_audio_output_dir}")

    except ImportError as e_imp:
        print(f"ImportError during standalone test setup: {e_imp}. Ensure BaseConfig and common.script_utils are importable.")
    except Exception as e_test:
        print(f"An error occurred during the standalone test: {e_test}") 