import argparse # Keep for standalone execution
import sys
from pathlib import Path
import pandas as pd

# Adjust sys.path to allow importing from the project root directory structure
SCRIPT_DIR = Path(__file__).resolve().parent  # .../scripts
PROJECT_WORKSPACE_ROOT = SCRIPT_DIR.parent      # .../emotion-classification-dlfa (workspace root)

# Add the workspace root to sys.path to allow absolute imports of the package
if str(PROJECT_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_WORKSPACE_ROOT))

from common.utils import ensure_dir
from configs.base_config import BaseConfig
from common.script_utils import load_meld_csvs, convert_mp4_to_wav_meld

def extract_wavs(config: BaseConfig, force_conversion: bool = False, ffmpeg_path: str = "ffmpeg"):
    """
    Converts MP4 video files from the MELD dataset to WAV audio files.

    Args:
        config (BaseConfig): Configuration object containing dataset paths and settings.
        force_conversion (bool): If True, overwrites existing WAV files.
        ffmpeg_path (str): Path to the ffmpeg executable.
    """
    video_base_raw_data_dir = config.raw_data_dir # Changed: Base for raw data (e.g., MELD.Raw folder)
    output_audio_dir = config.processed_audio_output_dir # Base for processed WAVs like meld_data/processed/audio
    # csv_data_dir is used by load_meld_csvs which should internally use config.raw_data_dir
    # For clarity, ensure load_meld_csvs is consistent with config.raw_data_dir
    csv_data_dir_for_loading = config.raw_data_dir 

    print(f"Starting MELD WAV extraction...")
    print(f"MELD CSVs expected in: {csv_data_dir_for_loading}") # Adjusted print
    print(f"Video subfolders (train_videos, etc.) expected within: {video_base_raw_data_dir}") # Adjusted print
    print(f"Processed WAVs will be saved to subdirectories of: {output_audio_dir}")

    ensure_dir(output_audio_dir) # Ensures meld_data/processed/audio exists

    splits_to_process = ['train', 'dev', 'test']
    
    if not csv_data_dir_for_loading.exists(): # Adjusted variable
        print(f"ERROR: CSV data directory not found: {csv_data_dir_for_loading}. Please check the path. Exiting.")
        return
        
    all_meld_dfs = load_meld_csvs(csv_data_dir_for_loading) # Adjusted variable

    if not all_meld_dfs or all(df.empty for df in all_meld_dfs.values()):
        print(f"No CSV data loaded from {csv_data_dir_for_loading}. Please check the files (e.g., train_sent_emo.csv). Exiting.")
        return

    for split in splits_to_process:
        print(f"\\\\\\nProcessing {split} split...")
        
        df = all_meld_dfs.get(split)
        if df is None or df.empty:
            print(f"No data found for {split} split in the loaded CSVs from {csv_data_dir_for_loading}. Skipping.")
            continue

        # Determine the correct MP4 subdirectory based on the split
        # User confirmed structure is consistently <split>_videos
        mp4_split_dir_name = f"{split}_videos"
            
        mp4_split_dir = video_base_raw_data_dir / mp4_split_dir_name
        wav_output_split_dir = output_audio_dir / split
        
        ensure_dir(wav_output_split_dir)

        if not mp4_split_dir.is_dir():
            print(f"WARNING: MP4 directory for {split} split not found: {mp4_split_dir}. Check if this directory exists and contains videos. Skipping this split.")
            continue
        
        print(f"  Input MP4s directory: {mp4_split_dir}")
        print(f"  Output WAVs directory: {wav_output_split_dir}")

        _ = convert_mp4_to_wav_meld( # Result not currently used
            mp4_base_dir=mp4_split_dir, 
            wav_output_base_dir=wav_output_split_dir,
            meld_csv_df=df.copy(),
            split_name=split, 
            ffmpeg_path=ffmpeg_path,
            force_conversion=force_conversion
        )

    print("\\\\nMELD WAV extraction script finished.")

if __name__ == "__main__":
    # This allows the script to be run standalone, using argparse for overrides
    # or BaseConfig defaults.
    parser = argparse.ArgumentParser(description="Extract WAV audio from MELD MP4s.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Create a default config to get default paths for help messages
    temp_cfg_for_defaults = BaseConfig(dataset_name="meld")

    parser.add_argument(
        "--video_raw_data_dir", type=str, default=str(temp_cfg_for_defaults.raw_data_dir), # Changed to raw_data_dir
        help="Base directory for MELD video subfolders (e.g., meld_data/raw/MELD.Raw/). Overrides config."
    )
    parser.add_argument(
        "--output_audio_dir", type=str, default=str(temp_cfg_for_defaults.processed_audio_output_dir),
        help="Base directory to save WAVs (e.g., meld_data/processed/audio/). Overrides config."
    )
    parser.add_argument(
        "--csv_data_dir", type=str, default=str(temp_cfg_for_defaults.raw_data_dir), # Changed: CSVs are in raw_data_dir
        help="Directory with MELD CSVs (e.g., data/raw/MELD.Raw/). Overrides config."
    )
    parser.add_argument(
        "--ffmpeg_path", type=str, default="ffmpeg",
        help="Path to the ffmpeg executable."
    )
    parser.add_argument(
        "--force_conversion", action="store_true",
        help="Force overwrite of existing WAV files."
    )
    args = parser.parse_args()

    # We'll create a BaseConfig and then manually update its path attributes if CLI args differ from defaults.
    # This is a bit hacky because BaseConfig path properties are derived.
    # A better BaseConfig would allow explicit path setting or have from_args handle these.

    # Create a config instance. For standalone, we might need to adjust its paths based on args.
    # The main `extract_wavs` function expects a config object where properties like `raw_data_dir` are correctly set.
    
    # For standalone run, construct a config object. If CLI args provide paths,
    # those should ideally be used to initialize the BaseConfig, or the BaseConfig
    # needs to be flexible enough.
    # The current BaseConfig uses a root_dir and derives paths.
    # For standalone, we will assume the user provides the *final* paths if they override.

    class StandalonePathConfigAdapter: # Simplified adapter for standalone
        def __init__(self, base_cfg: BaseConfig, cli_args: argparse.Namespace):
            # Prioritize CLI args for paths, fall back to BaseConfig defaults if not provided
            # Ensure paths are Path objects
            self.raw_data_dir = Path(cli_args.video_raw_data_dir) if cli_args.video_raw_data_dir and cli_args.video_raw_data_dir != str(base_cfg.raw_data_dir) else base_cfg.raw_data_dir
            self.processed_audio_output_dir = Path(cli_args.output_audio_dir) if cli_args.output_audio_dir and cli_args.output_audio_dir != str(base_cfg.processed_audio_output_dir) else base_cfg.processed_audio_output_dir
            # raw_csv_data_dir is essentially raw_data_dir for MELD CSVs
            # The load_meld_csvs function should expect the root of MELD.Raw where CSVs are.
            
            # Copy other necessary attributes from base_cfg
            self.dataset_name = base_cfg.dataset_name
            # Add any other attributes `extract_wavs` might indirectly access from config.
            # Specifically, `raw_csv_data_dir` was used by extract_wavs. Let's ensure it's set appropriately.
            # Since CSVs are in raw_data_dir for MELD:
            self.raw_csv_data_dir = self.raw_data_dir 


    # Use temp_cfg_for_defaults as the base for the adapter
    effective_config_standalone = StandalonePathConfigAdapter(temp_cfg_for_defaults, args)


    extract_wavs(config=effective_config_standalone, # Pass the potentially modified config
                 force_conversion=args.force_conversion,
                 ffmpeg_path=args.ffmpeg_path) 