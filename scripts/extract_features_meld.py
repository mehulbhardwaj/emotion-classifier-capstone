import argparse
import os
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
from common.script_utils import (
    load_meld_csvs,
    init_feature_extractors_globals,
    preprocess_and_cache_meld_features
)

# It seems the old script relied on specific functions from an old data_loader and config.
# We'll need to adapt it to use MELDDataModule or refactor those functions into common.utils or scripts.
# For now, let's assume we need `load_meld_csvs`, `init_feature_extractors_globals`, 
# `preprocess_and_cache_meld_features` which were in the old `meld_audiotext_emotion_classifier.data_loader`
# These will need to be moved to common.utils or be part of a script-specific utility.

# Placeholder: Define these functions or ensure they are available from common modules
# For this merge, we'll assume they will be moved to common.utils or similar if not in MELDDataModule
# This part requires careful refactoring of where those functions now live.
# For now, this script will likely be broken until those functions are placed correctly.

# Let's try to get required path from a BaseConfig instance
cfg = BaseConfig()
MELD_RAW_DATA_DIR_PATH = cfg.raw_data_dir
MELD_AUDIO_DIR_PATH = cfg.processed_audio_output_dir
PROCESSED_FEATURES_DIR_PATH = cfg.processed_features_dir
DEVICE_STR = cfg.device_name
AUDIO_MODEL_NAME_CFG = cfg.audio_encoder_model_name
TEXT_MODEL_NAME_CFG = cfg.text_encoder_model_name

# --- Functions that were previously in meld_audiotext_emotion_classifier.data_loader ---
# These need to be defined here, or imported from their new location (e.g., common.utils or a new script_utils.py)

# For the purpose of this step, I cannot redefine the full functions here.
# This script will need further refactoring to call the correct, newly placed functions.
# The original script's `main()` function calls:
# - init_feature_extractors_globals(device=DEVICE)
# - load_meld_csvs(csv_dir)
# - preprocess_and_cache_meld_features(...)

# This highlights a dependency: the logic from the old data_loader.py needs to be migrated first
# or this script needs to be significantly adapted.

def main():
    parser = argparse.ArgumentParser(description="Extract features from MELD dataset (WavLM for audio, Whisper for text).",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--csv_data_dir",
        type=str,
        default=str(MELD_RAW_DATA_DIR_PATH),
        help="Directory containing the original MELD CSV files (e.g., data/raw)."
    )
    parser.add_argument(
        "--processed_audio_dir",
        type=str,
        default=str(MELD_AUDIO_DIR_PATH),
        help="Base directory where processed WAV files are stored (expects train/dev/test subfolders)."
    )
    parser.add_argument(
        "--feature_output_dir",
        type=str,
        default=str(PROCESSED_FEATURES_DIR_PATH),
        help="Base directory to save extracted features (subfolders train/dev/test will be created)."
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force reprocessing of features even if cached files exist."
    )
    args = parser.parse_args()

    print(f"Starting MELD feature extraction...")
    print(f"  Device for feature extraction: {DEVICE_STR}")
    print(f"  Input CSVs directory: {args.csv_data_dir}")
    print(f"  Input WAVs directory: {args.processed_audio_dir}")
    print(f"  Output features directory: {args.feature_output_dir}")

    print("\nInitializing feature extraction models...")
    init_feature_extractors_globals(device=DEVICE_STR, audio_model_name=AUDIO_MODEL_NAME_CFG, text_model_name=TEXT_MODEL_NAME_CFG)

    csv_dir = Path(args.csv_data_dir)
    all_meld_dfs = load_meld_csvs(csv_dir)

    if not all_meld_dfs or all(df.empty for df in all_meld_dfs.values()):
        print(f"No CSV data loaded from {csv_dir}. Cannot proceed with feature extraction. Exiting.")
        return

    base_processed_audio_path = Path(args.processed_audio_dir)
    base_feature_output_path = Path(args.feature_output_dir)

    for split_name in ['train', 'dev', 'test']:
        print(f"\nProcessing {split_name} split...")
        df = all_meld_dfs.get(split_name)

        if df is None or df.empty:
            print(f"No data for {split_name} split. Skipping feature extraction.")
            continue

        def get_wav_path(row):
            return base_processed_audio_path / split_name / f"dia{row.get('Dialogue_ID', '0')}_utt{row.get('Utterance_ID', '0')}.wav"
        
        df['audio_path'] = df.apply(get_wav_path, axis=1)
        
        current_split_dfs = {split_name: df}
        
        split_feature_dir = base_feature_output_path / split_name
        ensure_dir(split_feature_dir)
        
        print(f"  Number of samples in {split_name} split: {len(df)}")
        print(f"  Feature output directory for {split_name}: {split_feature_dir}")

        preprocess_and_cache_meld_features(
            datasets_dfs=current_split_dfs,
            feature_dir=base_feature_output_path, 
            config=cfg,
            force_reprocess=args.force_reprocess,
            device=DEVICE_STR
        )

    print("\nMELD feature extraction script finished.")

if __name__ == "__main__":
    main() 