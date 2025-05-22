import argparse
from pathlib import Path
import sys

# To allow running this script directly and importing from parent dirs or sibling packages:
# This assumes 'scripts' is a subdirectory of your project root, and 'configs' is another subdirectory of root.
# Adjust paths as necessary for your project structure.
# PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent # Adjust if cli.py is nested deeper
# sys.path.append(str(PROJECT_ROOT))
# print(f"Temporarily added to sys.path: {PROJECT_ROOT}")

# Attempt to import BaseConfig and Orchestrator
# These imports assume a certain project structure. 
# If running as part of a package, relative imports might be preferred if cli.py is part of the package.
# If running as a standalone script, sys.path manipulation or setting PYTHONPATH might be needed.

try:
    # Assuming configs.base_config and scripts.prepare_dataset.orchestrator are findable
    from configs.base_config import BaseConfig # Adjust if your BaseConfig is elsewhere
    from scripts.prepare_dataset.orchestrator import MELDDatasetOrchestrator
    from scripts.prepare_dataset.audio_converter import extract_meld_wavs_from_mp4_files # Import for the new command
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure that the 'configs' and 'scripts' directories are in your PYTHONPATH, ")
    print("or run this script from a location where these modules can be found.")
    print("If BaseConfig is in 'meld_audiotext_emotion_classifier/configs/base_config.py' and cli.py is in ")
    print("'meld_audiotext_emotion_classifier/scripts/prepare_dataset/cli.py', you might need to run")
    print("this script from the 'meld_audiotext_emotion_classifier' directory using:")
    print("python -m scripts.prepare_dataset.cli --your-args")
    sys.exit(1)

# import logging
# # Configure basic logging if you want to use logger.info etc.
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logger = logging.getLogger(__name__) # Logger for this module

def main():
    parser = argparse.ArgumentParser(description="Build, process, or convert MELD datasets.")
    subparsers = parser.add_subparsers(dest="command", title="Available commands", required=True)

    # --- Subparser for 'prepare' command (existing functionality) ---
    parser_prepare = subparsers.add_parser("prepare", help="Build and process Hugging Face datasets from MELD CSVs and existing WAVs.")
    parser_prepare.add_argument("--dataset_name", type=str, default="meld", help="Name of the dataset (for config).")
    parser_prepare.add_argument("--config_file", type=str, default=None, help="Path to a custom YAML config file.")
    parser_prepare.add_argument("--force_reprocess_items", action="store_true", help="Force reprocessing of all items.")
    parser_prepare.add_argument("--splits_to_process", nargs="+", default=["train", "dev", "test"], help="Dataset splits to process.")
    parser_prepare.add_argument("--num_workers_dataset_map", type=int, default=None, help="Number of workers for .map().")
    parser_prepare.add_argument("--limit_dialogues_train", type=int, default=None, help="Limit train dialogues.")
    parser_prepare.add_argument("--limit_dialogues_dev", type=int, default=None, help="Limit dev dialogues.")
    parser_prepare.add_argument("--limit_dialogues_test", type=int, default=None, help="Limit test dialogues.")
    parser_prepare.add_argument("--audio_input_type", type=str, choices=['raw_wav', 'hf_features'], help="Override config: audio input type.")
    parser_prepare.add_argument("--use_asr", dest='use_asr_for_text_generation_in_hf_dataset', 
                                action=argparse.BooleanOptionalAction, help="Override config: ASR usage (--use-asr/--no-use-asr).")

    # --- Subparser for 'extract-wavs' command (new functionality) ---
    parser_extract_wavs = subparsers.add_parser("extract-wavs", help="Extract WAV audio files from MELD MP4 videos.")
    parser_extract_wavs.add_argument("--dataset_name", type=str, default="meld", help="Name of the dataset (for config).")
    parser_extract_wavs.add_argument("--config_file", type=str, default=None, help="Path to a custom YAML config file.")
    parser_extract_wavs.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="Path to the ffmpeg executable.")
    parser_extract_wavs.add_argument("--force_conversion", action="store_true", help="Force overwrite of existing WAV files.")
    # Add arguments to override specific paths from BaseConfig if needed for WAV extraction, e.g.:
    # parser_extract_wavs.add_argument("--video_raw_data_dir", type=str, help="Override BaseConfig.raw_data_dir for MP4s.")
    # parser_extract_wavs.add_argument("--output_audio_dir", type=str, help="Override BaseConfig.processed_audio_output_dir for WAVs.")

    args = parser.parse_args()

    # --- Initialize Configuration (common for both commands) ---
    print(f"Initializing BaseConfig for command: {args.command}...")
    config_overrides = {}
    if hasattr(args, 'config_file') and args.config_file:
        config_overrides['config_file'] = args.config_file
    
    # Handle specific CLI overrides for 'prepare' command
    if args.command == "prepare":
        cli_prepare_override_keys = ['audio_input_type', 'use_asr_for_text_generation_in_hf_dataset']
        for key in cli_prepare_override_keys:
            if getattr(args, key, None) is not None:
                config_overrides[key] = getattr(args, key)
    
    # Note: If extract-wavs needs to override BaseConfig paths like raw_data_dir or 
    # processed_audio_output_dir via CLI, add those args to parser_extract_wavs 
    # and handle them here for config_overrides similarly.

    try:
        current_cfg = BaseConfig(dataset_name=args.dataset_name, cli_overrides=config_overrides)
        current_cfg._update_dataset_specifics()
        current_cfg._setup_paths() 
        current_cfg._set_device()
        print(f"BaseConfig initialized. Using raw_data_dir: {current_cfg.raw_data_dir}, processed_audio_output_dir: {current_cfg.processed_audio_output_dir}")
    except Exception as e_cfg:
        print(f"Error initializing BaseConfig: {e_cfg}")
        sys.exit(1)

    # --- Execute Command ---
    if args.command == "prepare":
        print("Initializing MELDDatasetOrchestrator for 'prepare' command...")
        try:
            orchestrator = MELDDatasetOrchestrator(config=current_cfg)
            print(f"Starting dataset preparation for splits: {args.splits_to_process}")
            datasets = orchestrator.prepare_all_splits(
                splits_to_process=args.splits_to_process,
                force_reprocess_items=args.force_reprocess_items,
                num_workers_dataset_map=args.num_workers_dataset_map,
                limit_dialogues_train=args.limit_dialogues_train,
                limit_dialogues_dev=args.limit_dialogues_dev,
                limit_dialogues_test=args.limit_dialogues_test
            )
            # ... (rest of your reporting for 'prepare') ...
            print("\n--- Dataset Preparation Summary ---")
            if datasets:
                for split_name, ds_object in datasets.items():
                    if ds_object:
                        print(f"Split '{split_name}': Successfully processed {len(ds_object)} examples.")
                        print(f"  Saved to: {Path(current_cfg.processed_hf_dataset_dir) / split_name}")
                        print(f"  Columns: {ds_object.column_names}")
                    else:
                        print(f"Split '{split_name}': Processing resulted in no dataset or an error.")
            else:
                print("Warning: No datasets were returned from the preparation process.")

        except Exception as e_prepare:
            print(f"Error during 'prepare' command execution: {e_prepare}")
            sys.exit(1)

    elif args.command == "extract-wavs":
        print("Executing 'extract-wavs' command...")
        try:
            extract_meld_wavs_from_mp4_files(
                config=current_cfg, 
                force_conversion=args.force_conversion,
                ffmpeg_path=args.ffmpeg_path
            )
            print(f"WAV extraction process finished. Check output in: {current_cfg.processed_audio_output_dir}")
        except Exception as e_extract_wavs:
            print(f"Error during 'extract-wavs' command execution: {e_extract_wavs}")
            sys.exit(1)
    
    print("CLI script finished.")

if __name__ == "__main__":
    # This structure allows the script to be run directly using `python path/to/cli.py`
    # Make sure that the imports at the top can resolve, potentially by:
    # 1. Running from the project root directory with `python -m scripts.prepare_dataset.cli ...`
    # 2. Having the project root in PYTHONPATH.
    # 3. If `scripts` is part of a package, ensure the package structure is correct for relative imports.
    main() 