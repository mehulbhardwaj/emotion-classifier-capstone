import argparse
from pathlib import Path
import sys
from typing import Optional

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

def main(config: Optional[BaseConfig] = None, command_name: Optional[str] = None, **cmd_kwargs):
    current_cfg = None
    current_command_actual = None # Renamed to avoid conflict with outer scope 'command' if any
    
    # --- Command-specific default arguments (can be overridden by cmd_kwargs or args) ---
    prepare_defaults = {
        "splits_to_process": ["train", "dev", "test"],
        "force_reprocess_items": False,
        "num_workers_dataset_map": None,
        "limit_dialogues_train": None,
        "limit_dialogues_dev": None,
        "limit_dialogues_test": None
    }
    
    extract_wavs_defaults = {
        "force_conversion": False,
        "ffmpeg_path": "ffmpeg"
    }

    # Placeholder for command-specific arguments, to be populated based on execution mode
    parsed_args_for_prepare = {}
    parsed_args_for_extract_wavs = {}

    if config and command_name:
        # --- Mode 1: Called with pre-loaded config and command ---
        print(f"Executing '{command_name}' with pre-loaded configuration.")
        current_cfg = config
        current_command_actual = command_name
        
        if current_command_actual == "prepare":
            for key, default_val in prepare_defaults.items():
                parsed_args_for_prepare[key] = cmd_kwargs.get(key, default_val)
        elif current_command_actual == "extract-wavs":
            for key, default_val in extract_wavs_defaults.items():
                parsed_args_for_extract_wavs[key] = cmd_kwargs.get(key, default_val)

    else:
        # --- Mode 2: Standalone execution with argparse ---
        parser = argparse.ArgumentParser(description="Build, process, or convert MELD datasets.")
        # Note: 'command' used as dest in subparsers can be confusing. Renamed to 'subcommand_action'.
        subparsers = parser.add_subparsers(dest="subcommand_action", title="Available commands", required=True)

        # --- Subparser for 'prepare' command (existing functionality) ---
        parser_prepare = subparsers.add_parser("prepare", help="Build and process Hugging Face datasets from MELD CSVs and existing WAVs.")
        parser_prepare.add_argument("--dataset_name", type=str, default="meld", help="Name of the dataset (for config).")
        parser_prepare.add_argument("--config_file", type=str, default=None, help="Path to a custom YAML config file.")
        parser_prepare.add_argument("--force_reprocess_items", action="store_true", default=prepare_defaults["force_reprocess_items"], help="Force reprocessing of all items.")
        parser_prepare.add_argument("--splits_to_process", nargs="+", default=prepare_defaults["splits_to_process"], help="Dataset splits to process.")
        parser_prepare.add_argument("--num_workers_dataset_map", type=int, default=prepare_defaults["num_workers_dataset_map"], help="Number of workers for .map().")
        parser_prepare.add_argument("--limit_dialogues_train", type=int, default=prepare_defaults["limit_dialogues_train"], help="Limit train dialogues.")
        parser_prepare.add_argument("--limit_dialogues_dev", type=int, default=prepare_defaults["limit_dialogues_dev"], help="Limit dev dialogues.")
        parser_prepare.add_argument("--limit_dialogues_test", type=int, default=prepare_defaults["limit_dialogues_test"], help="Limit test dialogues.")
        parser_prepare.add_argument("--audio_input_type", type=str, choices=['raw_wav', 'hf_features'], help="Override config: audio input type.")
        parser_prepare.add_argument('--use-asr', dest='use_asr_for_text_generation_in_hf_dataset', action='store_true', help="Enable ASR (config default will be overridden).")
        parser_prepare.add_argument('--no-use-asr', dest='use_asr_for_text_generation_in_hf_dataset', action='store_false', help="Disable ASR (config default will be overridden).")
        parser_prepare.set_defaults(use_asr_for_text_generation_in_hf_dataset=None)

        # --- Subparser for 'extract-wavs' command ---
        parser_extract_wavs = subparsers.add_parser("extract-wavs", help="Extract WAV audio files from MELD MP4 videos.")
        parser_extract_wavs.add_argument("--dataset_name", type=str, default="meld", help="Name of the dataset (for config).")
        parser_extract_wavs.add_argument("--config_file", type=str, default=None, help="Path to a custom YAML config file.")
        parser_extract_wavs.add_argument("--ffmpeg_path", type=str, default=extract_wavs_defaults["ffmpeg_path"], help="Path to the ffmpeg executable.")
        parser_extract_wavs.add_argument("--force_conversion", action="store_true", default=extract_wavs_defaults["force_conversion"], help="Force overwrite of existing WAV files.")

        args = parser.parse_args()
        current_command_actual = args.subcommand_action

        # --- Initialize Configuration (only for standalone execution) ---
        print(f"Initializing BaseConfig for command: {current_command_actual} (standalone execution)...")
        yaml_path_from_cli = args.config_file if hasattr(args, 'config_file') else None
        
        effective_cli_args = argparse.Namespace(dataset_name=args.dataset_name if hasattr(args, 'dataset_name') else "meld")
        if current_command_actual == "prepare":
            cli_prepare_override_keys = ['audio_input_type', 'use_asr_for_text_generation_in_hf_dataset']
            for key in cli_prepare_override_keys:
                if hasattr(args, key) and getattr(args, key) is not None:
                    setattr(effective_cli_args, key, getattr(args, key))
        
        try:
            current_cfg = BaseConfig.from_args(args=effective_cli_args, yaml_config_path=yaml_path_from_cli)
            print(f"BaseConfig initialized. Using raw_data_dir: {current_cfg.raw_data_dir}, processed_audio_output_dir: {current_cfg.processed_audio_output_dir}")
            if yaml_path_from_cli and current_cfg.yaml_config_loaded_path:
                print(f"Successfully loaded and applied config from: {current_cfg.yaml_config_loaded_path}")
            elif yaml_path_from_cli:
                print(f"Warning: CLI specified config {yaml_path_from_cli}, but it was not applied (check BaseConfig.from_args or file existence).")
        except Exception as e_cfg:
            print(f"Error initializing BaseConfig: {e_cfg}")
            sys.exit(1)
            
        # Populate args for command execution block from parsed CLI args
        if current_command_actual == "prepare":
            for key in prepare_defaults:
                parsed_args_for_prepare[key] = getattr(args, key, prepare_defaults[key])
        elif current_command_actual == "extract-wavs":
            for key in extract_wavs_defaults:
                parsed_args_for_extract_wavs[key] = getattr(args, key, extract_wavs_defaults[key])

    # --- Execute Command (common logic for both modes) ---
    if not current_cfg or not current_command_actual:
        print("Error: Configuration or command not properly set after parsing.")
        sys.exit(1)

    if current_command_actual == "prepare":
        print("Initializing MELDDatasetOrchestrator for 'prepare' command...")
        try:
            orchestrator = MELDDatasetOrchestrator(config=current_cfg)
            print(f"Starting dataset preparation for splits: {parsed_args_for_prepare['splits_to_process']}")
            datasets = orchestrator.prepare_all_splits(
                splits_to_process=parsed_args_for_prepare['splits_to_process'],
                force_reprocess_items=parsed_args_for_prepare['force_reprocess_items'],
                num_workers_dataset_map=parsed_args_for_prepare['num_workers_dataset_map'],
                limit_dialogues_train=parsed_args_for_prepare['limit_dialogues_train'],
                limit_dialogues_dev=parsed_args_for_prepare['limit_dialogues_dev'],
                limit_dialogues_test=parsed_args_for_prepare['limit_dialogues_test']
            )
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

    elif current_command_actual == "extract-wavs":
        print("Executing 'extract-wavs' command...")
        try:
            extract_meld_wavs_from_mp4_files(
                config=current_cfg, 
                force_conversion=parsed_args_for_extract_wavs['force_conversion'],
                ffmpeg_path=parsed_args_for_extract_wavs['ffmpeg_path']
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