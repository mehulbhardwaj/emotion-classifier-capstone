#!/usr/bin/env python3
"""
Main entry point for the emotion classification project.

This script provides a command-line interface for training, evaluating,
and running inference with different model architectures.
"""

import os
import argparse
import torch
import time # Keep time import if used elsewhere, e.g. benchmarking
from pathlib import Path
import fireducks.pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from configs.base_config import BaseConfig
from common.utils import set_seed, ensure_dir
from common.data_loader import MELDDataModule
from common.inference import EmotionInferenceEngine
from architectures import get_model_architecture, get_default_config_class_for_arch, get_available_architectures
from src.callbacks.checkpoint import get_checkpoint_cb

# Define CONFIG_DIR at the top level, after imports
CONFIG_DIR = Path("configs")

# Helper functions from common.cli_handlers (or directly in main.py as before)
# Assuming they are now in common.cli_handlers as per previous refactoring
# from common.cli_handlers import (
#     run_data_preparation,
#     run_eda,
#     run_training,
#     run_evaluation,
#     run_inference
# )
# If _run_... functions are still in this file, the above import is not needed.
# For this full replacement, I'll assume they were moved as it was the last step of our previous refactoring.
# If they are still in main.py, those definitions will be part of this replacement.
# Re-checking the provided main.py context, helper functions are still in main.py. So I will keep them.


# --- Helper Functions for Main Modes (kept in main.py as per user's current file structure) ---

def _run_data_preparation(cfg: BaseConfig, args: argparse.Namespace):
    print("Starting data preparation pipeline...")
    if cfg.run_mp4_to_wav_conversion:
        print("Step 1: Converting MP4s to WAVs...")
        try:
            # Assuming extract_meld_wavs_main is imported or available
            from scripts.extract_meld_wavs_from_mp4s import extract_wavs as extract_meld_wavs_main
            extract_meld_wavs_main(config=cfg,
                                   force_conversion=args.force_wav_conversion,
                                   ffmpeg_path=cfg.ffmpeg_path)
            print("MP4 to WAV conversion complete.")
        except Exception as e:
            print(f"Error during MP4 to WAV conversion: {e}")
    else:
        print("Skipping MP4 to WAV conversion as per configuration.")
        
    if cfg.run_hf_dataset_creation:
        print("\nStep 2: Building Hugging Face dataset...")
        try:
            # Assuming prepare_meld_hf_dataset is imported or available
            from scripts.build_hf_dataset import prepare_meld_hf_dataset
            datasets = prepare_meld_hf_dataset(
                current_cfg=cfg,
                force_reprocess_items=args.force_reprocess_hf_dataset,
                splits_to_process=['train', 'dev', 'test'],
                num_workers_dataset_map=cfg.num_dataloader_workers,
                limit_dialogues_train=args.limit_dialogues_train,
                limit_dialogues_dev=args.limit_dialogues_dev,
                limit_dialogues_test=args.limit_dialogues_test
            )
            print("Hugging Face dataset preparation complete.")
            if datasets:
                for split_name, ds_object in datasets.items():
                    if ds_object:
                        # Ensure processed_hf_dataset_dir is a Path object and points to the parent directory
                        save_path = cfg.processed_hf_dataset_dir / split_name
                        ensure_dir(save_path.parent) # Ensure parent dir (e.g. .../hf_datasets/) exists
                        print(f"  Saving {split_name.capitalize()} dataset ({len(ds_object)} items) to disk at: {save_path}")
                        try:
                            ds_object.save_to_disk(str(save_path))
                            print(f"  {split_name.capitalize()} dataset saved successfully.")
                        except Exception as e_save:
                            print(f"  ERROR: Could not save {split_name} dataset to {save_path}: {e_save}")
                    else:
                        print(f"  No dataset object returned for {split_name} split, cannot save.")
            else:
                print("  prepare_meld_hf_dataset returned no datasets dict, nothing to save.")

            # Original printout of dataset lengths (can be redundant now but kept for consistency)
            if datasets: 
                for split_name, ds_object_print in datasets.items(): # Renamed ds to ds_object_print for clarity
                    print(f"  {split_name.capitalize()} split from returned dict: {len(ds_object_print)} items processed.")

        except Exception as e:
            print(f"Error during prepare_meld_hf_dataset: {e}")
    else:
        print("Skipping Hugging Face dataset creation as per configuration.")
    
def _run_eda(cfg: BaseConfig, args: argparse.Namespace):
    print("Running Exploratory Data Analysis...")
    print("Step 1: Preliminary EDA on raw data...")
    try:
        from scripts.preliminary_eda import main as preliminary_eda_main
        preliminary_eda_main(cfg_param=cfg)
    except Exception as e:
        print(f"Error during preliminary_eda_main: {e}")

    print("\nStep 2: EDA on processed Hugging Face features...")
    try:
        from scripts.processed_features_eda import main as processed_features_eda_main
        processed_features_eda_main(cfg_param=cfg)
    except Exception as e:
        print(f"Error during processed_features_eda_main: {e}")

def _run_training(cfg: BaseConfig, args: argparse.Namespace, model_class, trainer_class):
    print(f"Training model: {args.architecture} using PyTorch Lightning.")
    
    # 1. Instantiate DataModule
    data_module = MELDDataModule(cfg)
    # prepare_data and setup are called by PL Trainer if needed, 
    # but explicit call here ensures data is ready, good for debugging.
    # data_module.prepare_data() # Called by PL if implemented in DM
    # data_module.setup(stage='fit') # Called by PL if implemented in DM

    # 2. Instantiate Model (LightningModule)
    # model_class is MultimodalFusionMLP here
    model = model_class(cfg=cfg) 

    # 3. Instantiate PyTorch Lightning Trainer
    ckpt_dir = cfg.model_save_dir
    # ckpt_dir.mkdir(parents=True, exist_ok=True) # get_checkpoint_cb or PL Trainer might handle this. BaseConfig already does.
    
    checkpoint_cb = get_checkpoint_cb(str(ckpt_dir))
    
    # Setup other callbacks like EarlyStopping if desired
    # from pytorch_lightning.callbacks.early_stopping import EarlyStopping
    # early_stop_callback = EarlyStopping(monitor="val_wf1", mode="max", patience=cfg.early_stopping_patience)

    trainer_args = {
        'accelerator': cfg.device.type, # 'cpu', 'gpu', 'mps'
        'devices': 1, # Assuming single device
        'max_epochs': cfg.num_epochs,
        'callbacks': [checkpoint_cb], # Add early_stop_callback if used
        'deterministic': True, # For reproducibility, pairs well with set_seed
        'log_every_n_steps': 50, # How often to log within an epoch
    }
    
    # Configure logger
    logger = TensorBoardLogger(save_dir="logs/", name=cfg.experiment_name, version="")
    trainer_args['logger'] = logger

    if cfg.mixed_precision_training and cfg.device.type == 'cuda':
        trainer_args['precision'] = 16 # Or "bf16"

    pl_trainer = pl.Trainer(**trainer_args)

    # 4. Start Training
    print("Starting PyTorch Lightning training...")
    pl_trainer.fit(model, datamodule=data_module)
    print("Training complete.")
    
    # 5. Evaluate on the test set using the best checkpoint
    eval_split_after_train = getattr(cfg, 'eval_split_after_train', 'test') # Default to 'test'
    print(f"Evaluating on {eval_split_after_train} set after training using best checkpoint...")
    # datamodule.setup(stage='test') # Ensure test data is set up if not already
    # The datamodule passed to .test() should provide the correct dataloaders.
    test_results = pl_trainer.test(model, datamodule=data_module, ckpt_path='best')
    
    print(f"Metrics on {eval_split_after_train} set (from best checkpoint):")
    if test_results:
        for key, value in test_results[0].items(): # test_results is a list of dicts
            print(f"  {key}: {value:.4f}")

    # --- Old custom trainer logic (to be removed for mlp_fusion) ---
    # if trainer_class.__name__ == "MLPFusionTrainer" and args.architecture != "mlp_fusion_legacy":
    #     print("MLPFusionTrainer is being phased out for mlp_fusion. Using pl.Trainer.")
    #     # The above PL logic handles it.
    # elif trainer_class: # For other architectures if they still use custom trainers
    #     print(f"Using custom trainer: {trainer_class.__name__}")
    #     # Retain logic for other custom trainers if they exist and are not PL based.
    #     # This part needs to be architected carefully if mixing PL and custom.
    #     # For now, assuming all Path A refactors will use PL.
    #     dataloaders = data_module.get_dataloaders() # Custom trainers might need this
    #     custom_trainer_instance = trainer_class(cfg=cfg) # Or (cfg, model_class)
    #     custom_trainer_instance.train(
    #         train_dataloader=dataloaders['train'],
    #         val_dataloader=dataloaders['dev']
    #     )
    #     print("Custom training complete.")
    #     # ... custom evaluation ...
    # else:
    #     print(f"No specific trainer logic implemented for {args.architecture} in this PL path, or trainer_class is None.")

def _run_evaluation(cfg: BaseConfig, args: argparse.Namespace, model_class, trainer_class): # trainer_class unused for PL
    print(f"Evaluating model: {args.architecture} from checkpoint: {args.evaluate_model}")
    model_checkpoint_path = Path(args.evaluate_model)
    if not model_checkpoint_path.exists():
        print(f"Error: Model checkpoint {model_checkpoint_path} not found.")
        return
    
    # 1. Load model from checkpoint
    # model_class is MultimodalFusionMLP (LightningModule)
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path=str(model_checkpoint_path),
            cfg=cfg # Pass cfg if model's __init__ needs it and it wasn't saved in hparams
                    # or if you need to override some hparams.
                    # If save_hyperparameters(cfg_dict) was used, PL can often infer.
        )
        model.cfg = cfg # Ensure cfg is set on the loaded model, as load_from_checkpoint might not restore it if not in hparams
        print(f"Model loaded successfully from {model_checkpoint_path}")
    except Exception as e:
        print(f"Error loading model from checkpoint {model_checkpoint_path}: {e}")
        # Fallback: try loading with cfg if direct load fails (less common for PL modules)
        # try:
        #     print("Attempting to load model by first instantiating with cfg and then loading state_dict...")
        #     model = model_class(cfg=cfg)
        #     checkpoint = torch.load(model_checkpoint_path, map_location=cfg.device)
        #     # Adjust key if necessary (e.g., remove "model." prefix if checkpoint is from DataParallel/DDP)
        #     state_dict = checkpoint.get('state_dict', checkpoint) 
        #     model.load_state_dict(state_dict)
        #     print(f"Model state loaded successfully using fallback for {model_checkpoint_path}")
        # except Exception as e_fallback:
        #     print(f"Fallback model loading also failed: {e_fallback}")
        #     return
        return

    model.to(cfg.device) # Ensure model is on the correct device
    model.eval()         # Set to evaluation mode

    # 2. Instantiate DataModule
    data_module = MELDDataModule(cfg)
    # data_module.prepare_data() # Potentially called by PL
    # data_module.setup(stage='test') # Potentially called by PL

    # 3. Instantiate PyTorch Lightning Trainer for evaluation
    trainer_args = {
        'accelerator': cfg.device.type,
        'devices': 1,
        'logger': False # No need to log to default logger during .test() typically
    }
    pl_trainer = pl.Trainer(**trainer_args)

    # 4. Run evaluation
    eval_split = getattr(cfg, 'eval_split', 'test') # What split to evaluate on
    print(f"Evaluating on {eval_split} split...")
    # The datamodule needs to provide the correct dataloader for the specified split
    # when pl_trainer.test is called. MELDDataModule.test_dataloader() should be used.
    # If eval_split is 'dev', then MELDDataModule.val_dataloader()
    
    results = None
    if eval_split == 'test':
        results = pl_trainer.test(model, datamodule=data_module)
    elif eval_split == 'dev':
        results = pl_trainer.validate(model, datamodule=data_module) # Use .validate for dev split
    else:
        print(f"Warning: Evaluation split '{eval_split}' not directly supported by pl.Trainer's test/validate. Defaulting to test.")
        results = pl_trainer.test(model, datamodule=data_module)

    if results:
        print(f"Evaluation metrics for {eval_split} split:")
        for key, value in results[0].items(): # results is a list of dicts
             print(f"  {key}: {value:.4f}")
    else:
        print(f"No results returned from evaluation on {eval_split} split.")

    # --- Old custom evaluation logic ---
    # trainer = trainer_class(cfg=cfg, model_class=model_class) 
    # trainer.load_model(str(model_checkpoint_path))
    # metrics = trainer.evaluate(dataloader=dataloaders[eval_split])
    # print(f"Evaluation metrics for {eval_split} split: {metrics}")
    
def _run_inference(cfg: BaseConfig, args: argparse.Namespace, model_class):
    print(f"Running inference with model: {args.architecture} from checkpoint: {args.inference}")
    model_checkpoint_path = Path(args.inference)
    if not model_checkpoint_path.exists():
        print(f"Error: Model checkpoint {model_checkpoint_path} not found.")
        return

    if not hasattr(cfg, 'text_encoder_model_name') or not cfg.text_encoder_model_name:
        print(f"Critical Error: cfg.text_encoder_model_name is not set. Cannot initialize tokenizer.")
        return

    # Load LightningModule
    try:
        # model_class is MultimodalFusionMLP (LightningModule)
        model_instance = model_class.load_from_checkpoint(
            checkpoint_path=str(model_checkpoint_path),
            cfg=cfg # Pass cfg if model's __init__ needs it and it wasn't saved in hparams
        )
        model_instance.cfg = cfg # Ensure cfg is set
        print(f"Model loaded successfully from {model_checkpoint_path}")
    except Exception as e:
        print(f"Error loading model from checkpoint {model_checkpoint_path}: {e}")
        return
    
    model_instance.to(cfg.device)
    model_instance.eval()

    text_tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder_model_name)
    asr_processor = None
    asr_model_instance = None

    if cfg.input_mode == "audio_only_asr" and cfg.asr_model_name:
        print(f"ASR mode: Loading ASR model {cfg.asr_model_name}")
        try:
            asr_processor = WhisperProcessor.from_pretrained(cfg.asr_model_name)
            asr_model_instance = WhisperForConditionalGeneration.from_pretrained(cfg.asr_model_name).to(cfg.device)
            asr_model_instance.eval()
        except Exception as e:
            print(f"Warning: Could not load ASR components for {cfg.asr_model_name}: {e}")
    
    inference_engine = EmotionInferenceEngine(
        cfg=cfg, model=model_instance, tokenizer=text_tokenizer,
        processor=asr_processor, asr_model_instance=asr_model_instance
    )

    if args.infer_csv_path:
        _perform_csv_inference(cfg, args, inference_engine)
    elif args.infer_audio_path:
        _perform_single_audio_inference(cfg, args, inference_engine)
    else:
        print("No input specified for inference. Use --infer_csv_path or --infer_audio_path.")

def _perform_csv_inference(cfg: BaseConfig, args: argparse.Namespace, engine: EmotionInferenceEngine):
    print(f"Performing batch inference on CSV: {args.infer_csv_path}")
    csv_file_name = Path(args.infer_csv_path).name.lower()
    infer_split_name = "test" # Default
    if "train" in csv_file_name: infer_split_name = "train"
    elif "dev" in csv_file_name or "val" in csv_file_name: infer_split_name = "dev"

    # Ensure cfg.processed_audio_output_dir is a Path object
    # This should be handled by BaseConfig._setup_paths()
    csv_audio_dir = cfg.processed_audio_output_dir / infer_split_name
    if not csv_audio_dir.exists():
        print(f"Warning: Audio directory for CSV inference split '{infer_split_name}' derived as '{csv_audio_dir}' does not exist.")

    all_predictions = engine.infer_from_csv(
        csv_path=args.infer_csv_path,
        audio_dir=str(csv_audio_dir),
        text_column=cfg.csv_text_col_name,
        dialogue_id_col=cfg.csv_dialogue_id_col_name,
        utterance_id_col=cfg.csv_utterance_id_col_name,
        audio_path_col=cfg.csv_audio_path_col_name, # This might be an issue if not in source CSV
        emotion_col=cfg.csv_emotion_col_name,
        num_examples=getattr(cfg, 'infer_num_examples', None),
        use_asr_if_text_missing=(cfg.input_mode == "audio_only_asr")
    )

    if args.infer_output_path:
        try:
            results_df = pd.DataFrame(all_predictions)
            output_file = Path(args.infer_output_path)
            ensure_dir(output_file.parent) # ensure_dir needs to be available
            results_df.to_csv(output_file, index=False)
            print(f"Batch inference results saved to: {output_file}")
        except Exception as e_save:
            print(f"Error saving inference results to {args.infer_output_path}: {e_save}")
            print("Results (first few):", all_predictions[:5])
    else:
        print("\nBatch inference results (up to configured infer_num_examples or 5 default):")
        limit_print = getattr(cfg, 'infer_num_examples', 5) # Use 5 if infer_num_examples is None
        for i, pred in enumerate(all_predictions):
            if i < limit_print:
                print(pred)
            else:
                if i == limit_print: print(f"... (further results truncated, showing first {limit_print})")
                break

def _perform_single_audio_inference(cfg: BaseConfig, args: argparse.Namespace, engine: EmotionInferenceEngine):
    print(f"Performing inference on audio file: {args.infer_audio_path}")
    text_for_audio = None
    if cfg.input_mode == "audio_text":
        # Ensure text_for_audio is a string or None.
        text_for_audio_input = input(f"Model is in 'audio_text' mode. Provide text for {args.infer_audio_path} (or leave blank for ASR if engine supports fallback): ")
        text_for_audio = str(text_for_audio_input) if text_for_audio_input else None

    prediction = engine.infer_from_file(
        audio_file_path=args.infer_audio_path,
        text=text_for_audio,
        use_asr=(cfg.input_mode == "audio_only_asr" or (cfg.input_mode == "audio_text" and not text_for_audio))
    )
    print(f"Prediction for {args.infer_audio_path}: {prediction}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Emotion Classification from Audio + Text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Core arguments ---
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to a YAML configuration file. Overrides defaults. CLI args override YAML.")

    available_architectures = get_available_architectures()
    default_architecture = available_architectures[0] if available_architectures else "mlp_fusion"

    parser.add_argument("--architecture", type=str, default=default_architecture,
                        choices=available_architectures,
                        help=f"Model architecture to use. Default: {default_architecture}")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for the experiment, used for saving models/results. Overrides YAML.")

    # General options (can be overridden by YAML, then by these CLI args)
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset to use (e.g., meld, iemocap). Overrides YAML.")
    parser.add_argument("--input_mode", type=str, default=None,
                        choices=["audio_text", "audio_only_asr", "video_audio_text"],
                        help="Input modality combination. Overrides YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Overrides YAML & config default.")
    parser.add_argument("--device_name", type=str, default=None,
                        help="Device to use (cuda, cpu, mps). Overrides YAML & config default.")

    # --- Mode selection ---
    mode_group = parser.add_argument_group('Operational Modes')
    mode_group.add_argument("--prepare_data", action="store_true", help="Run full data preparation pipeline")
    mode_group.add_argument("--run_eda", action="store_true", help="Run all exploratory data analysis scripts")
    mode_group.add_argument("--train_model", action="store_true", help="Train model")
    mode_group.add_argument("--evaluate_model", type=str, default=None, metavar="MODEL_PATH",
                            help="Path to a trained model checkpoint to evaluate.")
    mode_group.add_argument("--inference", type=str, default=None, metavar="MODEL_PATH",
                            help="Run inference with a trained model at MODEL_PATH.")
    # mode_group.add_argument("--benchmark", action="store_true", help="Benchmark model inference speed (Not Implemented)")


    # --- Key Training/Data Overrides (can also be in YAML) ---
    override_group = parser.add_argument_group('Key Overrides for YAML/Defaults')
    override_group.add_argument("--num_epochs", type=int, default=None, help="Number of epochs. Overrides YAML.")
    override_group.add_argument("--batch_size", type=int, default=None, help="Batch size. Overrides YAML.")
    override_group.add_argument("--learning_rate", type=float, default=None, help="Learning rate. Overrides YAML.")
    override_group.add_argument("--num_dataloader_workers", type=int, default=None, help="Number of dataloader workers. Overrides YAML/BaseConfig default.")

    # --- Data Preparation Control (CLI for debug/quick tests) ---
    dataprep_group = parser.add_argument_group('Data Preparation Debug Overrides')
    dataprep_group.add_argument("--force_wav_conversion", action="store_true",
                                help="Force overwrite of existing WAV files.")
    dataprep_group.add_argument("--force_reprocess_hf_dataset", action="store_true",
                                help="Force reprocessing for Hugging Face dataset creation.")
    dataprep_group.add_argument("--limit_dialogues_train", type=int, default=None, help="Limit dialogues for train split. Overrides YAML.")
    dataprep_group.add_argument("--limit_dialogues_dev", type=int, default=None, help="Limit dialogues for dev split. Overrides YAML.")
    dataprep_group.add_argument("--limit_dialogues_test", type=int, default=None, help="Limit dialogues for test split. Overrides YAML.")

    # --- Inference options (inference mode) ---
    infer_group = parser.add_argument_group('Inference Options')
    infer_group.add_argument("--infer_csv_path", type=str, default=None,
                             help="Path to CSV file for batch inference.")
    infer_group.add_argument("--infer_audio_path", type=str, default=None,
                             help="Path to single audio file for inference.")
    infer_group.add_argument("--infer_output_path", type=str, default=None,
                             help="Path to save inference results (e.g., CSV). Default: print to console.")

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Determine the YAML configuration path to load
    yaml_path_to_load = args.config_file
    if not yaml_path_to_load and args.architecture:
        # Default to architecture-specific YAML if no general config_file is given
        # CONFIG_DIR is now defined
        yaml_path_to_load = os.path.join(CONFIG_DIR, f"{args.architecture}_default.yaml")

    if yaml_path_to_load and not os.path.exists(yaml_path_to_load):
        print(f"Warning: Specified YAML config file not found: {yaml_path_to_load}. Proceeding with defaults and CLI args.")
        yaml_path_to_load = None
    elif yaml_path_to_load:
        print(f"Attempting to load specified YAML config: {yaml_path_to_load}")


    # Dynamically get the config class based on architecture
    # The architecture name for config class lookup can come from CLI or be inferred
    # (e.g. from a 'model_type' field in a generic YAML, or a default)
    # For now, assume args.architecture is the primary source if specific config class is needed.
    # This part might need refinement based on how architecture dictates config class vs. generic YAML.

    # Get the appropriate configuration class
    # This assumes get_config_class can handle arch_for_config_class being None
    # if the YAML is expected to define the architecture.
    # Based on the error, MLPFusionConfig was already selected, so this part of your code works.
    arch_for_config_class = args.architecture # This might need to be smarter if arch isn't passed
                                            # but is defined in the YAML.
                                            # For now, the error shows MLPFusionConfig is already chosen.

    # Get the appropriate configuration class
    # Revert to using the imported get_default_config_class_for_arch
    config_class = get_default_config_class_for_arch(arch_for_config_class)
    if not config_class:
        print(f"Critical: No config class found for architecture '{arch_for_config_class}'. Exiting.")
        return
    
    cfg = config_class.from_args(args,
                                 architecture_name_cli=args.architecture, # Pass CLI architecture
                                 yaml_config_path=yaml_path_to_load)

    # ── Backfill experiment_name if the Config class didn't define it ──
    if not hasattr(cfg, "experiment_name"):
        # Prefer a supplied CLI --experiment_name; otherwise use the architecture name
        setattr(cfg, "experiment_name",
                args.experiment_name if args.experiment_name else args.architecture)

    print(f"Configuration loaded for architecture: {cfg.architecture_name}, "
          f"dataset: {cfg.dataset_name}, input_mode: {cfg.input_mode}")
    print(f"Experiment name: {cfg.experiment_name}")
    print(f"Device set to: {cfg.device}")
    print(f"Model save directory: {cfg.model_save_dir}")
    ensure_dir(cfg.model_save_dir) # Ensure model save directory exists
    ensure_dir(cfg.results_dir)   # Ensure results directory exists

    set_seed(cfg.random_seed)

    model_class, trainer_class = get_model_architecture(args.architecture)
    if not model_class:
        print(f"ERROR: Model class not found for architecture: {args.architecture}. Exiting.")
        return

    # Mode dispatching
    if args.prepare_data:
        _run_data_preparation(cfg, args)
    elif args.run_eda:
        _run_eda(cfg, args)
    elif args.train_model:
        # If trainer_class is None, it implies a direct PyTorch Lightning setup.
        if trainer_class is None:
            print(f"Info: For {args.architecture}, PyTorch Lightning Trainer will be used directly as trainer_class is None.")
            _run_training(cfg, args, model_class, trainer_class) # trainer_class will be None
        elif model_class: # If there's a model_class and a trainer_class, proceed (legacy or specific custom trainer)
             _run_training(cfg, args, model_class, trainer_class)
        else:
            # This case should ideally not be hit if get_model_architecture behaves correctly
            print(f"ERROR: Cannot proceed with training. Model class not found for {args.architecture}, or trainer setup is inconsistent.")
            return
    elif args.evaluate_model:
        # Similar logic for evaluation: if trainer_class is None, assume direct PL evaluation.
        if trainer_class is None:
            print(f"Info: For {args.architecture} evaluation, PyTorch Lightning Trainer will be used directly.")
            _run_evaluation(cfg, args, model_class, trainer_class) # trainer_class will be None
        elif model_class:
            _run_evaluation(cfg, args, model_class, trainer_class)
        else:
            print(f"ERROR: Cannot proceed with evaluation. Model class not found for {args.architecture}, or trainer setup is inconsistent.")
            return
    elif args.inference:
        _run_inference(cfg, args, model_class)
    # elif args.benchmark: # Re-enable if benchmark mode is implemented
    #     print("Benchmarking functionality not yet implemented in main.py")
    else:
        if not any([args.prepare_data, args.run_eda, args.train_model, args.evaluate_model, args.inference]):
             print("No operation selected. Use --help for options. Or specify a mode like --train_model.")


if __name__ == "__main__":
    main() 
