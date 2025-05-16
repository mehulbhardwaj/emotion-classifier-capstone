"""
Helper functions for orchestrating different modes of operation triggered from main.py CLI.
"""
import argparse
from pathlib import Path
import pandas as pd
import torch

# It's better to pass necessary modules/classes if they are not too many,
# or ensure this helper module can import them correctly.
# Assuming this file is in common/ and can import sibling modules:
from .config import BaseConfig
from .data_loader import MELDDataModule
from .inference import EmotionInferenceEngine
from .utils import ensure_dir # For saving CSV results

# Functions from scripts - these need to be callable with cfg
# Ensure these scripts expose their main logic as functions.
from scripts.extract_meld_wavs_from_mp4s import extract_wavs as extract_meld_wavs_main
from scripts.build_hf_dataset import prepare_meld_hf_dataset
from scripts.preliminary_eda import main as preliminary_eda_main
from scripts.processed_features_eda import main as processed_features_eda_main

# For type hinting if model_class/trainer_class are passed
from typing import Type 
# from torch.nn import Module as ModelType # Example for model class type hint
# from pytorch_lightning import LightningModule as TrainerType # Example for trainer class type hint

# For inference components
from transformers import AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration

# --- Helper Functions for Main Modes --- 

def run_data_preparation(cfg: BaseConfig, args: argparse.Namespace):
    print("Starting data preparation pipeline...")
    if cfg.run_mp4_to_wav_conversion:
        print("Step 1: Converting MP4s to WAVs...")
        try:
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
                for split_name, ds in datasets.items():
                    print(f"  {split_name.capitalize()} split: {len(ds)} items.")
        except Exception as e:
            print(f"Error during prepare_meld_hf_dataset: {e}")
    else:
        print("Skipping Hugging Face dataset creation as per configuration.")

def run_eda(cfg: BaseConfig, args: argparse.Namespace):
    print("Running Exploratory Data Analysis...")
    print("Step 1: Preliminary EDA on raw data...")
    try:
        preliminary_eda_main(cfg_param=cfg)
    except Exception as e:
        print(f"Error during preliminary_eda_main: {e}")

    print("\nStep 2: EDA on processed Hugging Face features...")
    try:
        processed_features_eda_main(cfg_param=cfg)
    except Exception as e:
        print(f"Error during processed_features_eda_main: {e}")

def run_training(cfg: BaseConfig, args: argparse.Namespace, model_class: Type, trainer_class: Type):
    print(f"Training model: {args.architecture}")
    data_module = MELDDataModule(cfg)
    data_module.prepare_data()
    _ = data_module.setup() # Ensure datasets are loaded
    dataloaders = data_module.get_dataloaders()
    
    trainer = trainer_class(cfg=cfg, model_class=model_class)
    trainer.train(
        train_dataloader=dataloaders['train'],
        val_dataloader=dataloaders['dev']
    )
    print("Training complete.")
    
    eval_split_after_train = getattr(cfg, 'eval_split', 'test') 
    print(f"Evaluating on {eval_split_after_train} set after training...")
    metrics = trainer.evaluate(
        dataloader=dataloaders[eval_split_after_train],
    )
    print(f"Metrics on {eval_split_after_train} set: {metrics}")

def run_evaluation(cfg: BaseConfig, args: argparse.Namespace, model_class: Type, trainer_class: Type):
    print(f"Evaluating model from: {args.evaluate_model}")
    model_checkpoint_path = Path(args.evaluate_model)
    if not model_checkpoint_path.exists():
        print(f"Error: Model checkpoint {model_checkpoint_path} not found.")
        return
    
    data_module = MELDDataModule(cfg)
    data_module.prepare_data()
    _ = data_module.setup()
    dataloaders = data_module.get_dataloaders()
    
    trainer = trainer_class(cfg=cfg, model_class=model_class) 
    trainer.load_model(str(model_checkpoint_path)) 
    
    eval_split = getattr(cfg, 'eval_split', 'test')
    print(f"Evaluating on {eval_split} split...")
    metrics = trainer.evaluate(
        dataloader=dataloaders[eval_split]
    )
    print(f"Evaluation metrics for {eval_split} split: {metrics}")

def run_inference(cfg: BaseConfig, args: argparse.Namespace, model_class: Type):
    print(f"Running inference with model: {args.inference}")
    model_checkpoint_path = Path(args.inference)
    if not model_checkpoint_path.exists():
        print(f"Error: Model checkpoint {model_checkpoint_path} not found.")
        return

    model_instance = model_class(cfg=cfg)
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=cfg.device)
        if 'state_dict' in checkpoint:
            state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
            model_instance.load_state_dict(state_dict)
            print(f"Model state loaded successfully from PL-style checkpoint: {model_checkpoint_path}")
        elif 'model_state_dict' in checkpoint:
             model_instance.load_state_dict(checkpoint['model_state_dict'])
             print(f"Model state loaded successfully from 'model_state_dict' key: {model_checkpoint_path}")
        else:
            model_instance.load_state_dict(checkpoint)
            print(f"Model state loaded successfully (assumed raw state_dict): {model_checkpoint_path}")
    except Exception as e:
        print(f"Error loading model state_dict from {model_checkpoint_path}: {e}")
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
            print(f"Warning: Could not load ASR components: {e}")
    
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
    infer_split_name = "test"
    if "train" in csv_file_name: infer_split_name = "train"
    elif "dev" in csv_file_name or "val" in csv_file_name: infer_split_name = "dev"
    
    csv_audio_dir = cfg.processed_audio_output_dir / infer_split_name
    if not csv_audio_dir.exists():
            print(f"Warning: Audio directory for CSV inference split '{infer_split_name}' derived as '{csv_audio_dir}' does not exist.")

    all_predictions = engine.infer_from_csv(
        csv_path=args.infer_csv_path,
        audio_dir=str(csv_audio_dir), 
        text_column=cfg.csv_text_col_name, 
        dialogue_id_col=cfg.csv_dialogue_id_col_name, 
        utterance_id_col=cfg.csv_utterance_id_col_name, 
        audio_path_col=cfg.csv_audio_path_col_name, 
        emotion_col=cfg.csv_emotion_col_name, 
        num_examples=getattr(cfg, 'infer_num_examples', None),
        use_asr_if_text_missing=(cfg.input_mode == "audio_only_asr")
    )
    
    if args.infer_output_path:
        try:
            results_df = pd.DataFrame(all_predictions)
            output_file = Path(args.infer_output_path)
            ensure_dir(output_file.parent)
            results_df.to_csv(output_file, index=False)
            print(f"Batch inference results saved to: {output_file}")
        except Exception as e_save:
            print(f"Error saving inference results to {args.infer_output_path}: {e_save}")
            print("Results (first few):", all_predictions[:5])
    else:
        print("\nBatch inference results (up to configured infer_num_examples):")
        limit_print = getattr(cfg, 'infer_num_examples', 5)
        for i, pred in enumerate(all_predictions):
            if i < limit_print:
                    print(pred)
            else:
                if i == limit_print: print(f"... (further results truncated)")
                break

def _perform_single_audio_inference(cfg: BaseConfig, args: argparse.Namespace, engine: EmotionInferenceEngine):
    print(f"Performing inference on audio file: {args.infer_audio_path}")
    text_for_audio = None
    if cfg.input_mode == "audio_text":
        text_for_audio = input(f"Model is in 'audio_text' mode. Provide text for {args.infer_audio_path} (or leave blank for ASR if engine supports fallback): ")
        text_for_audio = text_for_audio if text_for_audio else None
    
    prediction = engine.infer_from_file(
        audio_file_path=args.infer_audio_path, 
        text=text_for_audio,
        use_asr=(cfg.input_mode == "audio_only_asr" or (cfg.input_mode == "audio_text" and not text_for_audio))
    )
    print(f"Prediction for {args.infer_audio_path}: {prediction}") 