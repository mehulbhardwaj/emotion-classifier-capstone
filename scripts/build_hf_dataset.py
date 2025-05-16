# meld_audiotext_emotion_classifier/scripts/prepare_dataset.py

import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer
import subprocess
import json
import glob # Added for file checking
from functools import partial

# Import BaseConfig
from configs.base_config import BaseConfig
from common.utils import ensure_dir

# Constants previously imported are now accessed via cfg instance
# RAW_DATA_DIR -> cfg.raw_data_dir
# MELD_AUDIO_WAV_DIR -> cfg.processed_audio_output_dir (for MELD)
# PROCESSED_FEATURES_DIR -> cfg.processed_features_dir
# CACHE_DIR - Not directly used, caching is under PROCESSED_FEATURES_DIR
# SAMPLE_RATE -> cfg.sample_rate
# NUM_MEL_BINS -> cfg.num_mel_bins (default, could be overridden by arch-specific config if needed for feature extraction)
# HOP_LENGTH -> cfg.hop_length (default)
# EMOTION_TO_ID -> cfg.label_encoder
# ASR_MODEL_NAME -> cfg.asr_model_name
# TEXT_MODEL_NAME -> cfg.text_encoder_model_name
# DEVICE -> cfg.device


def create_dataset_from_csv(split, config: BaseConfig, limit_dialogues=None):
    """
    Create a Dataset object from MELD CSV and (pre-existing) audio files.
    Checks for file existence of WAV files in cfg.processed_audio_output_dir.
    
    Args:
        split (str): Dataset split ('train', 'dev', or 'test').
        config (BaseConfig): Configuration object.
        limit_dialogues (int, optional): If set, limits processing to this many dialogues.

    Returns:
        Dataset object
    """
    # Use cfg.raw_data_dir
    csv_path = config.raw_data_dir / f"{split}_sent_emo.csv"
    if not csv_path.exists():
        print(f"ERROR: CSV file not found for {split} split: {csv_path}")
        return Dataset.from_list([]) 
    df = pd.read_csv(csv_path)

    if limit_dialogues is not None:
        unique_dialogues = df['Dialogue_ID'].unique()[:limit_dialogues]
        df = df[df['Dialogue_ID'].isin(unique_dialogues)]

    data = []
    missing_audio_count = 0
    total_rows_for_split = len(df)

    for idx, row in tqdm(df.iterrows(), total=total_rows_for_split, desc=f"Creating dataset for {split}"):
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID']
        # Path where WAV file *should* be, using cfg.processed_audio_output_dir
        audio_path = config.processed_audio_output_dir / split / f"dia{dia_id}_utt{utt_id}.wav"
        
        # Check if audio file exists
        audio_file_exists = audio_path.exists()

        if not audio_file_exists:
            missing_audio_count += 1
            if missing_audio_count <= 5: # Print for first few misses
                 print(f"Warning: Audio file not found: {audio_path}, skipping item for dataset.")
            elif missing_audio_count == 6:
                 print(f"Further warnings for missing audio files in {split} dataset creation will be suppressed.")
            continue
            
        item = {
            'dialogue_id': dia_id,
            'utterance_id': utt_id,
            'speaker': row['Speaker'],
            'text': row['Utterance'],
            'emotion': row['Emotion'],
            'sentiment': row['Sentiment'],
            'audio_path': str(audio_path), # Store as string for dataset
            # Use cfg.label_encoder
            'label': config.label_encoder.get(row['Emotion'], -1) # Default to -1 if emotion not in map
        }
        if item['label'] == -1 and config.label_encoder: # only warn if map is populated
            print(f"Warning: Emotion '{row['Emotion']}' for dia{dia_id}_utt{utt_id} not in label_encoder map. Assigning label -1.")
        data.append(item)
    
    if missing_audio_count > 0:
        print(f"Warning: {missing_audio_count}/{total_rows_for_split} audio files were not found (or accessible) for the {split} split during dataset creation.")
    if not data:
        print(f"Warning: No data collected for {split} split. Returning empty dataset. Check audio extraction and CSV paths.")
        return Dataset.from_list([])
    return Dataset.from_list(data)


def extract_mel_spectrogram(audio_path_or_waveform, config: BaseConfig, input_sample_rate=None):
    """
    Extract log-mel spectrogram from audio file or waveform.
    Uses cfg.sample_rate, cfg.hop_length, cfg.num_mel_bins.
    
    Args:
        audio_path_or_waveform (str or Path or torch.Tensor): Path to audio file or loaded waveform tensor.
        config (BaseConfig): Configuration object.
        input_sample_rate (int, optional): Sample rate of the input waveform (if waveform is passed).
        
    Returns:
        torch.Tensor or None: Log-mel spectrogram tensor [time, n_mels] or None if error.
    """
    try:
        if isinstance(audio_path_or_waveform, (str, Path)):
            waveform, sample_rate = torchaudio.load(audio_path_or_waveform)
        elif torch.is_tensor(audio_path_or_waveform):
            waveform = audio_path_or_waveform
            sample_rate = input_sample_rate
            if sample_rate is None:
                sample_rate = config.sample_rate 
        else:
            raise TypeError("audio_path_or_waveform must be a path or a tensor.")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        if sample_rate != config.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.sample_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1: # If stereo, convert to mono by averaging channels
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=getattr(config, 'n_fft', 400), # Use n_fft from cfg if available, else default
            hop_length=config.hop_length, 
            n_mels=config.n_mels, # num_mel_bins from BaseConfig or arch-specific -> CHANGED to n_mels
            power=2.0 
        )
        
        mel_spec = mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-9) 
        
        return log_mel_spec.squeeze(0).T 
    except Exception as e:
        print(f"Error extracting mel spectrogram for {audio_path_or_waveform}: {e}")
        return None


def process_dataset_item(item, text_tokenizer, config: BaseConfig, whisper_processor=None, whisper_model=None, use_asr=False):
    """
    Processes a single item from the dataset (typically a dictionary).
    Extracts features (mel spectrogram, text tokens). Optionally performs ASR.
    Device for ASR model and tensors obtained from cfg.device.
    """
    # Use device from cfg (passed as 'device' argument to this function)
    current_device = config.device

    try:
        audio_path = item['audio_path']
        text = item['text']

        # Audio processing
        mel_spectrogram = extract_mel_spectrogram(audio_path, config=config) # extract_mel_spectrogram uses cfg internally
        if mel_spectrogram is None: 
            return None 

        item['input_features'] = mel_spectrogram.to(current_device) # Move to device

        # Text processing
        if use_asr and whisper_processor and whisper_model:
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != whisper_processor.feature_extractor.sampling_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=whisper_processor.feature_extractor.sampling_rate)
                    waveform = resampler(waveform)
                
                if waveform.ndim > 1 and waveform.shape[0] == 1: 
                    waveform = waveform.squeeze(0)
                elif waveform.ndim > 1 and waveform.shape[0] > 1: 
                    waveform = torch.mean(waveform, dim=0)

                inputs = whisper_processor(waveform, sampling_rate=whisper_processor.feature_extractor.sampling_rate, return_tensors="pt")
                # Move ASR input features to device
                input_features_asr = inputs.input_features.to(current_device) 
                
                with torch.no_grad():
                    # whisper_model should already be on current_device
                    predicted_ids = whisper_model.generate(input_features_asr) 
                
                transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                item['asr_text'] = transcription
                text_to_tokenize = transcription 
            except Exception as e_asr:
                print(f"Error during ASR for {audio_path}: {e_asr}. Falling back to original text.")
                item['asr_text'] = "" 
                text_to_tokenize = text 
        else:
            text_to_tokenize = text
            item['asr_text'] = ""

        tokenized_text = text_tokenizer(
            text_to_tokenize, 
            padding='max_length', 
            truncation=True, 
            max_length=config.max_seq_length_text, # Use max_seq_length_text from cfg
            return_tensors="pt"
        )
        # Move tokenized text to device
        item['input_ids'] = tokenized_text['input_ids'].squeeze(0).to(current_device) 
        item['attention_mask'] = tokenized_text['attention_mask'].squeeze(0).to(current_device)
        
        return item
    except Exception as e_item:
        print(f"Error processing item {item.get('audio_path', 'UNKNOWN')}: {e_item}")
        return None


def build_and_process_split_dataset(split, text_tokenizer, current_cfg: BaseConfig, num_proc=None, 
                                  use_asr=False, whisper_processor=None, whisper_model=None, 
                                  force_reprocess_item=False, limit_dialogues=None):
    """
    Builds the initial dataset from CSV for a split and then processes it.
    Uses current_cfg for paths and parameters.
    """
    print(f"\nBuilding initial dataset for {split} split from CSV and existing WAV files...")
    initial_dataset = create_dataset_from_csv(split, config=current_cfg, limit_dialogues=limit_dialogues) # create_dataset_from_csv now uses its 'config' parameter

    if not initial_dataset or len(initial_dataset) == 0:
        print(f"No data loaded for {split} split. Skipping further processing.")
        return None 

    num_items_to_process = len(initial_dataset)
    print(f"Processing {num_items_to_process} items for {split} split (feature extraction, tokenization)...")

    num_proc_actual = num_proc if num_proc is not None else os.cpu_count() // 2
    if num_proc_actual < 1: num_proc_actual = 1

    _process_func = partial(
        process_dataset_item,
        text_tokenizer=text_tokenizer,
        config=current_cfg, # Pass current_cfg here
        whisper_processor=whisper_processor,
        whisper_model=whisper_model, 
        use_asr=use_asr
    )

    print(f"Reprocessing {split} dataset items...")
    if num_items_to_process < 1000 or num_proc_actual == 1: # Use single process for small datasets or if num_proc is 1
        print(f"Using single process for mapping {split} data ({num_items_to_process} items).")
        processed_items = [_process_func(item) for item in tqdm(initial_dataset, desc=f"Processing {split} items") if item is not None]
        processed_dataset = Dataset.from_list([item for item in processed_items if item is not None])
    else:
        print(f"Using {num_proc_actual} processes for mapping.")
        # Note: Hugging Face map can have issues with lambda/partials if they are too complex or capture non-picklable objects.
        # Ensure all components (tokenizer, models for ASR if used) are picklable or handled correctly for multiprocessing.
        # Whisper models might not be easily picklable. ASR within .map() with multiprocessing can be tricky.
        # If ASR is used, consider single-process mapping or careful setup for multiprocessing.
        if use_asr and whisper_model:
            print("Warning: Using ASR with multiprocessing. Ensure Whisper model is compatible or consider single process if errors occur.")
        
        processed_dataset = initial_dataset.map(
            _process_func, 
            num_proc=num_proc_actual,
            # batched=False, # process_dataset_item handles one item at a time
            remove_columns=list(initial_dataset.column_names) # Remove old columns after processing. Careful if _process_func returns None.
        )
        # Filter out None items that resulted from processing errors
        processed_dataset = processed_dataset.filter(lambda example: example is not None)

    if not processed_dataset or len(processed_dataset) == 0:
        print(f"No items successfully processed for {split} dataset. Something went wrong.")
        return None

    return processed_dataset


def prepare_meld_hf_dataset(
    current_cfg: BaseConfig, # Changed to require current_cfg
    force_reprocess_items=False, 
    use_asr_for_processing=False, # This will be controlled by current_cfg.input_mode ideally
    splits_to_process=None, # Default to train, dev, test
    num_workers_dataset_map=None,
    limit_dialogues_train=None, 
    limit_dialogues_dev=None,   
    limit_dialogues_test=None   
    ):
    """
    Prepares the MELD Hugging Face dataset.
    Uses current_cfg for all configuration.
    """
    print("\nStarting MELD Hugging Face dataset preparation...")

    if splits_to_process is None:
        splits_to_process = ['train', 'dev', 'test']
    if num_workers_dataset_map is None:
        num_workers_dataset_map = current_cfg.num_dataloader_workers

    # Determine if ASR should be used based on config's input_mode
    # This function's use_asr_for_processing param can override for testing/explicitness if needed.
    # For now, let's respect the function's parameter if set, else derive from config.
    # A more robust way would be for main.py to set this based on config and pass it.
    actual_use_asr = use_asr_for_processing
    if current_cfg.input_mode == "audio_only_asr":
        print(f"Config input_mode is '{current_cfg.input_mode}', enabling ASR for text processing.")
        actual_use_asr = True
    
    # Use current_cfg.text_encoder_model_name
    text_tokenizer = AutoTokenizer.from_pretrained(current_cfg.text_encoder_model_name, use_fast=True)
    print(f"Text tokenizer loaded from: {current_cfg.text_encoder_model_name}")

    whisper_processor = None
    whisper_model = None 
    if actual_use_asr:
        # Use current_cfg.asr_model_name and current_cfg.device
        print(f"ASR processing enabled. Loading Whisper processor and model from: {current_cfg.asr_model_name}")
        try:
            whisper_processor = WhisperProcessor.from_pretrained(current_cfg.asr_model_name)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(current_cfg.asr_model_name).to(current_cfg.device)
            whisper_model.eval() 
            print("Whisper processor and ASR model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Could not load Whisper processor/model for ASR from {current_cfg.asr_model_name}: {e}")
            print("ASR processing will be effectively disabled.")
            actual_use_asr = False # Disable if models failed to load
            whisper_processor = None
            whisper_model = None
    
    limit_dialogues_map = {
        'train': limit_dialogues_train,
        'dev': limit_dialogues_dev,
        'test': limit_dialogues_test
    }

    all_datasets = {}
    for split in splits_to_process:
        print(f"\n--- Processing {split} split ---")
        current_limit = limit_dialogues_map.get(split)
        
        dataset = build_and_process_split_dataset(
            split=split,
            text_tokenizer=text_tokenizer,
            current_cfg=current_cfg, # Pass the main config object
            num_proc=num_workers_dataset_map,
            use_asr=actual_use_asr, 
            whisper_processor=whisper_processor,
            whisper_model=whisper_model, 
            force_reprocess_item=force_reprocess_items, # This flag is for item-level, overall reprocess controlled above
            limit_dialogues=current_limit
        )
        if dataset:
            all_datasets[split] = dataset
            print(f"Successfully prepared dataset for {split} split with {len(dataset)} items.")
        else:
            print(f"Failed to prepare dataset for {split} split.")

    print("\nFinished MELD Hugging Face dataset preparation.")
    return all_datasets


if __name__ == '__main__':
    print("Running script: build_hf_dataset.py (Example Usage)")

    # Create a config for the example run. 
    # In a real scenario, this would come from main.py or command-line args.
    example_cfg = BaseConfig(dataset_name="meld", input_mode="audio_text") # Default input_mode

    print(f"Using device: {example_cfg.device}")
    print(f"Target sample rate: {example_cfg.sample_rate}")
    print(f"Text tokenizer model: {example_cfg.text_encoder_model_name}")
    print(f"ASR model (if used by input_mode): {example_cfg.asr_model_name}")
    print(f"Processed features will be saved relative to: {example_cfg.processed_features_dir}")
    print(f"Raw data expected at: {example_cfg.raw_data_dir}")
    print(f"Processed audio (WAVs) expected at: {example_cfg.processed_audio_output_dir}")


    # Example: Ensure MELD raw data dir and processed audio dir exist for the test to run
    # This is just for the __main__ example.
    ensure_dir(example_cfg.raw_data_dir)
    ensure_dir(example_cfg.processed_audio_output_dir / 'dev') # for dev split test

    # Create dummy CSV and WAV for testing if they don't exist (very basic)
    dev_csv_path = example_cfg.raw_data_dir / "dev_sent_emo.csv"
    if not dev_csv_path.exists():
        print(f"Creating dummy {dev_csv_path} for example run...")
        dummy_dev_data = {'Dialogue_ID': [0,0,1], 'Utterance_ID': [0,1,0], 'Speaker': ['A','B','A'], 
                          'Utterance': ['Hi there', 'Hello back', 'Testing'], 
                          'Emotion': ['neutral', 'joy', 'neutral'], 'Sentiment': ['neutral','positive','neutral']}
        pd.DataFrame(dummy_dev_data).to_csv(dev_csv_path, index=False)

    dummy_wav_path = example_cfg.processed_audio_output_dir / 'dev' / "dia0_utt0.wav"
    if not dummy_wav_path.exists():
        print(f"Creating dummy WAV file {dummy_wav_path} for example run...")
        sample_rate = example_cfg.sample_rate
        duration = 1  # 1 second
        freq = 440  # A4 note
        t = torch.linspace(0, duration, int(sample_rate * duration), dtype=torch.float32)
        waveform = 0.5 * torch.sin(2 * np.pi * freq * t)
        torchaudio.save(dummy_wav_path, waveform.unsqueeze(0), sample_rate)


    datasets = prepare_meld_hf_dataset(
        current_cfg=example_cfg, # Pass the example config
        force_reprocess_items=True,  # Force for this example to ensure it runs
        use_asr_for_processing=False, # Explicitly false for this example_cfg with audio_text mode
        splits_to_process=['dev'],    
        num_workers_dataset_map=1, # Use 1 for simplicity in example
        limit_dialogues_dev=2       
    )

    if datasets and 'dev' in datasets:
        print(f"\nDev dataset sample (first 2 items if available):")
        dev_dataset_list = list(datasets['dev']) # Convert to list to iterate if it's a streamable dataset
        for i in range(min(2, len(dev_dataset_list))):
            print(dev_dataset_list[i])
        
        print(f"\nDev dataset columns: {datasets['dev'].column_names}")
        
        if len(dev_dataset_list) > 0:
            sample_item = dev_dataset_list[0]
            # Check for 'input_features' (new name for mel_spectrogram tensor in dataset)
            if 'input_features' in sample_item and sample_item['input_features'] is not None:
                print(f"Sample input_features shape/type: {type(sample_item['input_features'])}, tensor shape: {sample_item['input_features'].shape}")
            else:
                print("Sample input_features not found or is None.")
            
            if 'input_ids' in sample_item and sample_item['input_ids'] is not None:
                print(f"Sample input_ids len: {len(sample_item['input_ids'])}, type: {type(sample_item['input_ids'])}\")")
                print(f"Sample input_ids (first 10): {sample_item['input_ids'][:10]}")
            else:
                print("Sample input_ids not found or is None.")
    else:
        print("No datasets were generated or dev split not processed for the example.")

    print("\nExample run finished.") 