"""
Data loading and processing functionality for MELD dataset.

This module provides the core functionality for loading the MELD dataset,
preprocessing audio files, and creating PyTorch DataLoaders.
"""

import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, WhisperProcessor
import torchaudio
import torchaudio.transforms as T
from datasets import load_from_disk
import pytorch_lightning as pl
from typing import Optional

from utils.utils import ensure_dir # Updated import


class MELDDataModule(pl.LightningDataModule):
    """
    Data module for loading and processing MELD dataset using pre-built Hugging Face datasets.
    Handles loading HF datasets and creating dataloaders. Now a LightningDataModule.
    """
    
    def __init__(self, cfg):
        """
        Initialize the MELD data module.
        
        Args:
            cfg: Configuration object with data paths and parameters
        """
        super().__init__()
        self.cfg = cfg
        self.splits = ['train', 'dev', 'test']
        self.datasets = {}
        self.dataloaders_cache = {}
        
    def _get_hf_dataset_path(self, split):
        """
        Constructs the path to the preprocessed Hugging Face dataset for a given split.
        This path should match where main.py/_run_data_preparation saves the dataset.
        Original save path in main.py: cfg.processed_hf_dataset_dir / split_name
        """
        # cfg.processed_hf_dataset_dir is project_root/dataset_name_data/processed/features/hf_datasets
        dataset_path = self.cfg.processed_hf_dataset_dir / split
        
        # The old logic for cache_params_str is removed to align with main.py's current save behavior.
        # If versioning of processed datasets is needed later, both saving and loading logic would need updates.
        # For now, assume main.py --prepare_data (with relevant --limit args) creates the definitive version.
        # use_asr = self.cfg.input_mode == "audio_only_asr"
        # limit_dialogues = None
        # if split == 'train':
        #     limit_dialogues = getattr(self.cfg, 'limit_dialogues_train', None)
        # elif split == 'dev':
        #     limit_dialogues = getattr(self.cfg, 'limit_dialogues_dev', None)
        # elif split == 'test':
        #     limit_dialogues = getattr(self.cfg, 'limit_dialogues_test', None)
        # cache_params_str = f"asr_{use_asr}_limit_{limit_dialogues if limit_dialogues is not None else 'all'}"
        # hf_dataset_dir_for_split = self.cfg.processed_features_dir / "hf_datasets" / split / "processed_data" / cache_params_str
        
        return dataset_path

    def prepare_data(self):
        """
        Checks if the processed Hugging Face dataset directory structure seems to exist.
        Actual data loading happens in setup().
        This method is called once per process, typically for downloads or initial checks.
        """
        # Base directory where hf_datasets for the current dataset should be
        # e.g. data/processed/meld/features/hf_datasets
        base_hf_dir = self.cfg.processed_features_dir / "hf_datasets" 
        if not base_hf_dir.exists():
            print(f"Warning: Base Hugging Face dataset directory not found: {base_hf_dir}")
            print(f"Please ensure data preparation (e.g., main.py --prepare_data) has been run for dataset '{self.cfg.dataset_name}'.")
            # Not raising an error here, setup() will handle it more specifically.
            return
        
        # Check for one of the splits as a further heuristic
        # This is just a high-level check. Detailed check per split in setup().
        sample_split_path = self._get_hf_dataset_path(self.splits[0])
        if not sample_split_path.exists():
            print(f"Warning: Sample Hugging Face dataset path for '{self.splits[0]}' split not found: {sample_split_path}")
            print(f"  This might indicate that data preparation is incomplete or paths are misconfigured.")
        
        print(f"Data preparation check: Using Hugging Face datasets from {base_hf_dir} (expected structure).")
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup the datasets for each split by loading preprocessed Hugging Face datasets.
        Called on every GPU in DDP.
        
        Args:
            stage (str, optional): Stage ('fit', 'validate', 'test', or None). 
                                   If None, sets up all splits.
        """
        print(f"Setting up MELDDataModule for stage: {stage}, dataset: {self.cfg.dataset_name}")
        self.datasets = {}
        splits_to_load = []

        if stage == 'fit' or stage is None:
            splits_to_load.extend(['train', 'dev'])
        if stage == 'validate': # Explicitly for validation pass if trainer.validate is called
            splits_to_load.append('dev')
        if stage == 'test' or stage is None:
            splits_to_load.append('test')
        if stage == 'predict': # For inference/prediction
            # Determine split based on cfg.eval_split or other inference settings if needed
            # For now, assuming predict might use dev or test, or a specific inference split
            # This part might need more specific logic based on how predict stage is used.
            splits_to_load.append(getattr(self.cfg, 'eval_split', 'dev')) 

        splits_to_load = sorted(list(set(splits_to_load))) # Unique, sorted splits

        for split_name in splits_to_load:
            dataset_path = self._get_hf_dataset_path(split_name)
            if not dataset_path.exists():
                print(f"Warning: Hugging Face dataset for split '{split_name}' not found at {dataset_path}.")
                print(f"Please run data preparation first (e.g., main.py --prepare_data).")
                self.datasets[split_name] = None 
                continue
            
            print(f"DEBUG: MELDDataModule.setup() - About to load raw HF '{split_name}' split from: {dataset_path}")
            try:
                from datasets import load_from_disk as hf_load_from_disk
                raw_hf_dataset = hf_load_from_disk(str(dataset_path))
                print(f"DEBUG: MELDDataModule.setup() - Successfully loaded raw HF '{split_name}' split. Length: {len(raw_hf_dataset) if raw_hf_dataset else 'None'}")
                
                # Wrap the raw Hugging Face dataset with our MELDDataset custom class
                if raw_hf_dataset:
                    self.datasets[split_name] = MELDDataset(raw_hf_dataset, self.cfg, split_name=split_name)
                    print(f"DEBUG: MELDDataModule.setup() - Wrapped HF '{split_name}' split with MELDDataset. Length: {len(self.datasets[split_name])}")
                else:
                    self.datasets[split_name] = None
                    print(f"DEBUG: MELDDataModule.setup() - Raw HF '{split_name}' split was None, so MELDDataset wrapper is None.")

            except Exception as e:
                print(f"DEBUG: MELDDataModule.setup() - ERROR processing '{split_name}' split from {dataset_path}: {e}")
                self.datasets[split_name] = None 
                if stage == 'fit' and split_name in ['train', 'dev']:
                    print(f"CRITICAL: Failed to process crucial '{split_name}' split for fitting stage. Raising error.")
                    raise e

        # Assign to specific attributes for PL compatibility if splits were loaded and wrapped
        # This is somewhat redundant now if self.datasets correctly holds MELDDataset instances
        if 'train' in self.datasets and isinstance(self.datasets['train'], MELDDataset):
            self.train_dataset = self.datasets['train']
        else:
            self.train_dataset = None # Or handle error

        if 'dev' in self.datasets and isinstance(self.datasets['dev'], MELDDataset):
            self.val_dataset = self.datasets['dev'] # PyTorch Lightning expects val_dataset
        else:
            self.val_dataset = None

        if 'test' in self.datasets and isinstance(self.datasets['test'], MELDDataset):
            self.test_dataset = self.datasets['test']
        else:
            self.test_dataset = None

    def _create_dataloader(self, split_name: str) -> DataLoader:
        """Helper to create a DataLoader for a given split."""
        if split_name not in self.datasets:
            print(f"Dataset for {split_name} not found in self.datasets. Attempting to run setup for this split.")
            self.setup(stage='fit' if split_name in ['train', 'dev'] else split_name) # Heuristic for stage
            if split_name not in self.datasets:
                 raise RuntimeError(f"Dataset for {split_name} could not be set up. Cannot create DataLoader.")

        shuffle = (split_name == 'train')
        collate_fn_for_loader = self.datasets[split_name].get_collate_fn()

        return DataLoader(
            self.datasets[split_name],
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_dataloader_workers,
            pin_memory=True,
            collate_fn=collate_fn_for_loader
        )

    def train_dataloader(self) -> DataLoader:
        if 'train' not in self.dataloaders_cache:
            self.dataloaders_cache['train'] = self._create_dataloader('train')
        return self.dataloaders_cache['train']

    def val_dataloader(self) -> DataLoader:
        if 'dev' not in self.dataloaders_cache:
            # MELD uses 'dev' for validation
            self.dataloaders_cache['dev'] = self._create_dataloader('dev') 
        return self.dataloaders_cache['dev']

    def test_dataloader(self) -> DataLoader:
        if 'test' not in self.dataloaders_cache:
            self.dataloaders_cache['test'] = self._create_dataloader('test')
        return self.dataloaders_cache['test']
    
    # Keep get_dataloaders for compatibility if any old code paths use it,
    # but ideally, they should transition to PL Trainer and its dataloader hooks.
    def get_dataloaders(self):
        """
        Create and return DataLoaders for each split.
        Ensures setup is called for all splits if not already done.
        This is more for external use if not using PL Trainer's fit/validate/test.
        """
        if not all(split in self.datasets for split in self.splits):
            self.setup(stage=None) # Ensure all datasets are loaded
        
        # Build dataloaders based on currently setup self.datasets
        # This part might be redundant if PL hooks are always used.
        # Consider if this method is still strictly necessary.
        # For now, it mirrors previous behavior but uses _create_dataloader.
        created_loaders = {}
        for split in self.splits:
            if split in self.datasets:
                # Use cached versions if available from PL calls, or create new
                if split in self.dataloaders_cache:
                    created_loaders[split] = self.dataloaders_cache[split]
                else:
                    created_loaders[split] = self._create_dataloader(split)
                    self.dataloaders_cache[split] = created_loaders[split] # Cache it
            else:
                print(f"Warning: Dataset for {split} split not setup. Cannot create DataLoader via get_dataloaders().")
        return created_loaders
    
    def get_class_distribution(self):
        """
        Get the class distribution for each split from the 'label' column of the HF dataset.
        
        Returns:
            dict: Dictionary with class distributions
        """
        class_dist = {}
        for split in self.splits:
            if split in self.datasets and self.datasets[split].hf_dataset:
                hf_ds = self.datasets[split].hf_dataset
                if 'label' in hf_ds.column_names:
                    labels = hf_ds['label']
                    # Ensure labels are Python ints if they are tensors/numpy arrays from dataset
                    if isinstance(labels, list) and len(labels) > 0 and not isinstance(labels[0], int):
                        try:
                            labels = [int(l.item()) if hasattr(l, 'item') else int(l) for l in labels]
                        except Exception as e:
                            print(f"Warning: Could not convert labels to int for class distribution in {split} split: {e}")
                            continue
                    elif not isinstance(labels, list): # If it's a single tensor/array column
                         try:
                            labels = [int(l.item()) if hasattr(l, 'item') else int(l) for l in labels]
                         except Exception as e:
                            print(f"Warning: Could not convert labels column to list of ints for {split} split: {e}")
                            continue


                    class_counts = np.bincount(labels, minlength=self.cfg.num_classes)
                    class_dist[split] = class_counts
                    print(f"Class distribution for {split} (ID: Count):")
                    for i, count in enumerate(class_counts):
                        print(f"  Class {i}: {count}")
                else:
                    print(f"Warning: 'label' column not found in {split} HF dataset. Cannot compute class distribution.")
            else:
                 print(f"Warning: Dataset for {split} not available for class distribution.")
                
        return class_dist


class MELDDataset(Dataset):
    """
    Custom Dataset for MELD, wrapping a Hugging Face dataset split.
    Handles loading raw audio if configured, otherwise passes through HF features.
    """
    
    def __init__(self, hf_dataset_split, cfg, split_name: str):
        """
        Args:
            hf_dataset_split: The loaded Hugging Face dataset for one split.
            cfg: Configuration object.
            split_name: Name of the split (e.g., 'train', 'dev', 'test').
        """
        self.hf_dataset = hf_dataset_split
        self.cfg = cfg
        self.split_name = split_name
        self.audio_input_type = getattr(cfg, 'audio_input_type', 'hf_features') # Default to hf_features
        self.target_raw_audio_sample_rate_for_model = getattr(cfg, 'target_raw_audio_sample_rate_for_model', None)

        print(f"DEBUG MELDDataset init: audio_input_type = {self.audio_input_type}")
        print(f"DEBUG MELDDataset init: cfg.processed_raw_wav_audio_dir = {getattr(cfg, 'processed_raw_wav_audio_dir', 'NOT SET')}")
        print(f"DEBUG MELDDataset init: cfg.processed_audio_output_dir = {getattr(cfg, 'processed_audio_output_dir', 'NOT SET')}")

        if self.audio_input_type == "raw_wav":
            if cfg.processed_raw_wav_audio_dir is None:
                raise ValueError("MELDDataset: 'audio_input_type' is 'raw_wav' but 'cfg.processed_raw_wav_audio_dir' is None or not set.")
            # Ensure the directory path is resolved correctly. cfg.processed_raw_wav_audio_dir is now a property.
            self.raw_wav_audio_base_dir = cfg.processed_raw_wav_audio_dir
            if not self.raw_wav_audio_base_dir.exists():
                 print(f"Warning: MELDDataset: Raw WAV audio base directory does not exist: {self.raw_wav_audio_base_dir}")
            if not hasattr(cfg, 'audio_path_column_name_in_hf_dataset') or not cfg.audio_path_column_name_in_hf_dataset:
                raise ValueError("MELDDataset: 'audio_input_type' is 'raw_wav' but 'cfg.audio_path_column_name_in_hf_dataset' is not set.")


    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        If audio_input_type is 'raw_wav', it loads the audio waveform.
        Otherwise, it expects pre-extracted features in the Hugging Face dataset.
        """
        item = self.hf_dataset[idx]
        sample = {}

        # Text features (always present from HF dataset)
        # Ensure these keys match what build_hf_dataset.py produces
        if 'input_ids' in item: # For standard text encoders
            sample['text_input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
            sample['text_attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.long)
        elif 'whisper_input_features' in item: # For Whisper encoder as text encoder
             # Assuming whisper_input_features is already a suitable tensor or list that collate can handle
            sample['text_input_ids'] = item['whisper_input_features'] # The key might be different based on Whisper's output for text
            # Whisper might not use a separate attention mask for its encoder input if features are fixed length
            # Or, build_hf_dataset.py needs to create/store one if the model expects it.
            # For simplicity, let's assume the model's text encoder part handles this.
            # sample['text_attention_mask'] = torch.ones(len(item['whisper_input_features']), dtype=torch.long) # Example, adjust as needed

        # Labels
        sample['labels'] = torch.tensor(item['label'], dtype=torch.long)

        # Audio processing
        if self.audio_input_type == "raw_wav":
            relative_audio_path = item.get(self.cfg.audio_path_column_name_in_hf_dataset)
            if not relative_audio_path:
                raise ValueError(f"Audio file path not found in item {idx} under column '{self.cfg.audio_path_column_name_in_hf_dataset}'.")
            
            # self.raw_wav_audio_base_dir is now correctly a Path object from cfg.processed_raw_wav_audio_dir
            full_audio_path = self.raw_wav_audio_base_dir / relative_audio_path
            
            if not full_audio_path.exists():
                # Try with .wav extension if not present in relative_audio_path
                if not relative_audio_path.endswith('.wav'):
                    full_audio_path = self.raw_wav_audio_base_dir / (relative_audio_path + '.wav')
                if not full_audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {full_audio_path} (tried with and without .wav extension) for item {idx}.")

            try:
                waveform, original_sr = torchaudio.load(str(full_audio_path))
                # Ensure waveform is mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Check if resampling is needed for the specific model (e.g., PANNs to 32kHz)
                # The base WAVs are assumed to be cfg.sample_rate (e.g., 16kHz)
                current_waveform = waveform
                current_sr = original_sr

                if self.target_raw_audio_sample_rate_for_model and original_sr != self.target_raw_audio_sample_rate_for_model:
                    # Define cache path for resampled audio
                    # Cache dir: PROJECT_ROOT / dataset_data_root / processed / audio_TARGETSR_cache / split / audio_filename
                    # Example: meld_data/processed/audio_32000_cache/train/dia0_utt0.wav
                    cache_dir_name = f"audio_{self.target_raw_audio_sample_rate_for_model}_cache"
                    # cfg.dataset_data_root should give PROJECT_ROOT/meld_data or similar
                    # cfg.processed_data_dir should give PROJECT_ROOT/meld_data/processed
                    resampled_cache_base_dir = self.cfg.processed_data_dir / cache_dir_name
                    ensure_dir(resampled_cache_base_dir / self.split_name) # Updated ensure_dir usage
                    # Use the name from relative_audio_path, ensure it has .wav
                    audio_filename = Path(relative_audio_path).name
                    if not audio_filename.endswith('.wav'): audio_filename += '.wav'
                    cached_resampled_path = resampled_cache_base_dir / self.split_name / audio_filename

                    if cached_resampled_path.exists():
                        # print(f"DEBUG: Loading cached resampled audio from {cached_resampled_path}")
                        waveform_resampled, sr_resampled = torchaudio.load(str(cached_resampled_path))
                        current_waveform = waveform_resampled
                        current_sr = sr_resampled
                    else:
                        # print(f"DEBUG: Resampling {full_audio_path} from {original_sr}Hz to {self.target_raw_audio_sample_rate_for_model}Hz")
                        resampler = T.Resample(orig_freq=original_sr, new_freq=self.target_raw_audio_sample_rate_for_model)
                        waveform_resampled = resampler(waveform)
                        current_waveform = waveform_resampled
                        current_sr = self.target_raw_audio_sample_rate_for_model
                        # Save to cache
                        try:
                            torchaudio.save(str(cached_resampled_path), current_waveform, current_sr)
                            # print(f"DEBUG: Saved resampled audio to {cached_resampled_path}")
                        except Exception as e_save:
                            print(f"Warning: Could not save resampled audio to cache {cached_resampled_path}: {e_save}")
                
                sample['raw_audio'] = current_waveform.squeeze(0) # Remove channel dim if mono, model expects 1D tensor
                sample['sampling_rate'] = current_sr 
                
            except Exception as e:
                print(f"Error loading/processing audio file {full_audio_path}: {e}")
                sample['raw_audio'] = torch.zeros(1) # Dummy tensor
                sample['sampling_rate'] = self.target_raw_audio_sample_rate_for_model if self.target_raw_audio_sample_rate_for_model else self.cfg.sample_rate

        elif self.audio_input_type == "hf_features":
            # Expect 'input_features' (for mel spectrograms from old STFT) or 'audio_input_values' (for Wav2Vec2/WavLM features)
            # The key should be consistent with what build_hf_dataset.py saves.
            # Let's assume 'audio_input_values' is the new standard for HF encoder features.
            audio_feature_key = 'audio_input_values' if 'audio_input_values' in item else 'input_features'
            
            if audio_feature_key not in item:
                raise KeyError(f"MELDDataset: Audio features ('audio_input_values' or 'input_features') not found in HF dataset item for 'hf_features' mode.")
            
            # Features are usually already processed (e.g. to numpy arrays or lists of floats by HF datasets)
            # Convert to tensor. Squeeze if it has an extra batch dim of 1 from HF dataset processing.
            audio_features = torch.tensor(item[audio_feature_key], dtype=torch.float)
            if audio_features.ndim == 1 and audio_feature_key == 'input_features': # For old STFT features that might be flat
                 # This case might need reshaping if it's a flattened spectrogram.
                 # However, build_hf_dataset.py should save them in a model-ready format (e.g., [num_frames, num_features] or [1, num_samples] for Wav2Vec-like)
                 # For now, assume build_hf_dataset provides features that can be directly batched.
                 # If it's (1, T) from Wav2Vec2FeatureExtractor, squeeze it.
                 if audio_features.shape[0] == 1:
                      sample['audio_input_values'] = audio_features.squeeze(0)
                 else: # Should be (T)
                      sample['audio_input_values'] = audio_features

            elif audio_features.ndim > 0 : # For wav2vec like [1, T] or mel [F, T]
                 sample['audio_input_values'] = audio_features.squeeze(0) if audio_features.shape[0] == 1 and audio_features.ndim == 2 else audio_features
            else: # Should not happen if data is prepared correctly
                 sample['audio_input_values'] = audio_features


            # Optional: audio_attention_mask if models use it and build_hf_dataset provides it
            if 'audio_attention_mask' in item:
                sample['audio_attention_mask'] = torch.tensor(item['audio_attention_mask'], dtype=torch.long)
        
        else:
            raise ValueError(f"Unsupported audio_input_type: {self.audio_input_type}")

        return sample

    def _collate_batch(self, batch):
        collated_batch = {}
        
        # Handle text (common to all modes)
        text_ids_list = [item['text_input_ids'] for item in batch if 'text_input_ids' in item]
        text_mask_list = [item['text_attention_mask'] for item in batch if 'text_attention_mask' in item]
        labels_list = [item['labels'] for item in batch if 'labels' in item]

        if text_ids_list:
            collated_batch['text_input_ids'] = torch.nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=0)
            collated_batch['text_attention_mask'] = torch.nn.utils.rnn.pad_sequence(text_mask_list, batch_first=True, padding_value=0)
        elif any('whisper_input_features' in item for item in batch):
             collated_batch['text_input_ids'] = [item['text_input_ids'] for item in batch if 'text_input_ids' in item]

        if labels_list:
            collated_batch['labels'] = torch.stack(labels_list)

        # Handle audio based on type
        if self.audio_input_type == "raw_wav":
            collated_batch['raw_audio'] = [item['raw_audio'] for item in batch if 'raw_audio' in item]
            sampling_rates = [item['sampling_rate'] for item in batch if 'sampling_rate' in item]
            if sampling_rates:
                collated_batch['sampling_rate'] = sampling_rates[0] if len(set(sampling_rates)) == 1 else self.cfg.sample_rate
                if len(set(sampling_rates)) > 1:
                    if self.target_raw_audio_sample_rate_for_model and sampling_rates[0] != self.target_raw_audio_sample_rate_for_model:
                         print(f"Critical Warning: Batch contains SR {sampling_rates[0]} but model expects {self.target_raw_audio_sample_rate_for_model}. Check resampling logic in MELDDataset.")
                    elif not self.target_raw_audio_sample_rate_for_model:
                         print(f"Warning: Batch contains varying sampling rates: {set(sampling_rates)}. Using target cfg.sample_rate {self.cfg.sample_rate} for AutoFeatureExtractor. Ensure WAVs are at target SR or model handles it.")

        elif self.audio_input_type == "hf_features":
            audio_features_list = [item['audio_input_values'] for item in batch if 'audio_input_values' in item]
            if audio_features_list:
                collated_batch['audio_input_values'] = torch.nn.utils.rnn.pad_sequence(audio_features_list, batch_first=True, padding_value=0.0)
            
            if 'audio_attention_mask' in batch[0] and batch[0]['audio_attention_mask'] is not None:
                audio_mask_list = [item['audio_attention_mask'] for item in batch if 'audio_attention_mask' in item]
                if audio_mask_list:
                     collated_batch['audio_attention_mask'] = torch.nn.utils.rnn.pad_sequence(audio_mask_list, batch_first=True, padding_value=0)
            else:
                if 'audio_input_values' in collated_batch:
                    collated_batch['audio_attention_mask'] = (collated_batch['audio_input_values'] != 0.0).long()
        return collated_batch

    def get_collate_fn(self):
        """Returns the appropriate collate function based on audio_input_type."""
        return self._collate_batch 