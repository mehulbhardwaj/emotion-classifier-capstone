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
from datasets import load_from_disk
import pytorch_lightning as pl
from typing import Optional

from .utils import ensure_dir


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
    PyTorch Dataset for MELD, using preprocessed Hugging Face datasets.
    """
    
    def __init__(self, hf_dataset_split, cfg, split_name):
        """
        Initialize the dataset with a loaded Hugging Face dataset split.
        
        Args:
            hf_dataset_split (datasets.Dataset): Loaded Hugging Face dataset for one split.
            cfg: Configuration object.
            split_name (str): Name of the split (e.g., 'train', 'dev', 'test').
        """
        self.hf_dataset = hf_dataset_split
        self.cfg = cfg
        self.split_name = split_name
        
        # Feature processors (Wav2Vec2Processor, WhisperProcessor) are no longer needed here,
        # as features are pre-extracted by scripts/build_hf_dataset.py.
        
        # Ensure essential columns are present
        required_cols = ['input_features', 'input_ids', 'attention_mask', 'label']
        missing_cols = [col for col in required_cols if col not in self.hf_dataset.column_names]
        if missing_cols:
            raise ValueError(f"MELDDataset (for {split_name}) is missing required columns in the Hugging Face dataset: {missing_cols}. "
                             f"Available columns: {self.hf_dataset.column_names}. Ensure build_hf_dataset.py produces these.")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """
        Get a sample from the Hugging Face dataset.
        Assumes features are already processed and are tensors (or can be converted).
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Sample dictionary with features ready for the model.
        """
        item = self.hf_dataset[idx]
        
        # Data should already be tensors from build_hf_dataset.py.
        # If not, convert them here. build_hf_dataset moves them to cfg.device.
        # For DataLoader, it's often better to keep them on CPU and move batch to GPU in training loop.
        # Let's assume build_hf_dataset stores them as tensors. DataLoader will batch them.
        
        audio_features = torch.tensor(item['input_features'], dtype=torch.float32) if not isinstance(item['input_features'], torch.Tensor) else item['input_features']
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long) if not isinstance(item['input_ids'], torch.Tensor) else item['input_ids']
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long) if not isinstance(item['attention_mask'], torch.Tensor) else item['attention_mask']
        emotion_label = torch.tensor(item['label'], dtype=torch.long) if not isinstance(item['label'], torch.Tensor) else item['label']

        sample = {
            'audio_input_values': audio_features,    # This is typically the mel spectrogram / fbank
            'text_input_ids': input_ids,
            'text_attention_mask': attention_mask,
            'emotion_id': emotion_label
        }
        
        # Optionally include other metadata if needed by the model or for debugging
        if 'dialogue_id' in item: sample['dialogue_id'] = item['dialogue_id']
        if 'utterance_id' in item: sample['utterance_id'] = item['utterance_id']
        # if 'text' in item: sample['utterance_text'] = item['text'] # Original text
        # if 'audio_path' in item: sample['audio_path'] = item['audio_path']
        
        return sample
    
    def get_collate_fn(self):
        """
        Get a collate function for batching, primarily for padding audio features.
        Text features ('input_ids', 'attention_mask') are assumed to be already padded to max_length
        by the `build_hf_dataset.py` script.
        
        Returns:
            callable: Collate function
        """
        def collate_fn(batch):
            """
            Custom collate function to handle variable-length audio features (input_features).
            
            Args:
                batch (list): List of samples from __getitem__.
                
            Returns:
                dict: Batched data, with audio features padded.
            """
            # 'audio_input_values' are the Mel spectrograms (or similar) which have shape [Time, Features]
            # We need to pad the Time dimension.
            
            # Find max audio length (Time dimension) in batch
            # item['audio_input_values'] is expected to be a 2D tensor [Time, NumFeatures]
            max_audio_len = 0
            if batch and 'audio_input_values' in batch[0]:
                 max_audio_len = max(sample['audio_input_values'].shape[0] for sample in batch) # Pad first dim (Time)
            
            audio_features_padded = []
            audio_attention_masks = [] # Create attention masks for padded audio

            for sample in batch:
                audio = sample['audio_input_values'] # Should be [Time, NumFeatures]
                
                # Ensure audio is 2D
                if audio.ndim == 1: # Should not happen if build_hf_dataset is correct
                    print(f"Warning: audio_input_values in collate_fn is 1D for an item. Shape: {audio.shape}. This might be an error upstream.")
                    # Attempt to make it [Time, 1] if it's just a flat array of time steps.
                    # This part is heuristic and depends on what the feature actually is.
                    # For Mel spectrograms, it MUST be 2D.
                    # If it's raw waveform processed by Wav2Vec2, it's 1D [Time].
                    # The name "input_features" from build_hf_dataset suggests Mel-specs.
                    if max_audio_len > 0 and audio.shape[0] == max_audio_len : # if it's already max length, just unsqueeze
                         audio = audio.unsqueeze(1) # [Time, 1] - assuming single feature dim
                    else: # if it's shorter, pad and then unsqueeze.
                        padding_len_audio = max_audio_len - audio.shape[0]
                        audio_padded_item = torch.nn.functional.pad(audio, (0, padding_len_audio), value=0.0) # Pad 1D
                        audio = audio_padded_item.unsqueeze(1) # [Time, 1]
                
                padding_len = max_audio_len - audio.shape[0]
                
                if padding_len > 0:
                    # Pad the Time dimension (dim 0). Value 0.0 for audio features.
                    # Pad tuple is (pad_left, pad_right, pad_top, pad_bottom) for 2D.
                    # For [Time, NumFeatures], we pad dim 0, so (0,0) for NumFeatures dim, and (0, padding_len) for Time dim.
                    # Pad format for 2D tensor T (H, W) with F.pad(T, (padding_left, padding_right, padding_top, padding_bottom))
                    # Here, audio is (Time, N_Mels). We want to pad Time dim.
                    audio_padded_item = torch.nn.functional.pad(audio, (0, 0, 0, padding_len), value=0.0)
                    
                    # Create attention mask for audio: 1 for real frames, 0 for padded frames
                    # Mask should be [Batch, Time] or [Batch, Time_padded]
                    mask = torch.ones(audio.shape[0], dtype=torch.long) # Mask for original length
                    mask_padded = torch.nn.functional.pad(mask, (0, padding_len), value=0)
                else:
                    audio_padded_item = audio
                    mask_padded = torch.ones(audio.shape[0], dtype=torch.long)
                
                audio_features_padded.append(audio_padded_item)
                audio_attention_masks.append(mask_padded)
            
            batched_audio_features = torch.stack(audio_features_padded)
            batched_audio_attention_mask = torch.stack(audio_attention_masks)
            
            # Text inputs are already padded to max_length by build_hf_dataset.py
            text_input_ids = torch.stack([sample['text_input_ids'] for sample in batch])
            text_attention_mask = torch.stack([sample['text_attention_mask'] for sample in batch])
            
            emotion_ids = torch.stack([sample['emotion_id'] for sample in batch]) # Assuming emotion_id is already a tensor
            if emotion_ids.ndim > 1 and emotion_ids.shape[0] == self.cfg.batch_size and emotion_ids.shape[1] == 1:
                 emotion_ids = emotion_ids.squeeze(1) # Ensure [B]
            elif emotion_ids.ndim == 0 and self.cfg.batch_size ==1 : # single item in batch
                 emotion_ids = emotion_ids.unsqueeze(0)


            batched = {
                'audio_input_values': batched_audio_features,     # [B, Time_padded, NumFeatures]
                'audio_attention_mask': batched_audio_attention_mask, # [B, Time_padded]
                'text_input_ids': text_input_ids,                 # [B, MaxTextSeqLen]
                'text_attention_mask': text_attention_mask,       # [B, MaxTextSeqLen]
                'emotion_ids': emotion_ids                        # [B]
            }
            
            # Include other metadata if present and stacked appropriately
            if batch and 'dialogue_id' in batch[0]:
                batched['dialogue_ids'] = [sample['dialogue_id'] for sample in batch]
            if batch and 'utterance_id' in batch[0]:
                batched['utterance_ids'] = [sample['utterance_id'] for sample in batch]
            
            return batched
        
        return collate_fn


def load_meld_csvs(csv_dir):
    """
    Load MELD CSVs from the specified directory.
    
    Args:
        csv_dir (Path): Directory containing MELD CSVs
        
    Returns:
        dict: Dictionary with DataFrames for each split
    """
    splits = ['train', 'dev', 'test']
    dfs = {}
    
    for split in splits:
        csv_path = csv_dir / f"{split}_sent_emo.csv"
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                dfs[split] = df
                print(f"Loaded {split} CSV with {len(df)} rows from {csv_path}")
            except Exception as e:
                print(f"Error loading {split} CSV from {csv_path}: {e}")
                dfs[split] = pd.DataFrame()  # Empty DataFrame as placeholder
        else:
            print(f"Warning: CSV file not found: {csv_path}")
            dfs[split] = pd.DataFrame()  # Empty DataFrame as placeholder
    
    return dfs 