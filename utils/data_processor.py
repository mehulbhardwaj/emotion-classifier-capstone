"""Simplified data processing for MELD dataset.

Handles downloading, preprocessing, and loading the MELD dataset.
"""

import os
import torch
import pandas as pd
import numpy as np
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset as HFDataset
import pytorch_lightning as pl
from typing import Optional, Dict, List, Any, Tuple


def ensure_dir(dir_path):
    """Ensure a directory exists, creating it if necessary."""
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


class MELDDataset(Dataset):
    """Dataset for MELD emotion classification."""
    
    def __init__(
        self, 
        hf_dataset_split,
        text_encoder_model_name: str,
        audio_input_type: str = "hf_features",
        text_max_length: int = 128,
        split_name: str = "train"
    ):
        """Initialize the dataset.
        
        Args:
            hf_dataset_split: HuggingFace dataset split
            text_encoder_model_name: Name of text encoder model
            audio_input_type: Type of audio input ("hf_features" or "raw_wav")
            text_max_length: Maximum text length for tokenization
            split_name: Name of the split ("train", "dev", "test")
        """
        self.hf_dataset = hf_dataset_split
        self.text_encoder_model_name = text_encoder_model_name
        self.audio_input_type = audio_input_type
        self.text_max_length = text_max_length
        self.split_name = split_name
        
        # Load tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_name)
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """Get a data sample."""
        item = self.hf_dataset[idx]
        sample = {}
        
        # Process text
        if "text" in item:
            text = item["text"]
            encoded_text = self.tokenizer(
                text,
                max_length=self.text_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            sample["text_input_ids"] = encoded_text["input_ids"].squeeze(0)
            sample["text_attention_mask"] = encoded_text["attention_mask"].squeeze(0)
        
        # Process audio
        if self.audio_input_type == "hf_features" and "audio_features" in item:
            # Use pre-extracted features
            audio_features = torch.tensor(item["audio_features"])
            sample["audio_input_values"] = audio_features
        elif self.audio_input_type == "raw_wav" and "audio_path" in item:
            # Load raw audio (to be processed by the model)
            try:
                waveform, sample_rate = torchaudio.load(item["audio_path"])
                sample["raw_audio"] = waveform
                sample["sampling_rate"] = sample_rate
            except Exception as e:
                print(f"Error loading audio file {item['audio_path']}: {e}")
                # Create empty tensor as fallback
                sample["raw_audio"] = torch.zeros(1, 16000)  # 1 second of silence
                sample["sampling_rate"] = 16000
        
        # Process label
        if "label" in item:
            sample["labels"] = torch.tensor(item["label"], dtype=torch.long)
        
        return sample
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        collated_batch = {}
        
        # Handle text inputs
        if "text_input_ids" in batch[0]:
            text_ids = [item["text_input_ids"] for item in batch]
            text_mask = [item["text_attention_mask"] for item in batch]
            collated_batch["text_input_ids"] = torch.stack(text_ids)
            collated_batch["text_attention_mask"] = torch.stack(text_mask)
        
        # Handle audio inputs
        if self.audio_input_type == "raw_wav" and "raw_audio" in batch[0]:
            collated_batch["raw_audio"] = [item["raw_audio"] for item in batch]
            if all("sampling_rate" in item for item in batch):
                # Use the first sample rate (they should all be the same)
                collated_batch["sampling_rate"] = batch[0]["sampling_rate"]
        elif "audio_input_values" in batch[0]:
            audio_features = [item["audio_input_values"] for item in batch]
            # Pad sequences to the same length
            collated_batch["audio_input_values"] = torch.nn.utils.rnn.pad_sequence(
                audio_features, batch_first=True, padding_value=0.0
            )
            # Create attention mask based on padding
            collated_batch["audio_attention_mask"] = \
                (collated_batch["audio_input_values"] != 0.0).long()
        
        # Handle labels
        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
            collated_batch["labels"] = torch.stack(labels)
        
        return collated_batch


class MELDDataset(Dataset):
    """Dataset for MELD emotion classification."""
    
    def __init__(
        self, 
        hf_dataset_split,
        text_encoder_model_name: str,
        audio_input_type: str = "hf_features",
        text_max_length: int = 128,
        split_name: str = "train"
    ):
        """Initialize the dataset.
        
        Args:
            hf_dataset_split: HuggingFace dataset split
            text_encoder_model_name: Name of text encoder model
            audio_input_type: Type of audio input ("hf_features" or "raw_wav")
            text_max_length: Maximum text length for tokenization
            split_name: Name of the split ("train", "dev", "test")
        """
        self.hf_dataset = hf_dataset_split
        self.text_encoder_model_name = text_encoder_model_name
        self.audio_input_type = audio_input_type
        self.text_max_length = text_max_length
        self.split_name = split_name
        
        # Load tokenizer for text processing
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_model_name)
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        """Get a data sample."""
        item = self.hf_dataset[idx]
        sample = {}
        
        # Process text
        if "text" in item:
            text = item["text"]
            encoded_text = self.tokenizer(
                text,
                max_length=self.text_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            sample["text_input_ids"] = encoded_text["input_ids"].squeeze(0)
            sample["text_attention_mask"] = encoded_text["attention_mask"].squeeze(0)
        
        # Process audio
        if self.audio_input_type == "hf_features" and "audio_features" in item:
            # Use pre-extracted features
            audio_features = torch.tensor(item["audio_features"])
            sample["audio_input_values"] = audio_features
        elif self.audio_input_type == "raw_wav" and "audio_path" in item:
            # Load raw audio (to be processed by the model)
            try:
                waveform, sample_rate = torchaudio.load(item["audio_path"])
                sample["raw_audio"] = waveform
                sample["sampling_rate"] = sample_rate
            except Exception as e:
                print(f"Error loading audio file {item['audio_path']}: {e}")
                # Create empty tensor as fallback
                sample["raw_audio"] = torch.zeros(1, 16000)  # 1 second of silence
                sample["sampling_rate"] = 16000
        
        # Process label
        if "label" in item:
            sample["labels"] = torch.tensor(item["label"], dtype=torch.long)
        
        return sample
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        collated_batch = {}
        
        # Handle text inputs
        if "text_input_ids" in batch[0]:
            text_ids = [item["text_input_ids"] for item in batch]
            text_mask = [item["text_attention_mask"] for item in batch]
            collated_batch["text_input_ids"] = torch.stack(text_ids)
            collated_batch["text_attention_mask"] = torch.stack(text_mask)
        
        # Handle audio inputs
        if self.audio_input_type == "raw_wav" and "raw_audio" in batch[0]:
            collated_batch["raw_audio"] = [item["raw_audio"] for item in batch]
            if all("sampling_rate" in item for item in batch):
                # Use the first sample rate (they should all be the same)
                collated_batch["sampling_rate"] = batch[0]["sampling_rate"]
        elif "audio_input_values" in batch[0]:
            audio_features = [item["audio_input_values"] for item in batch]
            # Pad sequences to the same length
            collated_batch["audio_input_values"] = torch.nn.utils.rnn.pad_sequence(
                audio_features, batch_first=True, padding_value=0.0
            )
            # Create attention mask based on padding
            collated_batch["audio_attention_mask"] = \
                (collated_batch["audio_input_values"] != 0.0).long()
        
        # Handle labels
        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
            collated_batch["labels"] = torch.stack(labels)
        
        return collated_batch


def download_meld_dataset(data_dir: Path):
    """Download the MELD dataset.
    
    Args:
        data_dir: Directory to store the dataset
    """
    import subprocess
    from zipfile import ZipFile
    import shutil
    import urllib.request
    
    # Ensure directory exists
    data_dir = ensure_dir(data_dir)
    
    # URLs for MELD dataset
    meld_urls = {
        "raw": "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
        "features": "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Features.Models.tar.gz"
    }
    
    # Download and extract raw data
    raw_data_dir = data_dir / "raw"
    ensure_dir(raw_data_dir)
    
    for data_type, url in meld_urls.items():
        target_dir = data_dir / data_type
        ensure_dir(target_dir)
        
        # Download file
        tar_file = target_dir / f"MELD.{data_type.capitalize()}.tar.gz"
        if not tar_file.exists():
            print(f"Downloading {data_type} MELD dataset...")
            urllib.request.urlretrieve(url, tar_file)
        
        # Extract file
        if not (target_dir / "MELD").exists():
            print(f"Extracting {data_type} MELD dataset...")
            subprocess.run(["tar", "-xzf", str(tar_file), "-C", str(target_dir)])
    
    print("MELD dataset downloaded and extracted successfully.")
    return data_dir


def convert_mp4_to_wav(config, force=False):
    """Convert MP4 files to WAV format.
    
    Args:
        config: Configuration object
        force: Whether to force conversion even if WAV files exist
    """
    from pathlib import Path
    import subprocess
    import glob
    import os
    
    # Get paths
    mp4_dir = config.raw_data_dir / "MELD.Raw" / "train_splits"
    wav_output_dir = config.processed_audio_dir
    
    # Ensure output directory exists
    ensure_dir(wav_output_dir)
    
    # Find MP4 files
    mp4_files = list(mp4_dir.glob("**/*.mp4"))
    
    print(f"Found {len(mp4_files)} MP4 files to convert")
    
    # Process each MP4 file
    for i, mp4_file in enumerate(mp4_files):
        # Create relative path to maintain directory structure
        rel_path = mp4_file.relative_to(mp4_dir)
        wav_path = wav_output_dir / rel_path.with_suffix(".wav")
        
        # Create parent directories
        ensure_dir(wav_path.parent)
        
        # Skip if WAV file exists and force is False
        if wav_path.exists() and not force:
            continue
        
        # Convert MP4 to WAV using ffmpeg
        print(f"Converting {i+1}/{len(mp4_files)}: {mp4_file.name} -> {wav_path.name}")
        subprocess.run([
            "ffmpeg", 
            "-i", str(mp4_file),
            "-ac", "1",              # Mono audio
            "-ar", "16000",         # 16kHz sample rate
            "-y",                   # Overwrite existing files
            str(wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print("MP4 to WAV conversion completed.")
    


def prepare_hf_dataset(config, force=False):
    """Prepare Hugging Face dataset from MELD CSV files.
    
    Args:
        config: Configuration object
        force: Whether to force reprocessing even if dataset exists
    """
    from datasets import Dataset as HFDataset
    
    # Ensure directories exist
    ensure_dir(config.processed_features_dir)
    ensure_dir(config.hf_dataset_dir)
    
    # Check if dataset already exists
    all_splits_exist = all((config.hf_dataset_dir / split).exists() for split in ["train", "dev", "test"])
    if all_splits_exist and not force:
        print("Hugging Face datasets already exist. Use force=True to reprocess.")
        return
    
    # Process each split
    for split in ["train", "dev", "test"]:
        # Get CSV file path
        csv_file = config.raw_data_dir / "MELD.Raw" / f"{split}_splits" / f"{split}_sent_emo.csv"
        if not csv_file.exists():
            print(f"Warning: CSV file not found: {csv_file}")
            continue
        
        # Load CSV file
        df = pd.read_csv(csv_file)
        
        # Map emotion labels to integers
        emotion_labels = {
            "neutral": 0, "anger": 1, "disgust": 2, "fear": 3,
            "joy": 4, "sadness": 5, "surprise": 6
        }
        df["label"] = df["Emotion"].map(emotion_labels)
        
        # Add audio paths
        df["audio_path"] = df.apply(
            lambda row: str(config.processed_audio_dir / f"{split}_splits" / f"dia{row['Dialogue_ID']}" / f"utt{row['Utterance_ID']}.wav"),
            axis=1
        )
        
        # Filter to include only rows with existing audio files
        df = df[df["audio_path"].apply(lambda path: Path(path).exists())]
        
        # Create HuggingFace dataset
        hf_dataset = HFDataset.from_pandas(df)
        
        # Map function to extract audio features (if needed)
        # You would implement feature extraction here if needed
        
        # Save the dataset
        output_path = config.hf_dataset_dir / split
        hf_dataset.save_to_disk(output_path)
        
        print(f"Processed {split} split with {len(hf_dataset)} samples.")
    
    print("HuggingFace dataset preparation completed.")


def get_class_distribution(hf_dataset_dir):
    """Get class distribution for each split.
    
    Args:
        hf_dataset_dir: Directory containing HuggingFace datasets
    """
    emotion_names = [
        "neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"
    ]
    
    result = {}
    
    for split in ["train", "dev", "test"]:
        try:
            dataset_path = hf_dataset_dir / split
            if not dataset_path.exists():
                continue
            
            dataset = load_from_disk(str(dataset_path))
            
            # Count occurrences of each label
            label_counts = {}
            for i in range(7):  # 7 emotion classes
                count = sum(1 for label in dataset["label"] if label == i)
                label_counts[emotion_names[i]] = count
            
            result[split] = label_counts
            
        except Exception as e:
            print(f"Error getting class distribution for {split}: {e}")
    
    return result
