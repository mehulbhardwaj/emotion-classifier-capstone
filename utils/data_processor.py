"""utils/data_processor.py – self‑contained replacement.

This file defines:
1. **MELDDataset** – unchanged tokenisation & feature handling.
2. **MELDDataModule** – Lightning DataModule that *always* outputs
   `(wav, wav_mask, txt, txt_mask, labels)` tuples so model code can
   unpack consistently for train/val/test.

Assumed external deps are already installed (datasets, torchaudio,
transformers).
"""

from pathlib import Path
from typing import Dict, Tuple, List, Any

import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import datasets as hf

# -----------------------------------------------------------------------------
# 1.  Low‑level MELDDataset (same as original implementation)
# -----------------------------------------------------------------------------

class MELDDataset(Dataset):
    """HuggingFace split → tokenised + (optionally) raw‑audio tensors."""

    def __init__(
        self,
        hf_split: hf.arrow_dataset.Dataset,
        text_encoder_name: str,
        audio_input_type: str = "hf_features",  # "hf_features" | "raw_wav"
        text_max_len: int = 128,
    ) -> None:
        self.ds = hf_split
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.audio_input_type = audio_input_type
        self.text_max_len = text_max_len

    # --------------------------- basic dataset api ------------------------- #
    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        sample: Dict[str, Any] = {}

        # ---- text ----------------------------------------------------------
        tokens = self.tokenizer(
            row["text"],
            max_length=self.text_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        sample["text_input_ids"] = tokens["input_ids"].squeeze(0)
        sample["text_attention_mask"] = tokens["attention_mask"].squeeze(0)

        # ---- audio ---------------------------------------------------------
        if self.audio_input_type == "hf_features":
            sample["audio_input_values"] = torch.tensor(row["audio_features"])
        else:  # raw_wav
            try:
                wav, sr = torchaudio.load(row["audio_path"])
            except Exception:
                wav, sr = torch.zeros(1, 16000), 16000
            sample["raw_audio"] = wav
            sample["sampling_rate"] = sr

        # ---- label ---------------------------------------------------------
        sample["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return sample

    # ---------------------------- collate_fn ------------------------------ #
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        coll: Dict[str, Any] = {}
        # text
        coll["text_input_ids"]     = torch.stack([b["text_input_ids"] for b in batch])
        coll["text_attention_mask"] = torch.stack([b["text_attention_mask"] for b in batch])
        # audio
        if self.audio_input_type == "hf_features":
            feats = [b["audio_input_values"] for b in batch]
            coll["audio_input_values"] = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
            coll["audio_attention_mask"] = (coll["audio_input_values"] != 0).long()
        else:
            coll["raw_audio"]     = [b["raw_audio"] for b in batch]
            coll["sampling_rate"] = batch[0]["sampling_rate"]
        # label
        coll["labels"] = torch.stack([b["labels"] for b in batch])
        return coll

# -----------------------------------------------------------------------------
# 2.  Lightning DataModule with unified 5‑tuple output
# -----------------------------------------------------------------------------

class MELDDataModule(pl.LightningDataModule):
    """DataModule that normalises every DataLoader batch to 5 tensors.

    `(wav, wav_mask, txt, txt_mask, labels)`
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.datasets: Dict[str, MELDDataset] = {}
        self._current_split: str | None = None  # used inside collate_fn

    # -------------------------- Lightning hooks --------------------------- #
    def setup(self, stage: str | None = None):
        if self.datasets:
            return  # already initialised

        root = Path(self.cfg.data_root) / "processed" / "features" / "hf_datasets"
        text_name  = self.cfg.text_encoder_model_name
        audio_type = self.cfg.audio_input_type

        for split in ["train", "dev", "test"]:
            hf_ds = hf.load_from_disk(root / split)
            self.datasets[split] = MELDDataset(hf_ds, text_name, audio_type)

    # --------------------------- unified collate -------------------------- #
        def _unified_collate_fn(self, batch):
        """Always emit (wav, wav_mask, txt, txt_mask, labels)."""
        raw = self.datasets[self._current_split].collate_fn(batch)

        # Case A: already a 5‑tuple
        if isinstance(raw, (list, tuple)) and len(raw) == 5:
            return raw

        # Case B: dict coming from MELDDataset.collate_fn
        txt       = raw["text_input_ids"]
        txt_mask  = raw["text_attention_mask"]
        labels    = raw["labels"]

        # ――― audio handling ―――
        if "audio_input_values" in raw:                          # hf_features path
            wav      = raw["audio_input_values"]
            wav_mask = raw.get("audio_attention_mask")
            if wav_mask is None:
                wav_mask = (wav.abs().sum(dim=-1) != 0).long()    # fallback mask
        else:                                                    # raw_wav path
            raw_audio = raw["raw_audio"]                        # list of (C,T) tensors
            # pad to (B, T)
            wav = torch.nn.utils.rnn.pad_sequence(
                [a.squeeze(0) for a in raw_audio], batch_first=True, padding_value=0.0
            )
            wav_mask = (wav.abs().sum(dim=-1) != 0).long()

        return wav, wav_mask, txt, txt_mask, labels

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        self._current_split = split
        return DataLoader(
            self.datasets[split],
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.cfg, "dataloader_num_workers", 4),
            pin_memory=True,
            collate_fn=self._unified_collate_fn,
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("dev", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Functions to download the MELD dataset from HF
# ──────────────────────────────────────────────────────────────────────────────

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
