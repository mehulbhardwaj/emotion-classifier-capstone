"""utils/data_processor.py – raw‑audio‑first version

* Works when `audio_input_type` is set to **"raw_wav"** (no pre‑computed
  features).  If `audio_features` happen to exist, it still handles them.
* Always outputs `(wav, wav_mask, txt, txt_mask, labels)`.
* Stereo clips are down‑mixed to mono via `.mean(0)` instead of `squeeze(0)`.
* PosixPath → str cast for `load_from_disk`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import torch
import torchaudio
import datasets as hf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# 1.  MELDDataset (lean but robust)
# ──────────────────────────────────────────────────────────────────────────────

class MELDDataset(Dataset):
    """Return dict with text, raw audio or pre‑extracted features, and label."""

    def __init__(
        self,
        hf_split: hf.arrow_dataset.Dataset,
        text_encoder_name: str,
        audio_input_type: str = "raw_wav",  # default to raw_wav
        text_max_len: int = 128,
    ) -> None:
        self.ds = hf_split
        self.tok = AutoTokenizer.from_pretrained(text_encoder_name)
        self.audio_input_type = audio_input_type
        self.text_max_len = text_max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        out: Dict[str, Any] = {}

        # --- text ----------------------------------------------------------
        toks = self.tok(
            row["text"],
            max_length=self.text_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        out["text_input_ids"] = toks["input_ids"].squeeze(0)
        out["text_attention_mask"] = toks["attention_mask"].squeeze(0)

        # --- audio ---------------------------------------------------------
        if self.audio_input_type == "hf_features" and "audio_features" in row:
            out["audio_input_values"] = torch.tensor(row["audio_features"])
        else:  # fallback to raw wav
            wav, sr = self._safe_load_wav(row["audio_path"], fallback_sr=16000)
            # normalise to ±1
            wav = wav / wav.abs().max().clamp(min=1e-5)
            out["raw_audio"] = wav
            out["sampling_rate"] = sr

        # --- label ---------------------------------------------------------
        out["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return out

    @staticmethod
    def _safe_load_wav(path: str, fallback_sr: int):
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            wav, sr = torch.zeros(1, fallback_sr), fallback_sr
        # down‑mix stereo → mono
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr

    # ---------------------------- collate_fn ------------------------------
    def collate_fn(self, batch):
        coll: Dict[str, Any] = {}
        coll["text_input_ids"]     = torch.stack([b["text_input_ids"] for b in batch])
        coll["text_attention_mask"] = torch.stack([b["text_attention_mask"] for b in batch])

        if "audio_input_values" in batch[0]:  # pre‑extracted features
            feats = [b["audio_input_values"] for b in batch]
            coll["audio_input_values"] = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
            coll["audio_attention_mask"] = (coll["audio_input_values"].abs().sum(-1) != 0).long()
        else:  # raw wav
            wavs = [b["raw_audio"].squeeze(0) for b in batch]
            coll["raw_audio"] = wavs
            coll["sampling_rate"] = batch[0]["sampling_rate"]
        coll["labels"] = torch.stack([b["labels"] for b in batch])
        return coll

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Lightning DataModule (unified 5‑tuple output)
# ──────────────────────────────────────────────────────────────────────────────

class MELDDataModule(pl.LightningDataModule):
    """Outputs (wav, wav_mask, txt, txt_mask, labels) for all splits."""

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.datasets: Dict[str, MELDDataset] = {}
        self._current_split: str | None = None

    def setup(self, stage: str | None = None):
        if self.datasets:
            return
        root = Path(self.cfg.data_root) / "processed" / "features" / "hf_datasets"
        for split in ("train", "dev", "test"):
            hf_ds = hf.load_from_disk(str(root / split))
            self.datasets[split] = MELDDataset(
                hf_ds,
                self.cfg.text_encoder_model_name,
                self.cfg.audio_input_type,
            )

    # ---------------- unified collate → 5‑tuple ---------------------------
    def _to_5tuple(self, raw):
        txt, txt_mask, labels = raw["text_input_ids"], raw["text_attention_mask"], raw["labels"]
        if "audio_input_values" in raw:
            wav = raw["audio_input_values"]
            wav_mask = raw["audio_attention_mask"]
        else:
            wavs = [a.squeeze(0) for a in raw["raw_audio"]]
            wav = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True, padding_value=0.0)
            #wav_mask = (wav.abs().sum(-1) != 0).long()
            wav_mask = (wav != 0.0).long()
        return wav, wav_mask, txt, txt_mask, labels

    def _loader(self, split: str, shuffle: bool):
        ds = self.datasets[split]
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=getattr(self.cfg, "dataloader_num_workers", 4),
            pin_memory=True,
            collate_fn=lambda batch: self._to_5tuple(ds.collate_fn(batch)),
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("dev", False)

    def test_dataloader(self):
        return self._loader("test", False)


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
