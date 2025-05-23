
"""utils/data_processor.py – dialogue‑level batching
====================================================
Drop‑in replacement that
* supports both utterance‑level (baseline) **and** dialogue‑level (Dialog‑RNN,
  TOD‑KAT‑lite) batches, selected via `cfg.architecture`.
* groups rows by `Dialogue_ID`, padds to the longest dialogue in the batch and
  returns `(B, T,·)` tensors + `speaker_id`, `dialog_mask`.
* keeps helper functions (`download_meld_dataset`, …) but wraps them in
  `if __name__ == "__main__":` so importing during training never triggers
  missing‑import errors.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torchaudio
import datasets as hf
import pandas as pd
import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer
from transformers import AutoTokenizer
from utils.sampler import DialogueBatchSampler
import torch.nn.functional as F


################################################################################
# small util (needed by helper funcs at end) -----------------------------------
################################################################################

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

################################################################################
# 1. MELDDataset – one row = one utterance
################################################################################

class MELDDataset(Dataset):
    """Returns a dict with text/audio/labels + metadata for a single utterance."""

    def __init__(
        self,
        hf_split: hf.arrow_dataset.Dataset,
        text_encoder_name: str,
        *,
        audio_input_type: str = "raw_wav",
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

        # ── text --------------------------------------------------------------
        toks = self.tok(
            row["text"], max_length=self.text_max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        out["text_input_ids"]   = toks["input_ids"].squeeze(0)
        out["text_attention_mask"] = toks["attention_mask"].squeeze(0)

        # ── audio -------------------------------------------------------------
        if self.audio_input_type == "hf_features" and "audio_features" in row:
            out["audio_input_values"] = torch.tensor(row["audio_features"])
        else:
            wav, sr = self._safe_load(str(row["audio_path"]))
            out["raw_audio"]    = wav / wav.abs().max().clamp(min=1e-5)
            out["sampling_rate"] = sr

        # ── labels & meta -----------------------------------------------------
        out["labels"]      = torch.tensor(row["label"], dtype=torch.long)
        did = row.get("Dialogue_ID", row.get("dialogue_id"))
        out["dialogue_id"] = int(did) if did is not None else -1
        uid = row.get("Utterance_ID", row.get("utterance_id", 0))
        out["utt_id"] = int(uid)
        # map speaker string → stable integer id
        sid = row.get("Speaker", row.get("speaker", "UNK"))
        if not hasattr(self, "_spkmap"):
            self._spkmap = {}
        if sid not in self._spkmap:
            self._spkmap[sid] = len(self._spkmap)
        out["speaker_id"] = self._spkmap[sid]
        return out

    @staticmethod
    def _safe_load(path: str, sr: int = 16000):
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            wav, sr = torch.zeros(1, sr), sr
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        return wav, sr

################################################################################
# 2.  Lightning DataModule
################################################################################

class MELDDataModule(pl.LightningDataModule):
    """Switches between utterance‑ and dialogue‑level collation at runtime."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ds: Dict[str, MELDDataset] = {}

    # ------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        if self.ds:
            return
        root = Path(self.cfg.data_root) / "processed/features/hf_datasets"
        for split in ("train", "dev", "test"):
            hf_ds = hf.load_from_disk(str(root / split))
            self.ds[split] = MELDDataset(
                hf_ds,
                self.cfg.text_encoder_model_name,
                audio_input_type=getattr(self.cfg, "audio_input_type", "raw_wav"),
                text_max_len=getattr(self.cfg, "text_max_len", 128),
            )

    # ------------------------------------------------------------------
    # 2.1 utterance‑level collate (baseline)
    # ------------------------------------------------------------------
    def _to_5tuple(self, rows: List[Dict[str, Any]]):
        txt  = torch.stack([r["text_input_ids"]      for r in rows])
        tmsk = torch.stack([r["text_attention_mask"] for r in rows])
        lab  = torch.stack([r["labels"]              for r in rows])

        if "audio_input_values" in rows[0]:
            wav = torch.nn.utils.rnn.pad_sequence([r["audio_input_values"] for r in rows], batch_first=True)
            wmsk = (wav.abs().sum(-1) != 0).long()
        else:
            wavs = [r["raw_audio"].squeeze(0) for r in rows]
            wav  = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True)
            wmsk = (wav != 0.0).long()
        return wav, wmsk, txt, tmsk, lab

    # ------------------------------------------------------------------
    # 2.1-bis: helper to build 7- / 8-tuple from per-dialog buckets
    # ------------------------------------------------------------------
    def _pack_dialog(self, bucket: List[Dict[str,Any]]):
        """
        Returns:
          wav, wav_mask, txt, txt_mask, labels, speaker_id, dialog_mask
        Extra fields (topic_id, kn_vec) are added later for TOD-KAT.
        """
        T        = len(bucket)
        # ---------- audio ----------
        if "audio_input_values" in bucket[0]:
            a = torch.nn.utils.rnn.pad_sequence(
                    [u["audio_input_values"] for u in bucket], batch_first=True)
            m = (a.abs().sum(-1) != 0).long()
        else:
            raw = [u["raw_audio"].squeeze(0) for u in bucket]
            a   = torch.nn.utils.rnn.pad_sequence(raw, batch_first=True)
            m   = (a != 0.0).long()
        # ---------- text  ----------
        t  = torch.stack([u["text_input_ids"]      for u in bucket])
        tm = torch.stack([u["text_attention_mask"] for u in bucket])
        l  = torch.stack([u["labels"]              for u in bucket])
        s  = torch.tensor([u["speaker_id"]         for u in bucket])
        dmask = torch.ones(T, dtype=torch.bool)
        return a, m, t, tm, l, s, dmask
  
    # ------------------------------------------------------------------
    # 2.2 dialogue‑level collate (context models)
    # ------------------------------------------------------------------
    def _collate_dialog(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # group by dialogue
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for row in batch:
            buckets[row["dialogue_id"]].append(row)
        for uts in buckets.values():
            uts.sort(key=lambda r: r["utt_id"])  # chronological

        max_T = max(len(uts) for uts in buckets.values())

        def pad2T(t: torch.Tensor, pad_val: float = 0.0) -> torch.Tensor:
            """Pad *time* dimension (dim-0) to max_T, leave other dims intact."""
            diff = max_T - t.shape[0]
            if diff <= 0:
                return t
            if t.dim() == 1:          # (T,) → use 1-D pad (left,right)
                return F.pad(t, (0, diff), value=pad_val)
            else:                     # (T, …) → pad rows only
                return F.pad(t, (0, 0, 0, diff), value=pad_val)

        wavs  = []; wmsk = []; txt = []; tmsk = []
        lab   = []; spk  = []; dmask = []
        tid   = []; knv  = [] 
      
        for uts in buckets.values():
            a,m,t,tm,l,s,z = self._pack_dialog(uts)
            wavs.append(pad2T(a,0.0));   wmsk.append(pad2T(m,0))
            txt.append(pad2T(t,0));      tmsk.append(pad2T(tm,0))
            lab.append(pad2T(l,-1));     spk.append(pad2T(s,-1))
            dmask.append(pad2T(z,0))
            # ---- extras for TOD-KAT ----
            tid.append(pad2T(torch.zeros_like(l),0))                # topic-id all 0
            knv.append(pad2T(torch.zeros(l.size(0),50),0.0))        # kn vec zeros

        return {
            "wav":         torch.stack(wavs),
            "wav_mask":    torch.stack(wmsk),
            "txt":         torch.stack(txt),
            "txt_mask":    torch.stack(tmsk),
            "labels":      torch.stack(lab),
            "speaker_id":  torch.stack(spk),
            "dialog_mask": torch.stack(dmask).bool(),
            "topic_id":    torch.stack(tid),        # harmless extra for Dialog-RNN
            "kn_vec":      torch.stack(knv),        # idem
        }
 

    # ------------------------------------------------------------------
    # 2.3 generic loader builder
    # ------------------------------------------------------------------
    def _make_loader(self, split: str, shuffle: bool):
        ds = self.ds[split]

        sampler = None
        arch = getattr(self.cfg, "architecture_name", "").lower()
      
        if split == "train" and arch not in {"dialog_rnn","todkat_lite"}:
            # only do utterance‐level balancing for MLP baseline
            y = torch.tensor(ds.ds["label"])
            w = (1.0 / torch.bincount(y, minlength=self.cfg.output_dim))[y]
            sampler, shuffle = WeightedRandomSampler(w, len(w), replacement=True), False


        
        if arch in {"dialog_rnn", "todkat_lite"}:
            # build mapping from Dialogue_ID → list of dataset indices
            mapping: Dict[int, List[int]] = defaultdict(list)
            for idx in range(len(ds)):
                row = ds.ds[idx]
                mapping[row["dialogue_id"]].append(idx)

            batch_sampler = DialogueBatchSampler(
                dialogue_to_indices=mapping,
                batch_size=self.cfg.batch_size,
                shuffle=(split == "train"),
            )
            # collate all utterances in each dialogue together
            collate = self._collate_dialog
            return DataLoader(
                ds,
                batch_sampler=batch_sampler,            # yields lists of utterance‐indices
                num_workers=getattr(self.cfg, "dataloader_num_workers", 4),
                pin_memory=True,
                collate_fn=collate,
            )
        else:
            # utterance‐level baseline
            collate = self._to_5tuple
            return DataLoader(
                ds,
                batch_size=self.cfg.batch_size,
                sampler=sampler,
                shuffle=shuffle,
                num_workers=getattr(self.cfg, "dataloader_num_workers", 4),
                pin_memory=True,
                collate_fn=collate,
            )

    # Lightning hooks ---------------------------------------------------
    def train_dataloader(self):
        return self._make_loader("train", True)

    def val_dataloader(self):
        return self._make_loader("dev", False)

    def test_dataloader(self):
        return self._make_loader("test", False)


################################################################################
# 3. helper scripts (optional) wrapped to avoid import‑time execution ----------
################################################################################

if __name__ == "__main__":
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

    pass
