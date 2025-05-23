"""ULTRA-VERBOSE DEBUG VERSION OF DATA PROCESSOR"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pytorch_lightning as pl

print("🔧 LOADING DATA_PROCESSOR_DEBUG MODULE")

class MELDDataset(Dataset):
    """MELD dataset with verbose debugging."""
    
    def __init__(self, config, split: str):
        print(f"📊 CREATING MELDDataset for split: {split}")
        print(f"   Config architecture_name: {getattr(config, 'architecture_name', 'NOT_FOUND')}")
        
        self.config = config
        self.split = split
        self.data_dir = Path(config.data_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        print(f"   Data directory: {self.data_dir}")
        print(f"   Split: {split}")
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_dir / f"{split}_sent_emo.csv")
        print(f"   Loaded metadata: {len(self.metadata)} samples")
        
        # Emotion mapping
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3,
            'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        print(f"   Emotion mapping: {self.emotion_map}")
        
        # Architecture-specific data loading
        if hasattr(config, 'architecture_name'):
            arch = config.architecture_name
        else:
            arch = getattr(config, 'architecture', 'mlp_fusion')  # fallback
        
        print(f"   Detected architecture: {arch}")
        
        if arch in ["dialog_rnn", "todkat_lite"]:
            print(f"   Using dialog-level processing for {arch}")
            self._load_dialog_data()
        else:
            print(f"   Using utterance-level processing for {arch}")
            self._load_utterance_data()
    
    def _load_utterance_data(self):
        """Load data for utterance-level models (MLP fusion)."""
        print(f"   🔧 Loading utterance-level data...")
        
        self.samples = []
        audio_dir = self.data_dir / f"{self.split}_audio"
        
        for idx, row in self.metadata.iterrows():
            if idx % 100 == 0:
                print(f"      Processing sample {idx}/{len(self.metadata)}")
            
            sample = {
                'dialogue_id': row['Dialogue_ID'],
                'utterance_id': row['Utterance_ID'],
                'speaker': row['Speaker'],
                'emotion': self.emotion_map[row['Emotion']],
                'text': row['Utterance'],
                'audio_path': audio_dir / f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
            }
            
            # Check if audio file exists
            if sample['audio_path'].exists():
                self.samples.append(sample)
            else:
                print(f"      ⚠️  Missing audio: {sample['audio_path']}")
        
        print(f"   ✅ Loaded {len(self.samples)} utterance samples")
    
    def _load_dialog_data(self):
        """Load data for dialog-level models (DialogRNN, TOD-KAT)."""
        print(f"   🔧 Loading dialog-level data...")
        
        # Group by dialogue
        dialogs = {}
        audio_dir = self.data_dir / f"{self.split}_audio"
        
        for idx, row in self.metadata.iterrows():
            dia_id = row['Dialogue_ID']
            if dia_id not in dialogs:
                dialogs[dia_id] = []
            
            utterance = {
                'utterance_id': row['Utterance_ID'],
                'speaker': row['Speaker'],
                'emotion': self.emotion_map[row['Emotion']],
                'text': row['Utterance'],
                'audio_path': audio_dir / f"dia{dia_id}_utt{row['Utterance_ID']}.wav"
            }
            
            if utterance['audio_path'].exists():
                dialogs[dia_id].append(utterance)
            else:
                print(f"      ⚠️  Missing audio: {utterance['audio_path']}")
        
        # Convert to list and sort
        self.samples = []
        for dia_id, utterances in dialogs.items():
            utterances.sort(key=lambda x: x['utterance_id'])
            self.samples.append({
                'dialogue_id': dia_id,
                'utterances': utterances
            })
        
        print(f"   ✅ Loaded {len(self.samples)} dialogs")
        
        # Dialog statistics
        lengths = [len(d['utterances']) for d in self.samples]
        print(f"   Dialog lengths - min: {min(lengths)}, max: {max(lengths)}, avg: {np.mean(lengths):.1f}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if hasattr(self.config, 'architecture_name'):
            arch = self.config.architecture_name
        else:
            arch = getattr(self.config, 'architecture', 'mlp_fusion')
        
        if arch in ["dialog_rnn", "todkat_lite"]:
            print(f"   📊 Getting dialog sample {idx} for {arch}")
            return self._get_dialog_sample(sample)
        else:
            print(f"   📊 Getting utterance sample {idx} for {arch}")
            return self._get_utterance_sample(sample)
    
    def _get_utterance_sample(self, sample):
        """Get single utterance sample."""
        print(f"      Processing utterance {sample['utterance_id']}")
        
        # Load and process audio
        audio_path = sample['audio_path']
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            if len(audio) == 0:
                audio = np.zeros(16000)  # 1 second of silence
            print(f"      Audio loaded: {len(audio)} samples")
        except Exception as e:
            print(f"      ⚠️  Audio load failed: {e}")
            audio = np.zeros(16000)
        
        # Tokenize text
        text_tokens = self.tokenizer(
            sample['text'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        print(f"      Text tokenized: {text_tokens['input_ids'].shape}")
        
        return {
            'wav': torch.FloatTensor(audio),
            'wav_mask': torch.ones(len(audio), dtype=torch.bool),
            'txt': text_tokens['input_ids'].squeeze(0),
            'txt_mask': text_tokens['attention_mask'].squeeze(0),
            'labels': torch.LongTensor([sample['emotion']]),
            'speaker_id': torch.LongTensor([hash(sample['speaker']) % 100])
        }
    
    def _get_dialog_sample(self, sample):
        """Get dialog sample."""
        utterances = sample['utterances']
        dialog_id = sample['dialogue_id']
        print(f"      Processing dialog {dialog_id} with {len(utterances)} utterances")
        
        max_len = getattr(self.config, 'max_sequence_length', 20)
        
        # Truncate if too long
        if len(utterances) > max_len:
            utterances = utterances[-max_len:]
            print(f"      Truncated to {len(utterances)} utterances")
        
        # Process each utterance
        wav_list, wav_mask_list = [], []
        txt_list, txt_mask_list = [], []
        labels_list, speaker_list = [], []
        
        for i, utt in enumerate(utterances):
            print(f"         Processing utterance {i+1}/{len(utterances)}")
            
            # Audio
            try:
                audio, sr = librosa.load(utt['audio_path'], sr=16000)
                if len(audio) == 0:
                    audio = np.zeros(16000)
            except:
                audio = np.zeros(16000)
            
            # Text
            tokens = self.tokenizer(
                utt['text'],
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            wav_list.append(torch.FloatTensor(audio))
            wav_mask_list.append(torch.ones(len(audio), dtype=torch.bool))
            txt_list.append(tokens['input_ids'].squeeze(0))
            txt_mask_list.append(tokens['attention_mask'].squeeze(0))
            labels_list.append(utt['emotion'])
            speaker_list.append(hash(utt['speaker']) % 100)
        
        # Pad sequences
        seq_len = len(utterances)
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            print(f"      Padding {pad_len} utterances")
            # Add padding
            for _ in range(pad_len):
                wav_list.append(torch.zeros(16000))
                wav_mask_list.append(torch.zeros(16000, dtype=torch.bool))
                txt_list.append(torch.zeros(128, dtype=torch.long))
                txt_mask_list.append(torch.zeros(128, dtype=torch.bool))
                labels_list.append(-1)  # Ignore index
                speaker_list.append(-1)
        
        # Create dialog mask
        dialog_mask = torch.zeros(max_len, dtype=torch.bool)
        dialog_mask[:seq_len] = True
        
        print(f"      Final dialog shape: {max_len} utterances, {seq_len} valid")
        
        return {
            'wav': torch.stack(wav_list),
            'wav_mask': torch.stack(wav_mask_list),
            'txt': torch.stack(txt_list),
            'txt_mask': torch.stack(txt_mask_list),
            'labels': torch.LongTensor(labels_list),
            'speaker_id': torch.LongTensor(speaker_list),
            'dialog_mask': dialog_mask
        }


def collate_fn_utterance(batch):
    """Collate function for utterance-level data."""
    print(f"🔧 COLLATING UTTERANCE BATCH: {len(batch)} samples")
    
    # Find max lengths
    max_wav = max(item['wav'].shape[0] for item in batch)
    max_txt = max(item['txt'].shape[0] for item in batch)
    
    print(f"   Max wav length: {max_wav}")
    print(f"   Max txt length: {max_txt}")
    
    # Pad and stack
    wav_list, wav_mask_list = [], []
    txt_list, txt_mask_list = [], []
    labels_list, speaker_list = [], []
    
    for item in batch:
        # Audio padding
        wav = item['wav']
        if len(wav) < max_wav:
            wav = torch.cat([wav, torch.zeros(max_wav - len(wav))])
        wav_mask = torch.cat([item['wav_mask'], torch.zeros(max_wav - len(item['wav_mask']), dtype=torch.bool)])
        
        # Text padding
        txt = item['txt']
        if len(txt) < max_txt:
            txt = torch.cat([txt, torch.zeros(max_txt - len(txt), dtype=torch.long)])
        txt_mask = torch.cat([item['txt_mask'], torch.zeros(max_txt - len(item['txt_mask']), dtype=torch.bool)])
        
        wav_list.append(wav)
        wav_mask_list.append(wav_mask)
        txt_list.append(txt)
        txt_mask_list.append(txt_mask)
        labels_list.append(item['labels'])
        speaker_list.append(item['speaker_id'])
    
    result = {
        'wav': torch.stack(wav_list),
        'wav_mask': torch.stack(wav_mask_list),
        'txt': torch.stack(txt_list),
        'txt_mask': torch.stack(txt_mask_list),
        'labels': torch.cat(labels_list),
        'speaker_id': torch.cat(speaker_list)
    }
    
    print(f"   ✅ Collated batch shapes:")
    for k, v in result.items():
        print(f"      {k}: {v.shape}")
    
    return result


def collate_fn_dialog(batch):
    """Collate function for dialog-level data."""
    print(f"🔧 COLLATING DIALOG BATCH: {len(batch)} samples")
    
    # Find max lengths
    max_seq = max(item['wav'].shape[0] for item in batch)
    max_wav = max(item['wav'].shape[1] for item in batch)
    max_txt = max(item['txt'].shape[1] for item in batch)
    
    print(f"   Max sequence length: {max_seq}")
    print(f"   Max wav length: {max_wav}")
    print(f"   Max txt length: {max_txt}")
    
    # Pad and stack
    wav_list, wav_mask_list = [], []
    txt_list, txt_mask_list = [], []
    labels_list, speaker_list, dialog_mask_list = [], [], []
    
    for item in batch:
        # Sequence padding
        seq_len = item['wav'].shape[0]
        if seq_len < max_seq:
            pad_len = max_seq - seq_len
            
            # Pad wav
            wav_pad = torch.zeros(pad_len, max_wav)
            wav = torch.cat([item['wav'], wav_pad])
            
            wav_mask_pad = torch.zeros(pad_len, max_wav, dtype=torch.bool)
            wav_mask = torch.cat([item['wav_mask'], wav_mask_pad])
            
            # Pad txt
            txt_pad = torch.zeros(pad_len, max_txt, dtype=torch.long)
            txt = torch.cat([item['txt'], txt_pad])
            
            txt_mask_pad = torch.zeros(pad_len, max_txt, dtype=torch.bool)
            txt_mask = torch.cat([item['txt_mask'], txt_mask_pad])
            
            # Pad labels and speakers
            labels_pad = torch.full((pad_len,), -1, dtype=torch.long)
            labels = torch.cat([item['labels'], labels_pad])
            
            speaker_pad = torch.full((pad_len,), -1, dtype=torch.long)
            speaker = torch.cat([item['speaker_id'], speaker_pad])
            
            # Pad dialog mask
            dialog_mask_pad = torch.zeros(pad_len, dtype=torch.bool)
            dialog_mask = torch.cat([item['dialog_mask'], dialog_mask_pad])
        else:
            wav, wav_mask = item['wav'], item['wav_mask']
            txt, txt_mask = item['txt'], item['txt_mask']
            labels, speaker = item['labels'], item['speaker_id']
            dialog_mask = item['dialog_mask']
        
        # Audio length padding
        if wav.shape[1] < max_wav:
            wav_pad = torch.zeros(wav.shape[0], max_wav - wav.shape[1])
            wav = torch.cat([wav, wav_pad], dim=1)
            
            wav_mask_pad = torch.zeros(wav_mask.shape[0], max_wav - wav_mask.shape[1], dtype=torch.bool)
            wav_mask = torch.cat([wav_mask, wav_mask_pad], dim=1)
        
        # Text length padding
        if txt.shape[1] < max_txt:
            txt_pad = torch.zeros(txt.shape[0], max_txt - txt.shape[1], dtype=torch.long)
            txt = torch.cat([txt, txt_pad], dim=1)
            
            txt_mask_pad = torch.zeros(txt_mask.shape[0], max_txt - txt_mask.shape[1], dtype=torch.bool)
            txt_mask = torch.cat([txt_mask, txt_mask_pad], dim=1)
        
        wav_list.append(wav)
        wav_mask_list.append(wav_mask)
        txt_list.append(txt)
        txt_mask_list.append(txt_mask)
        labels_list.append(labels)
        speaker_list.append(speaker)
        dialog_mask_list.append(dialog_mask)
    
    result = {
        'wav': torch.stack(wav_list),
        'wav_mask': torch.stack(wav_mask_list),
        'txt': torch.stack(txt_list),
        'txt_mask': torch.stack(txt_mask_list),
        'labels': torch.stack(labels_list),
        'speaker_id': torch.stack(speaker_list),
        'dialog_mask': torch.stack(dialog_mask_list)
    }
    
    print(f"   ✅ Collated dialog batch shapes:")
    for k, v in result.items():
        print(f"      {k}: {v.shape}")
    
    return result


class MELDDataModule(pl.LightningDataModule):
    """MELD data module with verbose debugging."""
    
    def __init__(self, config):
        print(f"📊 CREATING MELDDataModule")
        print(f"   Config architecture_name: {getattr(config, 'architecture_name', 'NOT_FOUND')}")
        
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        
        # Determine collate function based on architecture
        if hasattr(config, 'architecture_name'):
            arch = config.architecture_name
        else:
            arch = getattr(config, 'architecture', 'mlp_fusion')
        
        print(f"   Architecture: {arch}")
        
        if arch in ["dialog_rnn", "todkat_lite"]:
            print(f"   Using dialog collate function")
            self.collate_fn = collate_fn_dialog
        else:
            print(f"   Using utterance collate function")
            self.collate_fn = collate_fn_utterance
        
        print(f"   ✅ MELDDataModule created")
    
    def setup(self, stage=None):
        print(f"📋 SETTING UP MELDDataModule (stage: {stage})")
        
        if stage == "fit" or stage is None:
            print(f"   Creating train dataset...")
            self.train_dataset = MELDDataset(self.config, "train")
            print(f"   Creating val dataset...")
            self.val_dataset = MELDDataset(self.config, "dev")
            print(f"   ✅ Train/val datasets created")
        
        if stage == "test" or stage is None:
            print(f"   Creating test dataset...")
            self.test_dataset = MELDDataset(self.config, "test")
            print(f"   ✅ Test dataset created")
        
        print(f"   ✅ Setup completed")
    
    def train_dataloader(self):
        print(f"🚂 CREATING TRAIN DATALOADER")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Collate function: {self.collate_fn.__name__}")
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        print(f"   ✅ Train dataloader created: {len(loader)} batches")
        return loader
    
    def val_dataloader(self):
        print(f"✅ CREATING VAL DATALOADER")
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        print(f"   ✅ Val dataloader created: {len(loader)} batches")
        return loader
    
    def test_dataloader(self):
        print(f"🧪 CREATING TEST DATALOADER")
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        print(f"   ✅ Test dataloader created: {len(loader)} batches")
        return loader

print("✅ DATA_PROCESSOR_DEBUG MODULE LOADED") 