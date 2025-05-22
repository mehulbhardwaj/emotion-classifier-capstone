#!/usr/bin/env python3
"""
Test script for utility functions.
"""

import os
import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from configs.base_config import BaseConfig
from utils.data_processor import (
    download_meld_dataset,
    convert_mp4_to_wav,
    prepare_hf_dataset,
    get_class_distribution
)

class TestDataProcessor:
    """Test data processing utilities."""
    
    @classmethod
    def setup_class(cls):
        """Set up test configuration."""
        cls.config = BaseConfig()
        test_data_dir = Path(__file__).parent / "test_data"
        cls.config.data_dir = str(test_data_dir)
        cls.config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_download_meld_dataset(self, tmp_path):
        """Test dataset download functionality."""
        # Test with a temporary directory
        test_dir = tmp_path / "meld_test"
        download_meld_dataset(str(test_dir))
        
        # Check if required files exist
        assert (test_dir / "train_sent_emo.csv").exists(), "Training CSV not found"
        assert (test_dir / "dev_sent_emo.csv").exists(), "Validation CSV not found"
        assert (test_dir / "test_sent_emo.csv").exists(), "Test CSV not found"
    
    def test_convert_mp4_to_wav(self, tmp_path):
        """Test MP4 to WAV conversion."""
        # Create a dummy MP4 file for testing
        mp4_path = tmp_path / "test.mp4"
        with open(mp4_path, "wb") as f:
            f.write(b"dummy mp4 data")
        
        # Test conversion
        wav_path = convert_mp4_to_wav(str(mp4_path), str(tmp_path))
        
        # Check if WAV file was created
        assert Path(wav_path).exists(), "WAV file not created"
        assert Path(wav_path).suffix == ".wav", "Output file is not a WAV file"
    
    def test_prepare_hf_dataset(self):
        """Test HuggingFace dataset preparation."""
        dataset = prepare_hf_dataset(self.config)
        
        # Check dataset splits
        assert set(dataset.keys()) == {"train", "validation", "test"}, \
            f"Expected splits ['train', 'validation', 'test'], got {list(dataset.keys())}"
        
        # Check features
        expected_features = {"text", "audio", "emotion", "sentiment", "dialogue_id", "utterance_id"}
        for split in dataset.values():
            assert set(split.features.keys()) >= expected_features, \
                f"Missing expected features in {split}"
    
    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        # Create a dummy dataset
        class DummyDataset:
            def __init__(self, labels):
                self.labels = labels
            
            def __getitem__(self, idx):
                return {"emotion": self.labels[idx]}
            
            def __len__(self):
                return len(self.labels)
        
        # Test with balanced classes
        balanced_labels = [0, 1, 2, 0, 1, 2]
        balanced_ds = DummyDataset(balanced_labels)
        dist = get_class_distribution(balanced_ds, num_classes=3)
        assert np.allclose(dist, [2/6, 2/6, 2/6]), f"Unexpected balanced distribution: {dist}"
        
        # Test with imbalanced classes
        imbalanced_labels = [0, 0, 1, 2, 2, 2]
        imbalanced_ds = DummyDataset(imbalanced_labels)
        dist = get_class_distribution(imbalanced_ds, num_classes=3)
        assert np.allclose(dist, [2/6, 1/6, 3/6]), f"Unexpected imbalanced distribution: {dist}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
