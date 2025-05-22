#!/usr/bin/env python3
"""
Test script for model training and evaluation.
"""

import os
import sys
import pytest
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from config import Config
from models.mlp_fusion import MultimodalFusionMLP
from utils.data_processor import prepare_hf_dataset

def setup_test_config():
    """Set up test configuration."""
    config = Config()
    
    # Set up test data paths
    test_data_dir = Path(__file__).parent / "test_data"
    config.data_dir = str(test_data_dir)
    
    # Use CPU for testing
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters for testing
    config.batch_size = 2
    config.num_epochs = 1
    config.learning_rate = 1e-4
    
    return config

def test_model_training():
    """Test model training loop."""
    config = setup_test_config()
    
    # Initialize model
    model = MultimodalFusionMLP(
        mlp_hidden_size=128,
        mlp_dropout_rate=0.1,
        text_encoder_model_name="distilbert-base-uncased",
        audio_encoder_model_name="facebook/wav2vec2-base",
        text_feature_dim=768,
        audio_feature_dim=768,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        audio_input_type="hf_features",
        output_dim=7,
        learning_rate=config.learning_rate
    ).to(config.device)
    
    # Load test dataset
    dataset = prepare_hf_dataset(config)
    train_loader = torch.utils.data.DataLoader(
        dataset["train"].with_format("torch"),
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Test training step
    model.train()
    batch = next(iter(train_loader))
    
    # Move batch to device
    batch = {k: v.to(config.device) for k, v in batch.items()}
    
    # Forward pass
    outputs = model(batch)
    
    # Check output shape
    assert outputs.logits.shape == (config.batch_size, 7), \
        f"Expected output shape {(config.batch_size, 7)}, got {outputs.logits.shape}"
    
    # Test loss calculation
    loss = outputs.loss
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients for backprop"

def test_model_evaluation():
    """Test model evaluation."""
    config = setup_test_config()
    
    # Initialize model
    model = MultimodalFusionMLP(
        mlp_hidden_size=128,
        mlp_dropout_rate=0.1,
        text_encoder_model_name="distilbert-base-uncased",
        audio_encoder_model_name="facebook/wav2vec2-base",
        text_feature_dim=768,
        audio_feature_dim=768,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        audio_input_type="hf_features",
        output_dim=7,
        learning_rate=config.learning_rate
    ).to(config.device)
    
    # Load test dataset
    dataset = prepare_hf_dataset(config)
    val_loader = torch.utils.data.DataLoader(
        dataset["validation"].with_format("torch"),
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Test evaluation step
    model.eval()
    batch = next(iter(val_loader))
    
    # Move batch to device
    batch = {k: v.to(config.device) for k, v in batch.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    # Check output shape
    assert outputs.logits.shape == (config.batch_size, 7), \
        f"Expected output shape {(config.batch_size, 7)}, got {outputs.logits.shape}"
    
    # Check predictions
    preds = torch.argmax(outputs.logits, dim=1)
    assert preds.shape == (config.batch_size,), \
        f"Expected predictions shape {(config.batch_size,)}, got {preds.shape}"

if __name__ == "__main__":
    test_model_training()
    test_model_evaluation()
    print("All tests passed!")
