#!/usr/bin/env python3
"""
Test script to verify the entire pipeline using test data.
"""

import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from configs.base_config import BaseConfig
from utils.data_processor import MELDDataModule, prepare_hf_dataset
from models.mlp_fusion import MultimodalFusionMLP
from models.teacher import TeacherTransformer
from models.student import StudentGRU
from models.panns_fusion import PaNNsFusion

# Test configuration
TEST_CONFIG = {
    "experiment_name": "test_experiment",
    "batch_size": 2,
    "num_epochs": 1,
    "text_encoder_model_name": "distilbert-base-uncased",
    "audio_encoder_model_name": "facebook/wav2vec2-base",
    "learning_rate": 1e-4,
    "num_workers": 0  # Use 0 workers for testing to avoid issues with forking
}

# Model configurations for testing
MODEL_CONFIGS = {
    "mlp_fusion": {
        "class": MultimodalFusionMLP,
        "params": {
            "mlp_hidden_size": 128,
            "mlp_dropout_rate": 0.1,
            "freeze_text_encoder": True,
            "freeze_audio_encoder": True,
            "audio_input_type": "hf_features"
        }
    },
    "teacher": {
        "class": TeacherTransformer,
        "params": {
            "hidden_size": 128,
            "num_heads": 4,
            "num_layers": 2,
            "dropout_rate": 0.1,
            "freeze_text_encoder": True,
            "freeze_audio_encoder": True,
            "audio_input_type": "hf_features"
        }
    },
    "student": {
        "class": StudentGRU,
        "params": {
            "hidden_size": 64,
            "num_layers": 1,
            "dropout_rate": 0.1,
            "freeze_text_encoder": True,
            "freeze_audio_encoder": True,
            "audio_input_type": "hf_features"
        }
    },
    "panns_fusion": {
        "class": PaNNsFusion,
        "params": {
            "hidden_size": 128,
            "dropout_rate": 0.1,
            "freeze_text_encoder": True,
            "freeze_audio_encoder": True,
            "audio_input_type": "hf_features"
        }
    }
}

class TestPipeline:
    """Test the entire pipeline from data loading to model training."""
    
    @classmethod
    def setup_class(cls):
        """Set up test configuration and data."""
        # Get test data directory
        cls.test_data_dir = Path(__file__).parent / "test_data"
        
        # Create test configuration
        cls.config = Config(
            experiment_name=TEST_CONFIG["experiment_name"],
            batch_size=TEST_CONFIG["batch_size"],
            num_epochs=TEST_CONFIG["num_epochs"],
            data_root=str(cls.test_data_dir),
            output_dir=str(cls.test_data_dir / "output"),
            text_encoder_model_name=TEST_CONFIG["text_encoder_model_name"],
            audio_encoder_model_name=TEST_CONFIG["audio_encoder_model_name"],
            learning_rate=TEST_CONFIG["learning_rate"],
            num_workers=TEST_CONFIG["num_workers"]
        )
        
        # Ensure output directory exists
        cls.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def test_data_loading(self):
        """Test data loading and preprocessing."""
        # Prepare dataset
        dataset = prepare_hf_dataset(self.config)
        
        # Check dataset splits
        assert set(dataset.keys()) == {"train", "validation", "test"}, \
            f"Expected splits ['train', 'validation', 'test'], got {list(dataset.keys())}"
        
        # Check features in each split
        expected_features = {"text", "audio", "emotion", "sentiment", "dialogue_id", "utterance_id"}
        for split_name, split_data in dataset.items():
            assert set(split_data.features.keys()) >= expected_features, \
                f"Missing features in {split_name} split"
            
            # Check data types
            assert isinstance(split_data[0]["text"], str), "Text should be a string"
            assert isinstance(split_data[0]["emotion"], int), "Emotion should be an integer"
            assert isinstance(split_data[0]["sentiment"], str), "Sentiment should be a string"
    
    def test_data_module(self):
        """Test PyTorch Lightning data module."""
        # Initialize data module
        dm = MELDDataModule(self.config)
        
        # Setup data module
        dm.setup()
        
        # Check dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
        
        # Check batch sizes
        batch = next(iter(train_loader))
        assert batch["text"].shape[0] == self.config.batch_size, \
            f"Expected batch size {self.config.batch_size}, got {batch['text'].shape[0]}"
    
    @pytest.mark.parametrize("model_name", MODEL_CONFIGS.keys())
    def test_model_initialization(self, model_name):
        """Test model initialization for all architectures."""
        model_config = MODEL_CONFIGS[model_name]
        model_class = model_config["class"]
        model_params = model_config["params"].copy()
        
        # Add common parameters
        model_params.update({
            "text_encoder_model_name": self.config.text_encoder_model_name,
            "audio_encoder_model_name": self.config.audio_encoder_model_name,
            "text_feature_dim": 768,  # For distilbert-base-uncased
            "audio_feature_dim": 768,  # For wav2vec2-base
            "output_dim": 7,  # Number of emotion classes
            "learning_rate": self.config.learning_rate
        })
        
        # Initialize model
        model = model_class(**model_params)
        
        # Check model device
        assert next(model.parameters()).device == self.config.device, \
            f"Model not on expected device {self.config.device}"
        
        # Check forward pass
        batch_size = 2
        dummy_batch = {
            "text": ["Test sentence 1", "Test sentence 2"],
            "audio": torch.randn(batch_size, 16000),  # 1 second of audio at 16kHz
            "emotion": torch.randint(0, 7, (batch_size,)),
            "sentiment": ["positive", "negative"]
        }
        
        # Move batch to device
        dummy_batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in dummy_batch.items()}
        
        # Forward pass
        outputs = model(dummy_batch)
        
        # Check output shape
        assert outputs.logits.shape == (batch_size, 7), \
            f"Expected output shape {(batch_size, 7)}, got {outputs.logits.shape}"
    
    @pytest.mark.slow
    def test_training_loop(self):
        """Test training loop with a small number of batches."""
        # Use MLP Fusion for testing as it's the simplest model
        model_config = MODEL_CONFIGS["mlp_fusion"]
        model_class = model_config["class"]
        model_params = model_config["params"].copy()
        
        # Add common parameters
        model_params.update({
            "text_encoder_model_name": self.config.text_encoder_model_name,
            "audio_encoder_model_name": self.config.audio_encoder_model_name,
            "text_feature_dim": 768,
            "audio_feature_dim": 768,
            "output_dim": 7,
            "learning_rate": self.config.learning_rate
        })
        
        # Initialize model
        model = model_class(**model_params)
        
        # Initialize data module
        dm = MELDDataModule(self.config)
        dm.setup()
        
        # Train for 1 epoch with 2 batches
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            accelerator="auto",
            devices=1
        )
        
        # Train the model
        trainer.fit(model, datamodule=dm)
        
        # Check if model parameters were updated
        initial_params = [p.clone() for p in model.parameters()]
        trainer.fit_loop.epoch_loop.optimizer_loop.optim_progress.optimizer.step()
        updated_params = [p.clone() for p in model.parameters()]
        
        # At least some parameters should have changed
        assert any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, updated_params)), \
            "Model parameters were not updated during training"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    
    # Create data module with test data
    print("Creating data module with test data...")
    data_module = MELDDataModule(config)
    data_module.prepare_data()
    data_module.setup()
    
    # Create model based on architecture
    print(f"\nCreating {args.architecture} model...")
    if args.architecture == "mlp_fusion":
        from models.mlp_fusion import MultimodalFusionMLP
        model = MultimodalFusionMLP(
            mlp_hidden_size=64,  # Smaller for testing
            mlp_dropout_rate=0.3,
            text_encoder_model_name=config.text_encoder_model_name,
            audio_encoder_model_name=config.audio_encoder_model_name,
            text_feature_dim=768,
            audio_feature_dim=768,
            freeze_text_encoder=True,
            freeze_audio_encoder=True,
            audio_input_type="hf_features",
            output_dim=7,
            learning_rate=1e-4
        )
    elif args.architecture == "teacher":
        from models.teacher import TeacherTransformer
        model = TeacherTransformer(
            hidden_size=64,  # Smaller for testing
            num_transformer_layers=1,  # Fewer layers for testing
            num_transformer_heads=2,  # Fewer heads for testing
            dropout_rate=0.3,
            text_encoder_model_name=config.text_encoder_model_name,
            audio_encoder_model_name=config.audio_encoder_model_name,
            text_feature_dim=768,
            audio_feature_dim=768,
            freeze_text_encoder=True,
            freeze_audio_encoder=True,
            audio_input_type="hf_features",
            output_dim=7,
            learning_rate=1e-4
        )
    elif args.architecture == "student":
        from models.student import StudentGRU
        model = StudentGRU(
            hidden_size=64,  # Smaller for testing
            gru_layers=1,    # Fewer layers for testing
            dropout_rate=0.3,
            text_encoder_model_name=config.text_encoder_model_name,
            audio_encoder_model_name=config.audio_encoder_model_name,
            text_feature_dim=768,
            audio_feature_dim=768,
            freeze_text_encoder=True,
            freeze_audio_encoder=True,
            audio_input_type="hf_features",
            output_dim=7,
            learning_rate=1e-4
        )
    elif args.architecture == "panns_fusion":
        from models.panns_fusion import PaNNsFusion
        model = PaNNsFusion(
            hidden_size=64,  # Smaller for testing
            dropout_rate=0.3,
            text_encoder_model_name=config.text_encoder_model_name,
            use_panns_features=True,
            panns_feature_dim=2048,
            text_feature_dim=768,
            freeze_text_encoder=True,
            output_dim=7,
            learning_rate=1e-4
        )
    else:
        raise ValueError(f"Unknown architecture: {args.architecture}")
    
    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_save_dir,
        filename=f"{args.architecture}-test-{{epoch:02d}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    
    # Set up trainer with limited epochs
    print(f"\nSetting up trainer for {args.epochs} epochs...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        logger=False,  # Disable logging for test
        enable_checkpointing=True,
        enable_model_summary=True,
        num_sanity_val_steps=1
    )
    
    # Run training for a few epochs
    print("\nStarting test training...")
    try:
        trainer.fit(model, data_module)
        print(f"\n✓ Training completed successfully for {args.epochs} epochs")
        
        # Run evaluation
        print("\nRunning evaluation on test data...")
        test_results = trainer.test(model, data_module)[0]
        print(f"\nTest results: {test_results}")
        
        print(f"\n===== Test completed successfully for {args.architecture} =====\n")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
