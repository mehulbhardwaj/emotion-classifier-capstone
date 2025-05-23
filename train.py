#!/usr/bin/env python3
"""
Training script for emotion classification models.

Simplified training process using PyTorch Lightning.
"""

import os
import argparse
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
from configs import Config
from utils.data_processor import MELDDataModule

# Import model architectures
from models.mlp_fusion import MultimodalFusionMLP
from models.teacher import TeacherTransformer
from models.student import StudentGRU
from models.panns_fusion import PaNNsFusion
from models.dialog_rnn import DialogRNNMLP
from models.todkat_lite import TodkatLiteMLP


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def train(config):
    """Train a model based on the provided configuration."""
    # Set random seed
    set_seed(config.random_seed)
    
    # Create output directories
    config.create_directories()
    
    # Create data module
    data_module = MELDDataModule(config)
    
    # Create the model based on architecture name
    if config.architecture_name == "mlp_fusion":
        model = MultimodalFusionMLP(config)
    elif config.architecture_name == "todkat_lite":
        model = TodkatLiteMLP(config)
    elif config.architecture_name == "dialog_rnn":
        model = DialogRNNMLP(config)    
    elif config.architecture_name == "teacher":
        model = TeacherTransformer(
            hidden_size=getattr(config, "hidden_size", 256),
            num_transformer_layers=getattr(config, "num_transformer_layers", 2),
            num_transformer_heads=getattr(config, "num_transformer_heads", 4),
            dropout_rate=getattr(config, "dropout_rate", 0.3),
            text_encoder_model_name=config.text_encoder_model_name,
            audio_encoder_model_name=config.audio_encoder_model_name,
            text_feature_dim=config.text_feature_dim,
            audio_feature_dim=config.audio_feature_dim,
            freeze_text_encoder=config.freeze_text_encoder,
            freeze_audio_encoder=config.freeze_audio_encoder,
            audio_input_type=config.audio_input_type,
            output_dim=config.output_dim,
            learning_rate=config.learning_rate
        )
    elif config.architecture_name == "student":
        model = StudentGRU(
            hidden_size=getattr(config, "hidden_size", 128),
            gru_layers=getattr(config, "gru_layers", 2),
            dropout_rate=getattr(config, "dropout_rate", 0.3),
            text_encoder_model_name=config.text_encoder_model_name,
            audio_encoder_model_name=config.audio_encoder_model_name,
            text_feature_dim=config.text_feature_dim,
            audio_feature_dim=config.audio_feature_dim,
            freeze_text_encoder=config.freeze_text_encoder,
            freeze_audio_encoder=config.freeze_audio_encoder,
            audio_input_type=config.audio_input_type,
            output_dim=config.output_dim,
            learning_rate=config.learning_rate,
            use_distillation=getattr(config, "use_distillation", False)
        )
    elif config.architecture_name == "panns_fusion":
        model = PaNNsFusion(
            hidden_size=getattr(config, "hidden_size", 256),
            dropout_rate=getattr(config, "dropout_rate", 0.3),
            text_encoder_model_name=config.text_encoder_model_name,
            use_panns_features=getattr(config, "use_panns_features", True),
            panns_feature_dim=getattr(config, "panns_feature_dim", 2048),
            text_feature_dim=config.text_feature_dim,
            freeze_text_encoder=config.freeze_text_encoder,
            output_dim=config.output_dim,
            learning_rate=config.learning_rate
        )
    else:
        raise ValueError(f"Architecture {config.architecture_name} not implemented. "
                         f"Choose from: mlp_fusion, teacher, student, panns_fusion, todkat_lite, dialog_rnn.")
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=config.logs_dir,
        name=config.experiment_name,
        version=config.run_name
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_save_dir,
        filename=f"{config.architecture_name}-{config.experiment_name}-{{epoch:02d}}-{{val_f1:.4f}}",
        save_top_k=3,
        monitor="val_f1",
        mode="max"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_f1",
        patience=config.early_stopping_patience,
        mode="max"
    )
    
    # Set up trainer
    precision = "16-mixed" if config.use_mixed_precision else "32-true"
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",  # Automatically choose CPU/GPU/MPS
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        precision=precision,
        accumulate_grad_batches=config.grad_accumulation_steps,
        deterministic=True  # For reproducibility
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    trainer.test(model, data_module)
    
    # Save configuration
    config_path = config.experiment_output_dir / "config.yaml"
    config.save_to_yaml(str(config_path))
    
    # Return the best checkpoint path
    return checkpoint_callback.best_model_path


def main():
    """Main function for training emotion classification models."""
    parser = argparse.ArgumentParser(description="Train emotion classification models")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--architecture", type=str, default="mlp_fusion", help="Model architecture")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--seed", type=int, help="Random seed")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        # Load from YAML file
        config = Config.from_yaml(args.config)
    else:
        # Create with default values
        config = Config()
    
    # Override with command-line arguments
    if args.architecture:
        config.architecture_name = args.architecture
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.seed:
        config.random_seed = args.seed
    
    # Run training
    best_model_path = train(config)
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
