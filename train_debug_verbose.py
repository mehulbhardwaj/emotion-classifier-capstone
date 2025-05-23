#!/usr/bin/env python3
"""ULTRA-VERBOSE DEBUG VERSION OF TRAIN.PY - EVERY STEP TRACED"""

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

print("🚀 STARTING ULTRA-VERBOSE DEBUG TRAINING")
print("=" * 80)

# Import model architectures with debug prints
print("📦 IMPORTING MODEL ARCHITECTURES...")
from models.mlp_fusion import MultimodalFusionMLP
print("   ✅ Imported MultimodalFusionMLP")
from models.dialog_rnn import DialogRNNMLP
print("   ✅ Imported DialogRNNMLP")
from models.todkat_lite import TodkatLiteMLP
print("   ✅ Imported TodkatLiteMLP")

def set_seed(seed):
    """Set random seed for reproducibility."""
    print(f"🎲 SETTING RANDOM SEED: {seed}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        print("   ✅ CUDA seed set")
    import numpy as np
    np.random.seed(seed)
    print("   ✅ NumPy seed set")
    import random
    random.seed(seed)
    print("   ✅ Python random seed set")

def train(config):
    """Train a model based on the provided configuration."""
    print(f"\n🏋️ STARTING TRAIN FUNCTION")
    print(f"   Config type: {type(config)}")
    print(f"   Config architecture_name: '{config.architecture_name}'")
    print(f"   Config dict keys: {[k for k in dir(config) if not k.startswith('_')]}")
    
    # Set random seed
    print(f"\n🎲 SETTING SEED...")
    set_seed(config.random_seed)
    
    # Create output directories
    print(f"\n📁 CREATING DIRECTORIES...")
    config.create_directories()
    print(f"   ✅ Directories created")
    
    # Create data module
    print(f"\n📊 CREATING DATA MODULE...")
    print(f"   About to create MELDDataModule with config.architecture_name = '{config.architecture_name}'")
    data_module = MELDDataModule(config)
    print(f"   ✅ MELDDataModule created: {type(data_module)}")
    
    # Create the model based on architecture name - CRITICAL SECTION
    print(f"\n🏗️ CRITICAL: MODEL CREATION SECTION")
    print(f"   Raw architecture_name: {repr(config.architecture_name)}")
    print(f"   Architecture_name length: {len(config.architecture_name)}")
    print(f"   Architecture_name type: {type(config.architecture_name)}")
    print(f"   Architecture_name stripped: '{config.architecture_name.strip()}'")
    
    # Test each condition explicitly
    print(f"\n🔍 TESTING CONDITIONS:")
    mlp_test = config.architecture_name == "mlp_fusion"
    dialog_test = config.architecture_name == "dialog_rnn"
    todkat_test = config.architecture_name == "todkat_lite"
    
    print(f"   config.architecture_name == 'mlp_fusion': {mlp_test}")
    print(f"   config.architecture_name == 'dialog_rnn': {dialog_test}")
    print(f"   config.architecture_name == 'todkat_lite': {todkat_test}")
    
    # Character by character comparison for debug
    if config.architecture_name == "dialog_rnn":
        print(f"   ✅ MATCHED dialog_rnn")
    else:
        print(f"   ❌ Did NOT match dialog_rnn")
        print(f"   Expected: 'dialog_rnn' (len={len('dialog_rnn')})")
        print(f"   Got:      '{config.architecture_name}' (len={len(config.architecture_name)})")
        for i, (expected, actual) in enumerate(zip("dialog_rnn", config.architecture_name)):
            if expected != actual:
                print(f"   Mismatch at position {i}: expected '{expected}' got '{actual}'")
    
    print(f"\n🚧 ENTERING MODEL SELECTION LOGIC...")
    
    if config.architecture_name == "mlp_fusion":
        print("   🎯 BRANCH: Creating MultimodalFusionMLP")
        model = MultimodalFusionMLP(config)
        print(f"   ✅ Created MultimodalFusionMLP: {type(model)}")
    elif config.architecture_name == "todkat_lite":
        print("   🎯 BRANCH: Creating TodkatLiteMLP")
        model = TodkatLiteMLP(config)
        print(f"   ✅ Created TodkatLiteMLP: {type(model)}")
    elif config.architecture_name == "dialog_rnn":
        print("   🎯 BRANCH: Creating DialogRNNMLP")
        model = DialogRNNMLP(config)
        print(f"   ✅ Created DialogRNNMLP: {type(model)}")
    else:
        error_msg = f"Architecture {config.architecture_name} not implemented. Choose from: mlp_fusion, todkat_lite, dialog_rnn."
        print(f"   ❌ ERROR: {error_msg}")
        raise ValueError(error_msg)
    
    # Verify model after creation
    print(f"\n✅ MODEL VERIFICATION:")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Model class: {model.__class__.__name__}")
    print(f"   Model module: {model.__class__.__module__}")
    
    # Check model components
    components = []
    for name, module in model.named_modules():
        if '.' not in name and name != '':
            components.append(name)
    print(f"   Model components: {components}")
    
    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Architecture-specific verification
    if hasattr(model, 'gru_global'):
        print(f"   ✅ Has gru_global: {type(model.gru_global)}")
    if hasattr(model, 'fusion_mlp'):
        print(f"   ✅ Has fusion_mlp: {type(model.fusion_mlp)}")
    if hasattr(model, 'topic_emb'):
        print(f"   ✅ Has topic_emb: {type(model.topic_emb)}")
    
    # Set up logger
    print(f"\n📝 SETTING UP LOGGER...")
    logger = TensorBoardLogger(
        save_dir=config.logs_dir,
        name=config.experiment_name,
        version=config.run_name
    )
    print(f"   ✅ Logger created")
    
    # Set up callbacks
    print(f"\n🔄 SETTING UP CALLBACKS...")
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
    print(f"   ✅ Callbacks created")
    
    # Set up trainer
    print(f"\n🏃 SETTING UP TRAINER...")
    precision = "16-mixed" if config.use_mixed_precision else "32-true"
    print(f"   Precision: {precision}")
    
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        precision=precision,
        accumulate_grad_batches=config.grad_accumulation_steps,
        deterministic=True
    )
    print(f"   ✅ Trainer created")
    
    # Final check before training
    print(f"\n🎬 FINAL CHECK BEFORE TRAINING:")
    print(f"   Model being passed to trainer: {type(model).__name__}")
    print(f"   Data module being passed: {type(data_module).__name__}")
    print(f"   Config architecture_name: '{config.architecture_name}'")
    
    # Train model
    print(f"\n🏋️ STARTING TRAINING...")
    print("-" * 80)
    trainer.fit(model, data_module)
    print("-" * 80)
    print(f"✅ TRAINING COMPLETED")
    
    # Test model
    print(f"\n🧪 STARTING TESTING...")
    trainer.test(model, data_module)
    print(f"✅ TESTING COMPLETED")
    
    # Save configuration
    config_path = config.experiment_output_dir / "config.yaml"
    config.save_to_yaml(str(config_path))
    print(f"✅ CONFIG SAVED TO: {config_path}")
    
    # Return the best checkpoint path
    return checkpoint_callback.best_model_path

def main():
    """Main function for training emotion classification models."""
    print(f"\n🚀 STARTING MAIN FUNCTION")
    
    parser = argparse.ArgumentParser(description="Train emotion classification models")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--architecture", type=str, default="mlp_fusion", help="Model architecture")
    
    args = parser.parse_args()
    print(f"   Command line args: {args}")
    
    # Create configuration
    if args.config:
        print(f"   📋 Loading config from: {args.config}")
        config = Config.from_yaml(args.config)
        print(f"   ✅ Config loaded, architecture_name: '{config.architecture_name}'")
    else:
        print(f"   📋 Creating default config")
        config = Config()
        print(f"   ✅ Default config created, architecture_name: '{config.architecture_name}'")
    
    # Override with command-line arguments
    if args.architecture:
        print(f"   🔄 Overriding architecture with CLI arg: {args.architecture}")
        config.architecture_name = args.architecture
        print(f"   ✅ New architecture_name: '{config.architecture_name}'")
    
    # Run training
    print(f"\n🎯 CALLING TRAIN FUNCTION...")
    best_model_path = train(config)
    print(f"✅ TRAINING PIPELINE COMPLETED")
    print(f"📁 Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main() 