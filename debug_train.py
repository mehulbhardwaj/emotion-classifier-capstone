#!/usr/bin/env python3
"""Debug version of train.py with explicit model checking."""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pathlib import Path
from configs import Config
from utils.data_processor import MELDDataModule

# Import model architectures
from models.mlp_fusion import MultimodalFusionMLP
from models.dialog_rnn import DialogRNNMLP
from models.todkat_lite import TodkatLiteMLP


def debug_train(config_path):
    """Debug training with explicit model checking."""
    print(f"🔍 DEBUG TRAINING WITH CONFIG: {config_path}")
    print("=" * 60)
    
    # 1. Load and check config
    print("\n📋 STEP 1: Loading config...")
    config = Config.from_yaml(config_path)
    print(f"   Config architecture_name: '{config.architecture_name}'")
    print(f"   Config type: {type(config)}")
    
    # 2. Create model with explicit debugging
    print("\n🏗️  STEP 2: Creating model...")
    print(f"   Checking architecture_name: '{config.architecture_name}'")
    print(f"   architecture_name == 'mlp_fusion': {config.architecture_name == 'mlp_fusion'}")
    print(f"   architecture_name == 'dialog_rnn': {config.architecture_name == 'dialog_rnn'}")
    print(f"   architecture_name == 'todkat_lite': {config.architecture_name == 'todkat_lite'}")
    
    if config.architecture_name == "mlp_fusion":
        print("   → Creating MultimodalFusionMLP")
        model = MultimodalFusionMLP(config)
    elif config.architecture_name == "todkat_lite":
        print("   → Creating TodkatLiteMLP")
        model = TodkatLiteMLP(config)
    elif config.architecture_name == "dialog_rnn":
        print("   → Creating DialogRNNMLP")
        model = DialogRNNMLP(config)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture_name}")
    
    # 3. Verify model after creation
    print(f"\n✅ STEP 3: Model verification...")
    print(f"   Created model type: {type(model).__name__}")
    print(f"   Model class: {model.__class__.__name__}")
    
    # Check model components
    components = [name for name, _ in model.named_modules() if '.' not in name and name != '']
    print(f"   Model components: {components}")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 4. Check specific components for DialogRNN
    if config.architecture_name == "dialog_rnn":
        print(f"\n🔍 DialogRNN specific checks:")
        print(f"   Has gru_global: {hasattr(model, 'gru_global')}")
        print(f"   Has gru_speaker: {hasattr(model, 'gru_speaker')}")
        print(f"   Has gru_emotion: {hasattr(model, 'gru_emotion')}")
        print(f"   Has fusion_mlp: {hasattr(model, 'fusion_mlp')}")
        
        if hasattr(model, 'gru_global'):
            gru_params = sum(p.numel() for p in model.gru_global.parameters())
            print(f"   GRU global parameters: {gru_params:,}")
    
    # 5. Create data module
    print(f"\n📊 STEP 4: Creating data module...")
    data_module = MELDDataModule(config)
    
    # 6. Create trainer and check what it sees
    print(f"\n🚀 STEP 5: Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=1,  # Just 1 epoch for debugging
        accelerator="auto",
        devices=1,
        logger=False,  # No logging for debug
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 7. Print what trainer will see
    print(f"\n🎭 STEP 6: What trainer sees...")
    print(f"   Model passed to trainer: {type(model).__name__}")
    
    # 8. Start training and see the Lightning summary
    print(f"\n🏃 STEP 7: Starting training (1 epoch)...")
    print("   Watch the PyTorch Lightning model summary below:")
    print("-" * 60)
    
    try:
        trainer.fit(model, data_module)
        print(f"\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    debug_train(args.config) 