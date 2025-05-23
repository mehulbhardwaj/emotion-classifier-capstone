#!/usr/bin/env python3
"""NUCLEAR DEBUG TRAINING SCRIPT - TRACES EVERYTHING"""

import sys
import os
import argparse
from pathlib import Path

print("🚀 STARTING NUCLEAR DEBUG TRAINING")
print("=" * 80)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
print(f"Added to path: {current_dir}")

# Import debug modules with explicit prints
print("\n📦 IMPORTING DEBUG MODULES...")

print("   Importing torch...")
import torch
print(f"   ✅ PyTorch version: {torch.__version__}")

print("   Importing pytorch_lightning...")
import pytorch_lightning as pl
print(f"   ✅ PyTorch Lightning version: {pl.__version__}")

print("   Importing debug config...")
from configs.base_config_debug import Config
print("   ✅ Debug Config imported")

print("   Importing debug data processor...")
from utils.data_processor_debug import MELDDataModule
print("   ✅ Debug Data Processor imported")

print("   Importing debug models...")
from models.mlp_fusion import MultimodalFusionMLP
print("   ✅ MLP Fusion imported")

from models.dialog_rnn_debug import DialogRNNMLP
print("   ✅ Debug DialogRNN imported")

from models.todkat_lite import TodkatLiteMLP  
print("   ✅ TOD-KAT Lite imported")

print("   Importing PyTorch Lightning components...")
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
print("   ✅ Lightning components imported")

print("✅ ALL IMPORTS COMPLETED")

def nuclear_train(config_path):
    """Train with nuclear-level debugging."""
    print(f"\n💥 NUCLEAR TRAINING STARTED")
    print(f"   Config path: {config_path}")
    
    # 1. Load config with nuclear debugging
    print(f"\n📋 STEP 1: LOADING CONFIG WITH NUCLEAR DEBUGGING")
    print("-" * 60)
    config = Config.from_yaml(config_path)
    print("-" * 60)
    print(f"✅ CONFIG LOADED")
    print(f"   Config type: {type(config)}")
    print(f"   Config repr: {repr(config)}")
    print(f"   Config architecture_name: '{config.architecture_name}'")
    
    # 2. Create data module with nuclear debugging
    print(f"\n📊 STEP 2: CREATING DATA MODULE WITH NUCLEAR DEBUGGING")
    print("-" * 60)
    data_module = MELDDataModule(config)
    print("-" * 60)
    print(f"✅ DATA MODULE CREATED")
    print(f"   Data module type: {type(data_module)}")
    
    # 3. Model selection with nuclear debugging
    print(f"\n🏗️ STEP 3: MODEL SELECTION WITH NUCLEAR DEBUGGING")
    print("-" * 60)
    print(f"🔍 DETAILED MODEL SELECTION ANALYSIS:")
    print(f"   config.architecture_name: '{config.architecture_name}'")
    print(f"   config.architecture_name type: {type(config.architecture_name)}")
    print(f"   config.architecture_name repr: {repr(config.architecture_name)}")
    print(f"   config.architecture_name length: {len(config.architecture_name)}")
    
    # Test exact string matches
    mlp_match = config.architecture_name == "mlp_fusion"
    dialog_match = config.architecture_name == "dialog_rnn" 
    todkat_match = config.architecture_name == "todkat_lite"
    
    print(f"   STRING MATCH TESTS:")
    print(f"      config.architecture_name == 'mlp_fusion': {mlp_match}")
    print(f"      config.architecture_name == 'dialog_rnn': {dialog_match}")
    print(f"      config.architecture_name == 'todkat_lite': {todkat_match}")
    
    # Character-by-character analysis
    expected = "dialog_rnn"
    actual = config.architecture_name
    print(f"   CHARACTER-BY-CHARACTER ANALYSIS:")
    print(f"      Expected: '{expected}' (len={len(expected)})")
    print(f"      Actual:   '{actual}' (len={len(actual)})")
    
    if len(actual) >= len(expected):
        for i in range(len(expected)):
            exp_char = expected[i] if i < len(expected) else "?"
            act_char = actual[i] if i < len(actual) else "?"
            match = exp_char == act_char
            print(f"         {i:2}: '{exp_char}' vs '{act_char}' -> {match}")
    
    # Show control/invisible characters
    print(f"   BYTE ANALYSIS:")
    print(f"      Expected bytes: {expected.encode('utf-8')}")
    print(f"      Actual bytes:   {actual.encode('utf-8')}")
    
    print(f"\n🚧 ENTERING MODEL CREATION LOGIC...")
    
    # Model creation with explicit debugging
    if config.architecture_name == "mlp_fusion":
        print(f"   🎯 BRANCH TAKEN: mlp_fusion")
        print(f"   Creating MultimodalFusionMLP...")
        model = MultimodalFusionMLP(config)
        print(f"   ✅ Created: {type(model).__name__}")
        
    elif config.architecture_name == "dialog_rnn":
        print(f"   🎯 BRANCH TAKEN: dialog_rnn")
        print(f"   Creating DialogRNNMLP (DEBUG VERSION)...")
        model = DialogRNNMLP(config)
        print(f"   ✅ Created: {type(model).__name__}")
        
    elif config.architecture_name == "todkat_lite":
        print(f"   🎯 BRANCH TAKEN: todkat_lite")
        print(f"   Creating TodkatLiteMLP...")
        model = TodkatLiteMLP(config)
        print(f"   ✅ Created: {type(model).__name__}")
        
    else:
        print(f"   ❌ NO BRANCH TAKEN!")
        print(f"   Unknown architecture: '{config.architecture_name}'")
        raise ValueError(f"Unknown architecture: {config.architecture_name}")
    
    print("-" * 60)
    print(f"✅ MODEL CREATED")
    
    # 4. Model verification with nuclear debugging
    print(f"\n✅ STEP 4: MODEL VERIFICATION WITH NUCLEAR DEBUGGING")
    print("-" * 60)
    print(f"   Model class: {model.__class__.__name__}")
    print(f"   Model module: {model.__class__.__module__}")
    print(f"   Model type: {type(model)}")
    print(f"   Model repr: {repr(model)}")
    
    # Check for architecture-specific components
    print(f"   COMPONENT CHECK:")
    if hasattr(model, 'gru_global'):
        print(f"      ✅ Has gru_global: {type(model.gru_global)}")
    else:
        print(f"      ❌ No gru_global")
        
    if hasattr(model, 'fusion_mlp'):
        print(f"      ✅ Has fusion_mlp: {type(model.fusion_mlp)}")
    else:
        print(f"      ❌ No fusion_mlp")
        
    if hasattr(model, 'topic_emb'):
        print(f"      ✅ Has topic_emb: {type(model.topic_emb)}")
    else:
        print(f"      ❌ No topic_emb")
    
    # Parameter counting
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   PARAMETER COUNT:")
    print(f"      Total parameters: {total_params:,}")
    print(f"      Trainable parameters: {trainable_params:,}")
    
    # Expected parameter counts for verification
    expected_params = {
        "mlp_fusion": (427271, "MultimodalFusionMLP"),
        "dialog_rnn": (5356295, "DialogRNNMLP"), 
        "todkat_lite": (23968279, "TodkatLiteMLP")
    }
    
    if config.architecture_name in expected_params:
        expected_count, expected_class = expected_params[config.architecture_name]
        print(f"   PARAMETER VERIFICATION:")
        print(f"      Expected: {expected_count:,} for {expected_class}")
        print(f"      Actual:   {trainable_params:,} for {type(model).__name__}")
        if trainable_params == expected_count:
            print(f"      ✅ PARAMETER COUNT MATCHES!")
        else:
            print(f"      ❌ PARAMETER COUNT MISMATCH!")
    
    print("-" * 60)
    print(f"✅ MODEL VERIFICATION COMPLETED")
    
    # 5. Setup trainer with nuclear debugging
    print(f"\n🏃 STEP 5: SETTING UP TRAINER WITH NUCLEAR DEBUGGING")
    print("-" * 60)
    
    print(f"   Creating logger...")
    logger = TensorBoardLogger(
        save_dir=config.logs_dir,
        name=config.experiment_name,
        version=config.run_name
    )
    print(f"   ✅ Logger created: {type(logger)}")
    
    print(f"   Creating callbacks...")
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
    
    print(f"   Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=2,  # Just 2 epochs for nuclear debugging
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        precision="32-true",
        accumulate_grad_batches=config.grad_accumulation_steps,
        deterministic=True,
        enable_model_summary=True,  # Make sure model summary is enabled
    )
    print(f"   ✅ Trainer created: {type(trainer)}")
    
    print("-" * 60)
    print(f"✅ TRAINER SETUP COMPLETED")
    
    # 6. Final checks before training
    print(f"\n🎬 STEP 6: FINAL CHECKS BEFORE TRAINING")
    print("-" * 60)
    print(f"   Model being passed to trainer:")
    print(f"      Type: {type(model).__name__}")
    print(f"      Class: {model.__class__.__name__}")
    print(f"      Module: {model.__class__.__module__}")
    print(f"   Data module being passed:")
    print(f"      Type: {type(data_module).__name__}")
    print(f"   Config architecture_name: '{config.architecture_name}'")
    
    print("-" * 60)
    print(f"✅ FINAL CHECKS COMPLETED")
    
    # 7. Start training with nuclear debugging
    print(f"\n🏋️ STEP 7: STARTING TRAINING WITH NUCLEAR DEBUGGING")
    print("=" * 80)
    print(f"🚨 WATCH THE PYTORCH LIGHTNING OUTPUT CAREFULLY!")
    print(f"🚨 ESPECIALLY THE MODEL SUMMARY TABLE!")
    print("=" * 80)
    
    try:
        trainer.fit(model, data_module)
        print("=" * 80)
        print(f"✅ NUCLEAR TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print("=" * 80)
        print(f"❌ NUCLEAR TRAINING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 8. Post-training analysis
    print(f"\n📊 STEP 8: POST-TRAINING ANALYSIS")
    print("-" * 60)
    print(f"   Model after training:")
    print(f"      Type: {type(model).__name__}")
    print(f"      Class: {model.__class__.__name__}")
    print(f"   Architecture that was supposedly trained: '{config.architecture_name}'")
    print("-" * 60)
    print(f"✅ POST-TRAINING ANALYSIS COMPLETED")
    
    return trainer.checkpoint_callback.best_model_path

def main():
    """Main function for nuclear debugging."""
    print(f"\n🚀 MAIN FUNCTION STARTING")
    
    parser = argparse.ArgumentParser(description="Nuclear debug emotion classification training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    
    args = parser.parse_args()
    print(f"   Command line args: {args}")
    print(f"   Config file: {args.config}")
    
    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        return
    
    print(f"   ✅ Config file exists: {config_path}")
    
    # Run nuclear training
    print(f"\n🎯 LAUNCHING NUCLEAR TRAINING...")
    best_model_path = nuclear_train(str(config_path))
    
    if best_model_path:
        print(f"\n🎉 NUCLEAR TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"📁 Best model saved at: {best_model_path}")
    else:
        print(f"\n💥 NUCLEAR TRAINING FAILED!")

if __name__ == "__main__":
    main() 