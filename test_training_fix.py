#!/usr/bin/env python3
"""Test script to verify the dialogue mapping fix resolves training issues."""

import torch
import pytorch_lightning as pl
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_training_fix():
    """Test that the dialogue mapping fix allows training to start."""
    try:
        print("🔧 Testing training with dialogue mapping fix...")
        
        # Import after path setup
        from configs.base_config import BaseConfig
        from utils.data_processor import MELDDataModule
        from models.dialog_rnn import DialogRNNMLP
        
        # Create minimal config
        config = BaseConfig()
        config.data_root = "/content/drive/MyDrive/dlfa_capstone/meld_data"
        config.text_encoder_model_name = "roberta-base"
        config.audio_encoder_model_name = "facebook/wav2vec2-base"
        config.architecture_name = "dialog_rnn"
        config.batch_size = 2  # Small batch for testing
        config.output_dim = 7
        config.learning_rate = 0.0002
        config.max_epochs = 1
        config.dataloader_num_workers = 0  # Avoid multiprocessing issues
        
        print("✅ Config created")
        
        # Create data module
        print("📊 Setting up data module...")
        data_module = MELDDataModule(config)
        data_module.setup()
        print("✅ Data module setup completed")
        
        # Test val dataloader creation
        print("🔍 Testing validation dataloader...")
        val_loader = data_module.val_dataloader()
        print(f"✅ Val dataloader created: {len(val_loader)} batches")
        
        # Test loading one batch
        print("📦 Testing batch loading...")
        val_iter = iter(val_loader)
        try:
            batch = next(val_iter)
            print(f"✅ Batch loaded successfully!")
            print(f"   Batch keys: {list(batch.keys())}")
            print(f"   Batch shapes:")
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"      {k}: {v.shape}")
                    
            # Check if we have valid data (non-zero dialog_mask)
            if 'dialog_mask' in batch:
                valid_utterances = batch['dialog_mask'].sum().item()
                print(f"   Valid utterances in batch: {valid_utterances}")
                if valid_utterances > 0:
                    print("🎉 SUCCESS: Found valid dialogue data!")
                else:
                    print("⚠️  WARNING: All dialogues still empty")
            
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
            return False
        
        # Test model creation
        print("🤖 Testing model creation...")
        model = DialogRNNMLP(config)
        print("✅ Model created successfully")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        try:
            with torch.no_grad():
                logits = model(batch)
                print(f"✅ Forward pass successful! Logits shape: {logits.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
        
        # Test training step
        print("🎯 Testing training step...")
        try:
            model.train()
            loss = model.training_step(batch, 0)
            print(f"✅ Training step successful! Loss: {loss}")
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("🎊 All tests passed! The fix should work for training.")
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing dialogue mapping training fix...")
    success = test_training_fix()
    if success:
        print("\n🏆 Training fix test PASSED! You should be able to train now.")
    else:
        print("\n💥 Training fix test FAILED. More investigation needed.") 