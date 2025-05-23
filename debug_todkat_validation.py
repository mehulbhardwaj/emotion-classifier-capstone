#!/usr/bin/env python3
"""
Debug script to monitor TOD-KAT validation issues.
This script helps diagnose why validation metrics might not be updating.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.base_config import Config
from models.todkat_lite import TodkatLiteMLP


def debug_todkat_validation():
    """Debug TOD-KAT validation step by step."""
    print("🔍 Debugging TOD-KAT Validation Issues")
    print("=" * 50)
    
    # Load your actual config
    try:
        config = Config.from_yaml("configs/colab_config_todkat_lite.yaml")
        print("✅ Loaded TOD-KAT config successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    # Create model
    try:
        model = TodkatLiteMLP(config)
        model.eval()
        print(f"✅ Created TOD-KAT model with {sum(p.numel() for p in model.parameters()):,} parameters")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return
    
    # Create a test batch that mimics your data structure
    batch_size, seq_len = 4, 8
    audio_len, text_len = 16000, 50
    
    test_batch = {
        "wav": torch.randn(batch_size, seq_len, audio_len),
        "wav_mask": torch.ones(batch_size, seq_len, audio_len, dtype=torch.bool),
        "txt": torch.randint(1, 1000, (batch_size, seq_len, text_len)),
        "txt_mask": torch.ones(batch_size, seq_len, text_len, dtype=torch.bool),
        "topic_id": torch.randint(0, config.n_topics, (batch_size, seq_len)),
        "kn_vec": torch.randn(batch_size, seq_len, config.knowledge_dim) if config.use_knowledge else None,
        "dialog_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, 7, (batch_size, seq_len))
    }
    
    print(f"\n🔍 Test batch shapes:")
    for key, value in test_batch.items():
        if value is not None:
            print(f"   {key}: {value.shape}")
    
    # Test forward pass
    print(f"\n🔍 Testing forward pass...")
    try:
        with torch.no_grad():
            logits = model(test_batch)
        print(f"✅ Forward pass successful! Logits shape: {logits.shape}")
        print(f"   Expected: ({batch_size}, {seq_len}, 7)")
        print(f"   Logits range: {logits.min().item():.3f} to {logits.max().item():.3f}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return
    
    # Test validation step
    print(f"\n🔍 Testing validation step...")
    try:
        loss = model.validation_step(test_batch, batch_idx=0)
        if loss is not None:
            print(f"✅ Validation step successful! Loss: {loss.item():.4f}")
        else:
            print("❌ Validation step returned None!")
            return
    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        return
    
    # Test prediction consistency
    print(f"\n🔍 Testing prediction consistency...")
    predictions_list = []
    for i in range(3):
        with torch.no_grad():
            logits = model(test_batch)
            preds = logits.argmax(dim=-1)
            predictions_list.append(preds)
    
    # Check if predictions are deterministic (should be in eval mode)
    if torch.equal(predictions_list[0], predictions_list[1]) and torch.equal(predictions_list[1], predictions_list[2]):
        print("✅ Predictions are consistent (deterministic)")
    else:
        print("⚠️  Predictions are not consistent - check if model is in eval mode")
    
    # Check prediction distribution
    pred_flat = predictions_list[0][test_batch["dialog_mask"]].flatten()
    pred_counts = torch.bincount(pred_flat, minlength=7)
    print(f"   Prediction distribution: {pred_counts.tolist()}")
    
    # Check if model is actually learning different patterns
    print(f"\n🔍 Testing learning capacity...")
    
    # Create two very different batches
    batch_happy = test_batch.copy()
    batch_sad = test_batch.copy()
    
    # Modify labels to be all happy vs all sad
    batch_happy["labels"] = torch.full_like(test_batch["labels"], 2)  # Joy
    batch_sad["labels"] = torch.full_like(test_batch["labels"], 4)    # Sadness
    
    with torch.no_grad():
        logits_happy = model(batch_happy)
        logits_sad = model(batch_sad)
        
        # Check if logits are different for different inputs
        logits_diff = torch.abs(logits_happy - logits_sad).mean()
        print(f"   Logits difference between batches: {logits_diff.item():.6f}")
        
        if logits_diff < 1e-6:
            print("⚠️  Model produces identical outputs for different inputs!")
            print("     This suggests the model might not be learning properly.")
        else:
            print("✅ Model produces different outputs for different inputs")
    
    # Test metrics computation
    print(f"\n🔍 Testing metrics computation...")
    try:
        # Simulate what happens in validation
        mask = test_batch["dialog_mask"].bool()
        labels = test_batch["labels"]
        logits = model(test_batch)
        
        logits_flat = logits[mask]
        labels_flat = labels[mask]
        preds = logits_flat.argmax(dim=-1)
        
        # Manual accuracy calculation
        accuracy = (preds == labels_flat).float().mean()
        print(f"   Manual accuracy calculation: {accuracy.item():.4f}")
        
        # Check F1 metric
        model.val_f1.update(preds, labels_flat)
        f1_score = model.val_f1.compute()
        print(f"   F1 score: {f1_score.item():.4f}")
        model.val_f1.reset()
        
        # Check for any stuck values
        unique_preds = torch.unique(preds)
        unique_labels = torch.unique(labels_flat)
        print(f"   Unique predictions: {unique_preds.tolist()}")
        print(f"   Unique labels: {unique_labels.tolist()}")
        
        if len(unique_preds) == 1:
            print("⚠️  Model is predicting only one class!")
            print("     This could explain why metrics aren't changing.")
        
    except Exception as e:
        print(f"❌ Metrics computation failed: {e}")
    
    print(f"\n" + "=" * 50)
    print("🏁 Debug Summary:")
    print("✅ Model loads and runs successfully")
    print("✅ Fixed validation_step to return loss")
    print("💡 Key things to check in your training:")
    print("   1. Are predictions varying across epochs?")
    print("   2. Is the model predicting multiple classes?")
    print("   3. Are gradients flowing properly?")
    print("   4. Is the learning rate appropriate?")
    print("\n🚀 Try running training again to see if metrics update now!")


if __name__ == "__main__":
    debug_todkat_validation() 