#!/usr/bin/env python3
"""
Debug validation-specific issues in TOD-KAT.
Focus on why validation metrics are frozen while training works.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.base_config import Config
from models.todkat_lite import TodkatLiteMLP


def debug_validation_specific():
    """Debug validation-specific issues."""
    print("🔍 Debugging Validation-Specific Issues")
    print("=" * 50)
    
    # Load config
    config = Config.from_yaml("configs/colab_config_todkat_lite.yaml")
    model = TodkatLiteMLP(config)
    
    # Create consistent test batch
    batch_size, seq_len = 2, 4
    test_batch = {
        "wav": torch.randn(batch_size, seq_len, 16000),
        "wav_mask": torch.ones(batch_size, seq_len, 16000, dtype=torch.bool),
        "txt": torch.randint(1, 1000, (batch_size, seq_len, 50)),
        "txt_mask": torch.ones(batch_size, seq_len, 50, dtype=torch.bool),
        "topic_id": torch.randint(0, config.n_topics, (batch_size, seq_len)),
        "kn_vec": torch.randn(batch_size, seq_len, config.knowledge_dim),
        "dialog_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, 7, (batch_size, seq_len))
    }
    
    print("🔍 Testing Model Mode Effects:")
    
    # Test in training mode
    model.train()
    train_logits = model(test_batch)
    train_loss = model.training_step(test_batch, 0)
    
    # Test in eval mode  
    model.eval()
    val_logits = model(test_batch)
    val_loss = model.validation_step(test_batch, 0)
    
    print(f"   Training mode - Loss: {train_loss.item():.6f}")
    print(f"   Validation mode - Loss: {val_loss.item():.6f}")
    
    # Check if outputs differ between modes
    logits_diff = torch.abs(train_logits - val_logits).mean()
    print(f"   Logits difference (train vs val mode): {logits_diff.item():.8f}")
    
    if logits_diff > 1e-6:
        print("   ✅ Model behaves differently in train vs eval mode")
    else:
        print("   ⚠️  Model outputs identical in both modes")
    
    print(f"\n🔍 Testing Validation Data Consistency:")
    
    # Test if validation gives same results for same input
    model.eval()
    val_results = []
    for i in range(3):
        loss = model.validation_step(test_batch, 0)
        val_results.append(loss.item())
    
    print(f"   Validation losses: {val_results}")
    if len(set([round(x, 6) for x in val_results])) == 1:
        print("   ✅ Validation is deterministic (same input = same output)")
    else:
        print("   ⚠️  Validation is non-deterministic")
    
    print(f"\n🔍 Testing Metrics Computation:")
    
    # Manual metrics computation to check if metrics are correct
    model.eval()
    with torch.no_grad():
        logits = model(test_batch)
        
    mask = test_batch["dialog_mask"].bool()
    labels = test_batch["labels"]
    
    # Flatten valid utterances
    logits_flat = logits[mask]  # (N_valid, C)
    labels_flat = labels[mask]  # (N_valid,)
    preds = logits_flat.argmax(dim=-1)
    
    # Manual calculations
    manual_acc = (preds == labels_flat).float().mean()
    
    # Using model's F1 metric
    model.val_f1.reset()  # Important: reset first
    model.val_f1.update(preds, labels_flat)
    model_f1 = model.val_f1.compute()
    
    print(f"   Manual accuracy: {manual_acc.item():.6f}")
    print(f"   Model F1 score: {model_f1.item():.6f}")
    print(f"   Predictions: {preds.tolist()}")
    print(f"   Labels: {labels_flat.tolist()}")
    
    # Check if predictions are diverse
    unique_preds = torch.unique(preds)
    unique_labels = torch.unique(labels_flat)
    print(f"   Unique predictions: {len(unique_preds)} classes: {unique_preds.tolist()}")
    print(f"   Unique labels: {len(unique_labels)} classes: {unique_labels.tolist()}")
    
    print(f"\n🔍 Testing Batch Processing:")
    
    # Test if validation step processes batches correctly
    model.eval()
    
    # Create two different batches
    batch1 = test_batch.copy()
    batch2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in test_batch.items()}
    
    # Make batch2 have different labels
    batch2["labels"] = torch.randint(0, 7, batch2["labels"].shape)
    
    loss1 = model.validation_step(batch1, 0)
    loss2 = model.validation_step(batch2, 0)
    
    loss_diff = abs(loss1.item() - loss2.item())
    print(f"   Loss for batch1: {loss1.item():.6f}")
    print(f"   Loss for batch2: {loss2.item():.6f}")
    print(f"   Loss difference: {loss_diff:.6f}")
    
    if loss_diff > 1e-6:
        print("   ✅ Validation step responds to different inputs")
    else:
        print("   ❌ Validation step produces identical results!")
        print("       This suggests the validation data might be identical every epoch")
    
    print(f"\n🔍 Testing PyTorch Lightning Integration:")
    
    # Check if model.log is working correctly during validation
    print("   Model has trainer:", hasattr(model, 'trainer') and model.trainer is not None)
    print("   Model current_epoch:", getattr(model, 'current_epoch', 'Not available'))
    
    # Check validation_step return value
    val_return = model.validation_step(test_batch, 0)
    print(f"   validation_step returns: {type(val_return)} with value {val_return.item():.6f}")
    
    if val_return is None:
        print("   ❌ validation_step returns None - this will break PyTorch Lightning!")
    else:
        print("   ✅ validation_step returns proper loss tensor")
    
    print(f"\n" + "=" * 50)
    print("🏁 Validation Debug Summary:")
    print("💡 Most likely causes of frozen validation metrics:")
    print("   1. Validation dataset is identical every epoch")
    print("   2. Model not switching modes properly")
    print("   3. Metrics not being reset between epochs")
    print("   4. PyTorch Lightning logging issue")
    print("   5. Data loader returning same validation data")
    
    print(f"\n🚀 Next steps:")
    print("   1. Check your validation data loader")
    print("   2. Add debug prints in actual training script")
    print("   3. Verify model.eval() is being called")
    print("   4. Check if validation dataset shuffle=False")


if __name__ == "__main__":
    debug_validation_specific() 