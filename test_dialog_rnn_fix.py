#!/usr/bin/env python3
"""Test the DialogRNN fix for context window shape mismatch."""

import torch
from configs import Config
from models.dialog_rnn import DialogRNNMLP


def test_dialog_rnn_context_window():
    """Test DialogRNN with context window to ensure no shape mismatch."""
    
    print("🧪 Testing DialogRNN context window fix...")
    
    # Load config
    config = Config.from_yaml("configs/colab_config_dialog_rnn.yaml")
    print(f"📋 Context window: {config.context_window}")
    
    # Create model
    model = DialogRNNMLP(config)
    model.eval()
    
    # Create dummy batch with sequence length > context_window (16 > 10)
    batch_size = 2
    seq_len = 16  # Longer than context_window=10
    audio_len = 1000
    text_len = 20
    
    # Dummy batch data
    batch = {
        "wav": torch.randn(batch_size, seq_len, audio_len),
        "wav_mask": torch.ones(batch_size, seq_len, audio_len, dtype=torch.bool),
        "txt": torch.randint(0, 1000, (batch_size, seq_len, text_len)),
        "txt_mask": torch.ones(batch_size, seq_len, text_len, dtype=torch.bool),
        "speaker_id": torch.randint(0, 5, (batch_size, seq_len)),
        "dialog_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, 7, (batch_size, seq_len)),
    }
    
    print(f"🔍 Input shapes:")
    print(f"   wav: {batch['wav'].shape}")
    print(f"   dialog_mask: {batch['dialog_mask'].shape}")
    print(f"   labels: {batch['labels'].shape}")
    
    try:
        # Test forward pass
        logits = model(batch)
        print(f"✅ Forward pass successful!")
        print(f"   logits shape: {logits.shape}")
        
        # Test training step (this was where the error occurred)
        loss = model._shared_step(batch, "val")
        print(f"✅ Validation step successful!")
        print(f"   loss: {loss.item():.4f}")
        
        # Verify shapes are consistent
        mask = batch["dialog_mask"].bool()
        labels = batch["labels"]
        
        # Apply same context window logic as in model
        if model.context_window > 0 and mask.shape[1] > model.context_window and logits.shape[1] == model.context_window:
            mask = mask[:, -model.context_window:]
            labels = labels[:, -model.context_window:]
        
        logits_flat = logits[mask]
        labels_flat = labels[mask]
        
        print(f"🔍 Final shapes:")
        print(f"   logits_flat: {logits_flat.shape}")
        print(f"   labels_flat: {labels_flat.shape}")
        
        assert logits_flat.shape[0] == labels_flat.shape[0], "Shape mismatch!"
        print(f"✅ Shape consistency verified!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dialog_rnn_context_window()
    if success:
        print(f"\n🎉 DialogRNN context window fix successful!")
    else:
        print(f"\n💥 DialogRNN fix failed!") 