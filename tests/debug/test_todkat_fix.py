#!/usr/bin/env python3
"""Test the TOD-KAT fix for invalid labels."""

import torch
from configs import Config
from models.todkat_lite import TodkatLiteMLP


def test_todkat_labels():
    """Test TOD-KAT with various label scenarios."""
    
    print("🧪 Testing TOD-KAT label handling...")
    
    # Load config
    config = Config.from_yaml("configs/colab_config_todkat_lite.yaml")
    
    # Create model
    model = TodkatLiteMLP(config)
    model.eval()
    
    # Test scenario 1: Valid labels
    print("\n1️⃣ Testing with valid labels...")
    batch_valid = create_test_batch(batch_size=4, seq_len=8, valid_labels=True)
    print(f"   Labels (last position): {batch_valid['labels'][:, -1]}")
    
    try:
        loss = model._shared_step(batch_valid, "test")
        print(f"   ✅ Valid labels test passed! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Valid labels test failed: {e}")
    
    # Test scenario 2: Invalid labels (negative)
    print("\n2️⃣ Testing with invalid labels (negative)...")
    batch_invalid = create_test_batch(batch_size=4, seq_len=8, valid_labels=False)
    batch_invalid['labels'][:, -1] = -1  # Set last labels to invalid
    print(f"   Labels (last position): {batch_invalid['labels'][:, -1]}")
    
    try:
        loss = model._shared_step(batch_invalid, "test")
        print(f"   ✅ Invalid labels handled! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Invalid labels test failed: {e}")
    
    # Test scenario 3: Mixed valid/invalid labels
    print("\n3️⃣ Testing with mixed valid/invalid labels...")
    batch_mixed = create_test_batch(batch_size=4, seq_len=8, valid_labels=True)
    batch_mixed['labels'][0, -1] = -1  # Make first sample invalid
    batch_mixed['labels'][2, -1] = 10  # Make third sample invalid (>= num_classes)
    print(f"   Labels (last position): {batch_mixed['labels'][:, -1]}")
    
    try:
        loss = model._shared_step(batch_mixed, "test")
        print(f"   ✅ Mixed labels handled! Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Mixed labels test failed: {e}")


def create_test_batch(batch_size, seq_len, valid_labels=True):
    """Create a test batch for TOD-KAT."""
    
    audio_len = 1000
    text_len = 20
    
    # Create dummy topic_id (required by TOD-KAT)
    topic_id = torch.randint(0, 50, (batch_size, seq_len))  # n_topics=50
    
    batch = {
        "wav": torch.randn(batch_size, seq_len, audio_len),
        "wav_mask": torch.ones(batch_size, seq_len, audio_len, dtype=torch.bool),
        "txt": torch.randint(0, 1000, (batch_size, seq_len, text_len)),
        "txt_mask": torch.ones(batch_size, seq_len, text_len, dtype=torch.bool),
        "topic_id": topic_id,
        "dialog_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, 7 if valid_labels else 10, (batch_size, seq_len)),
    }
    
    return batch


if __name__ == "__main__":
    test_todkat_labels() 