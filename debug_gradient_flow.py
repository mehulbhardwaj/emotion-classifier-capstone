#!/usr/bin/env python3
"""
Debug gradient flow in TOD-KAT to understand why outputs are identical.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.base_config import Config
from models.todkat_lite import TodkatLiteMLP


def debug_gradient_flow():
    """Debug gradient flow in TOD-KAT."""
    print("🔍 Debugging TOD-KAT Gradient Flow")
    print("=" * 50)
    
    # Load config
    config = Config.from_yaml("configs/colab_config_todkat_lite.yaml")
    model = TodkatLiteMLP(config)
    model.train()  # Set to training mode
    
    # Create test batch
    batch_size, seq_len = 2, 4
    test_batch = {
        "wav": torch.randn(batch_size, seq_len, 16000, requires_grad=True),
        "wav_mask": torch.ones(batch_size, seq_len, 16000, dtype=torch.bool),
        "txt": torch.randint(1, 1000, (batch_size, seq_len, 50)),
        "txt_mask": torch.ones(batch_size, seq_len, 50, dtype=torch.bool),
        "topic_id": torch.randint(0, config.n_topics, (batch_size, seq_len)),
        "kn_vec": torch.randn(batch_size, seq_len, config.knowledge_dim),
        "dialog_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "labels": torch.randint(0, 7, (batch_size, seq_len))
    }
    
    print("🔍 Parameter Analysis:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"   ✅ {name}: {param.numel():,} trainable")
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    # Test forward pass and backward pass
    print(f"\n🔍 Testing Gradient Flow:")
    
    # Store initial parameters
    initial_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_params[name] = param.clone().detach()
    
    # Forward pass
    loss = model.training_step(test_batch, 0)
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm
            if grad_norm > 1e-6:
                print(f"   ✅ {name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"   ⚠️  {name}: grad_norm = {grad_norm:.6f} (very small!)")
    
    # Test with different inputs
    print(f"\n🔍 Testing Input Sensitivity:")
    
    # Create two very different batches
    batch1 = test_batch.copy()
    batch2 = test_batch.copy()
    
    # Make batch2 completely different
    batch2["wav"] = torch.randn_like(batch1["wav"]) * 10  # Much larger audio
    batch2["txt"] = torch.randint(1, 1000, batch1["txt"].shape)  # Different text
    batch2["topic_id"] = torch.randint(0, config.n_topics, batch1["topic_id"].shape)
    batch2["kn_vec"] = torch.randn_like(batch1["kn_vec"]) * 5  # Different knowledge
    
    # Test outputs
    model.eval()
    with torch.no_grad():
        logits1 = model(batch1)
        logits2 = model(batch2)
    
    diff = torch.abs(logits1 - logits2).mean()
    print(f"   Logits difference: {diff.item():.8f}")
    
    if diff < 1e-6:
        print("   ❌ Outputs are identical - model not sensitive to inputs!")
        
        # Check if encoders are actually frozen
        print(f"\n🔍 Checking Encoder Status:")
        audio_trainable = sum(p.requires_grad for p in model.audio_encoder.parameters())
        text_trainable = sum(p.requires_grad for p in model.text_encoder.parameters())
        print(f"   Audio encoder trainable params: {audio_trainable}")
        print(f"   Text encoder trainable params: {text_trainable}")
        
        # Check encoder outputs
        with torch.no_grad():
            # Audio encoder
            a_emb1 = model.audio_encoder(
                input_values=batch1["wav"].flatten(0,1), 
                attention_mask=batch1["wav_mask"].flatten(0,1)
            ).last_hidden_state[:,0,:]
            a_emb2 = model.audio_encoder(
                input_values=batch2["wav"].flatten(0,1), 
                attention_mask=batch2["wav_mask"].flatten(0,1)
            ).last_hidden_state[:,0,:]
            
            audio_diff = torch.abs(a_emb1 - a_emb2).mean()
            print(f"   Audio encoder output diff: {audio_diff.item():.8f}")
            
            # Text encoder  
            t_emb1 = model.text_encoder(
                input_ids=batch1["txt"].flatten(0,1), 
                attention_mask=batch1["txt_mask"].flatten(0,1)
            ).last_hidden_state[:,0,:]
            t_emb2 = model.text_encoder(
                input_ids=batch2["txt"].flatten(0,1), 
                attention_mask=batch2["txt_mask"].flatten(0,1)
            ).last_hidden_state[:,0,:]
            
            text_diff = torch.abs(t_emb1 - t_emb2).mean()
            print(f"   Text encoder output diff: {text_diff.item():.8f}")
            
        if audio_diff < 1e-6:
            print("   ❌ Audio encoder produces identical outputs!")
        if text_diff < 1e-6:
            print("   ❌ Text encoder produces identical outputs!")
            
    else:
        print("   ✅ Outputs differ - model is input-sensitive!")
    
    print(f"\n" + "=" * 50)
    print("🏁 Gradient Flow Analysis:")
    
    if trainable_params < 1000:
        print("❌ Very few trainable parameters - model can't adapt!")
        print("💡 Solution: Unfreeze more encoder layers")
    
    small_grads = sum(1 for norm in grad_norms.values() if norm < 1e-6)
    if small_grads > len(grad_norms) * 0.5:
        print("❌ Many gradients are very small - vanishing gradient problem!")
        print("💡 Solution: Increase learning rate or reduce model depth")
    
    if diff < 1e-6:
        print("❌ Model produces identical outputs - not learning!")
        print("💡 Solutions:")
        print("   1. Unfreeze more encoder layers")
        print("   2. Check if inputs are being processed correctly")
        print("   3. Reduce model complexity")


if __name__ == "__main__":
    debug_gradient_flow() 