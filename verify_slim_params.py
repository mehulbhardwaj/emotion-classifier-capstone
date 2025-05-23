#!/usr/bin/env python3
"""
Verify parameter counts for slim model configurations.
Ensures all models are under the 10M parameter target.
"""

import torch
import torch.nn as nn
from configs.base_config import Config


def calculate_transformer_params(d_model, n_heads, n_layers, dim_feedforward):
    """Calculate transformer encoder parameters."""
    # Multi-head attention: 4 linear layers (Q, K, V, output) of size d_model x d_model
    attention_params = 4 * d_model * d_model * n_layers
    
    # Feed-forward: 2 linear layers 
    feedforward_params = 2 * d_model * dim_feedforward * n_layers
    
    # LayerNorm: 2 per layer (attention + feedforward), each has 2*d_model params
    layernorm_params = 2 * 2 * d_model * n_layers
    
    return attention_params + feedforward_params + layernorm_params


def calculate_gru_params(input_size, hidden_size, num_layers, bidirectional=True):
    """Calculate GRU parameters."""
    # GRU has 3 gates: reset, update, new
    # Each gate: input_to_hidden + hidden_to_hidden + bias
    directions = 2 if bidirectional else 1
    
    # For each layer
    params_per_layer = 3 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    total_params = params_per_layer * num_layers * directions
    
    return total_params


def verify_todkat_slim():
    """Verify TOD-KAT Slim parameter count."""
    print("🔍 TOD-KAT Slim Parameter Analysis")
    print("=" * 50)
    
    # Config values
    topic_embedding_dim = 32
    n_topics = 32
    projection_dim = 128
    mlp_hidden_size = 512
    rel_transformer_layers = 1
    rel_heads = 2
    transformer_dim_feedforward = 256
    
    # Calculate d_model
    d_model = projection_dim + projection_dim + topic_embedding_dim  # audio + text + topic
    print(f"d_model: {projection_dim} + {projection_dim} + {topic_embedding_dim} = {d_model}")
    
    # Check if heads divide d_model
    assert d_model % rel_heads == 0, f"d_model ({d_model}) must be divisible by heads ({rel_heads})"
    
    # Parameter calculations
    topic_emb_params = n_topics * topic_embedding_dim
    projection_params = 768 * projection_dim * 2  # audio + text projections
    transformer_params = calculate_transformer_params(d_model, rel_heads, rel_transformer_layers, transformer_dim_feedforward)
    
    # Classifier: [audio_proj + text_proj + context] -> mlp_hidden -> mlp_hidden//2 -> 7
    cls_input_dim = projection_dim + projection_dim + d_model
    classifier_params = (cls_input_dim * mlp_hidden + mlp_hidden +
                        mlp_hidden * (mlp_hidden // 2) + (mlp_hidden // 2) +
                        (mlp_hidden // 2) * 7 + 7)
    
    total_params = topic_emb_params + projection_params + transformer_params + classifier_params
    
    print(f"Topic embeddings: {topic_emb_params:,}")
    print(f"Projection layers: {projection_params:,}")
    print(f"Transformer: {transformer_params:,}")
    print(f"Classifier: {classifier_params:,}")
    print(f"Total trainable: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Under 10M target: {'✅' if total_params < 10e6 else '❌'}")
    return total_params


def verify_dialog_rnn_slim():
    """Verify DialogRNN Slim parameter count."""
    print("\n🔍 DialogRNN Slim Parameter Analysis")
    print("=" * 50)
    
    # Config values
    gru_hidden_size = 64
    mlp_hidden_size = 512
    enc_dim = 768 + 768  # audio + text encoder dims
    
    # 3 GRUs: global, speaker, emotion (each 2-layer bidirectional)
    gru_params_each = calculate_gru_params(enc_dim, gru_hidden_size, num_layers=2, bidirectional=True)
    total_gru_params = 3 * gru_params_each
    
    # Classifier input: enc_dim + 3 * (gru_hidden_size * 2)  # 2 for bidirectional
    total_dim = enc_dim + 3 * (gru_hidden_size * 2)
    classifier_params = (total_dim * mlp_hidden_size + mlp_hidden_size +
                        mlp_hidden_size * (mlp_hidden_size // 2) + (mlp_hidden_size // 2) +
                        (mlp_hidden_size // 2) * 7 + 7)
    
    # LayerNorm
    layernorm_params = total_dim * 2  # weight + bias
    
    total_params = total_gru_params + classifier_params + layernorm_params
    
    print(f"Input dimension: {enc_dim}")
    print(f"Total dimension: {total_dim}")
    print(f"GRU params (3 x 2-layer bidirectional): {total_gru_params:,}")
    print(f"Classifier params: {classifier_params:,}")
    print(f"LayerNorm params: {layernorm_params:,}")
    print(f"Total trainable: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Under 10M target: {'✅' if total_params < 10e6 else '❌'}")
    return total_params


def verify_mlp_fusion_slim():
    """Verify MLP Fusion Slim parameter count."""
    print("\n🔍 MLP Fusion Slim Parameter Analysis")
    print("=" * 50)
    
    # Config values
    mlp_hidden_size = 1024
    input_dim = 768 + 768  # audio + text encoder dims
    
    # MLP: input_dim -> mlp_hidden -> mlp_hidden//2 -> 7
    mlp_params = (input_dim * mlp_hidden_size + mlp_hidden_size +
                 mlp_hidden_size * (mlp_hidden_size // 2) + (mlp_hidden_size // 2) +
                 (mlp_hidden_size // 2) * 7 + 7)
    
    total_params = mlp_params
    
    print(f"Input dimension: {input_dim}")
    print(f"MLP hidden size: {mlp_hidden_size}")
    print(f"MLP params: {mlp_params:,}")
    print(f"Total trainable: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Under 10M target: {'✅' if total_params < 10e6 else '❌'}")
    return total_params


def main():
    """Main verification function."""
    print("🎯 Slim Model Parameter Verification")
    print("=" * 60)
    print("All models use FROZEN encoders (0 trainable encoder params)")
    print("=" * 60)
    
    todkat_params = verify_todkat_slim()
    dialog_params = verify_dialog_rnn_slim()
    mlp_params = verify_mlp_fusion_slim()
    
    print("\n📊 Summary")
    print("=" * 40)
    print(f"TOD-KAT Slim:    {todkat_params/1e6:.2f}M parameters")
    print(f"DialogRNN Slim:  {dialog_params/1e6:.2f}M parameters")
    print(f"MLP Fusion Slim: {mlp_params/1e6:.2f}M parameters")
    
    all_under_10m = all(p < 10e6 for p in [todkat_params, dialog_params, mlp_params])
    print(f"\nAll under 10M target: {'✅' if all_under_10m else '❌'}")
    
    if all_under_10m:
        print("🎉 All models successfully optimized for fast training!")
        print("💡 Expected benefits:")
        print("   - Much faster training (more steps/epoch)")
        print("   - Lower memory usage")
        print("   - Can use larger batch sizes")
        print("   - Still competitive F1 scores with frozen encoder features")


if __name__ == "__main__":
    main() 