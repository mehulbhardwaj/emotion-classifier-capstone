#!/usr/bin/env python3
"""Debug script to check exact dimensions in TOD-KAT."""

import torch
from transformers import Wav2Vec2Model, RobertaModel
from configs.base_config import Config

def check_dimensions():
    print("🔍 Checking TOD-KAT dimensions...")
    
    # Create config
    config = Config()
    config.topic_embedding_dim = 32  # Updated to match new config
    config.use_knowledge = True  # FIXED: Knowledge is core to TOD-KAT
    config.knowledge_dim = 16    # Updated to match new config
    config.rel_heads = 4
    config.projection_dim = 128  # NEW: projection dimension
    
    # Load encoders
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    text_encoder = RobertaModel.from_pretrained("roberta-base")
    
    # Get dimensions
    audio_dim = audio_encoder.config.hidden_size
    text_dim = text_encoder.config.hidden_size
    projection_dim = config.projection_dim
    topic_dim = config.topic_embedding_dim
    kn_dim = 0 if not config.use_knowledge else config.knowledge_dim
    
    d_model = projection_dim + projection_dim + topic_dim + kn_dim
    n_heads = config.rel_heads
    
    print(f"Original audio encoder dim: {audio_dim}")
    print(f"Original text encoder dim: {text_dim}")
    print(f"Projected audio/text dim: {projection_dim}")
    print(f"Topic embedding dim: {topic_dim}")
    print(f"Knowledge dim: {kn_dim}")
    print(f"Total d_model: {d_model}")
    print(f"Number of heads: {n_heads}")
    print(f"d_model / n_heads = {d_model / n_heads}")
    print(f"d_model % n_heads = {d_model % n_heads}")
    print(f"Is divisible? {d_model % n_heads == 0}")
    
    # Estimate transformer parameters
    # Self-attention: 4 * d_model^2 per layer (Q, K, V, O matrices)
    # Feed-forward: 2 * d_model * dim_feedforward per layer  
    dim_feedforward = 128  # From the model
    n_layers = 2
    
    attention_params = 4 * d_model * d_model * n_layers
    feedforward_params = 2 * d_model * dim_feedforward * n_layers
    total_transformer_params = attention_params + feedforward_params
    
    print(f"\nTransformer parameter breakdown:")
    print(f"  Attention params: {attention_params:,} ({attention_params/1e6:.1f}M)")
    print(f"  Feedforward params: {feedforward_params:,} ({feedforward_params/1e6:.1f}M)")
    print(f"  Total transformer: {total_transformer_params:,} ({total_transformer_params/1e6:.1f}M)")
    
    # Add projection layer parameters
    projection_params = audio_dim * projection_dim + text_dim * projection_dim
    print(f"  Projection layers: {projection_params:,} ({projection_params/1e6:.1f}M)")
    
    total_trainable = total_transformer_params + projection_params
    print(f"  Total trainable (est): {total_trainable:,} ({total_trainable/1e6:.1f}M)")
    
    # Test different head counts
    print("\nTesting different head counts:")
    for heads in [1, 2, 4, 8, 16]:
        divisible = (d_model % heads == 0)
        result = "✅" if divisible else "❌"
        print(f"  {heads} heads: {d_model}/{heads} = {d_model/heads} {result}")

if __name__ == "__main__":
    check_dimensions() 