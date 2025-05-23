#!/usr/bin/env python3
"""Debug script to check exact dimensions in TOD-KAT."""

import torch
from transformers import Wav2Vec2Model, RobertaModel
from configs.base_config import Config

def check_dimensions():
    print("🔍 Checking TOD-KAT dimensions...")
    
    # Create config
    config = Config()
    config.topic_embedding_dim = 128
    config.use_knowledge = True  # Test with knowledge enabled
    config.rel_heads = 4
    
    # Load encoders
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    text_encoder = RobertaModel.from_pretrained("roberta-base")
    
    # Get dimensions
    audio_dim = audio_encoder.config.hidden_size
    text_dim = text_encoder.config.hidden_size
    topic_dim = config.topic_embedding_dim
    kn_dim = 64 if config.use_knowledge else 0
    
    d_model = audio_dim + text_dim + topic_dim + kn_dim
    n_heads = config.rel_heads
    
    print(f"Audio encoder dim: {audio_dim}")
    print(f"Text encoder dim: {text_dim}")
    print(f"Topic embedding dim: {topic_dim}")
    print(f"Knowledge dim: {kn_dim}")
    print(f"Total d_model: {d_model}")
    print(f"Number of heads: {n_heads}")
    print(f"d_model / n_heads = {d_model / n_heads}")
    print(f"d_model % n_heads = {d_model % n_heads}")
    print(f"Is divisible? {d_model % n_heads == 0}")
    
    # Test different head counts
    print("\nTesting different head counts:")
    for heads in [1, 2, 4, 8, 16]:
        divisible = (d_model % heads == 0)
        result = "✅" if divisible else "❌"
        print(f"  {heads} heads: {d_model}/{heads} = {d_model/heads} {result}")

if __name__ == "__main__":
    check_dimensions() 