#!/usr/bin/env python3
"""
Test script to verify model initialization.
"""

import torch
from config import Config
from models.mlp_fusion import MultimodalFusionMLP
from models.teacher import TeacherTransformer
from models.student import StudentGRU
from models.panns_fusion import PaNNsFusion

# Create a basic configuration
config = Config()
print(f"Created Config with device: {config.device}")

# Test MLP Fusion model initialization
print("\nInitializing MLP Fusion model...")
try:
    mlp_model = MultimodalFusionMLP(
        mlp_hidden_size=256,
        mlp_dropout_rate=0.3,
        text_encoder_model_name="distilbert-base-uncased",  # Use a smaller model for testing
        audio_encoder_model_name="facebook/wav2vec2-base", 
        text_feature_dim=768,
        audio_feature_dim=768,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        audio_input_type="hf_features",
        output_dim=7,
        learning_rate=1e-4
    )
    print("✓ MLP Fusion model initialization successful")
    # Print model parameters count
    total_params = sum(p.numel() for p in mlp_model.parameters())
    trainable_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ MLP Fusion model initialization failed: {e}")

# Test Teacher model initialization
print("\nInitializing Teacher model...")
try:
    teacher_model = TeacherTransformer(
        hidden_size=256,
        num_transformer_layers=2,
        num_transformer_heads=4,
        dropout_rate=0.3,
        text_encoder_model_name="distilbert-base-uncased",
        audio_encoder_model_name="facebook/wav2vec2-base",
        text_feature_dim=768,
        audio_feature_dim=768,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        audio_input_type="hf_features",
        output_dim=7,
        learning_rate=1e-4
    )
    print("✓ Teacher model initialization successful")
    # Print model parameters count
    total_params = sum(p.numel() for p in teacher_model.parameters())
    trainable_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ Teacher model initialization failed: {e}")

# Test Student model initialization
print("\nInitializing Student model...")
try:
    student_model = StudentGRU(
        hidden_size=128,
        gru_layers=2,
        dropout_rate=0.3,
        text_encoder_model_name="distilbert-base-uncased",
        audio_encoder_model_name="facebook/wav2vec2-base",
        text_feature_dim=768,
        audio_feature_dim=768,
        freeze_text_encoder=True,
        freeze_audio_encoder=True,
        audio_input_type="hf_features",
        output_dim=7,
        learning_rate=1e-4
    )
    print("✓ Student model initialization successful")
    # Print model parameters count
    total_params = sum(p.numel() for p in student_model.parameters())
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ Student model initialization failed: {e}")

# Test PaNNs Fusion model initialization
print("\nInitializing PaNNs Fusion model...")
try:
    panns_model = PaNNsFusion(
        hidden_size=256,
        dropout_rate=0.3,
        text_encoder_model_name="distilbert-base-uncased",
        use_panns_features=True,
        panns_feature_dim=2048,
        text_feature_dim=768,
        freeze_text_encoder=True,
        output_dim=7,
        learning_rate=1e-4
    )
    print("✓ PaNNs Fusion model initialization successful")
    # Print model parameters count
    total_params = sum(p.numel() for p in panns_model.parameters())
    trainable_params = sum(p.numel() for p in panns_model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
except Exception as e:
    print(f"✗ PaNNs Fusion model initialization failed: {e}")

print("\nModel initialization test complete.")
