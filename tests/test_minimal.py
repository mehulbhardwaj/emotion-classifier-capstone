"""Minimal test to verify core imports are working."""

def test_imports():
    """Test if we can import the main modules."""
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Test importing main modules
    try:
        # Test model imports
        from models import mlp_fusion
        from models import teacher
        from models import student
        from models import panns_fusion
        
        # Test utils imports
        from utils import data_processor
        
        # If we get here, all imports worked
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"

def test_teacher_model_init():
    """Test if we can initialize a Teacher model."""
    import torch
    from models.teacher import TeacherTransformer
    
    # Create a simple config dictionary
    config = {
        'hidden_size': 256,
        'num_transformer_layers': 2,
        'num_transformer_heads': 4,
        'dropout_rate': 0.3,
        'text_encoder_model_name': 'distilroberta-base',
        'audio_encoder_model_name': 'facebook/wav2vec2-base',
        'text_feature_dim': 768,
        'audio_feature_dim': 768,
        'output_dim': 7,
        'learning_rate': 1e-4,
        'freeze_text_encoder': True,
        'freeze_audio_encoder': True,
        'audio_input_type': 'raw_wav'
    }
    
    try:
        model = TeacherTransformer(**config)
        assert model is not None
        
        # Test forward pass with dummy data
        batch_size = 2
        seq_len = 32
        audio_len = 16000
        
        # Create dummy batch
        batch = {
            'text_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),  # (batch_size, seq_len)
            'text_attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),  # (batch_size, seq_len)
            'input_values': torch.randn(batch_size, audio_len),  # (batch_size, audio_len)
            # Wav2Vec2 reduces the time dimension, so we'll use a smaller attention mask
            'attention_mask': torch.ones(batch_size, 49, dtype=torch.long)  # (batch_size, reduced_seq_len)
        }
        
        # Move batch to the same device as model
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        output = model(batch)
        assert output.shape == (batch_size, config['output_dim'])
        
    except Exception as e:
        assert False, f"Teacher model initialization or forward pass failed: {e}"
