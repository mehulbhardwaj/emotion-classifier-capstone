"""Tests for the MLP Fusion model."""
import pytest
import torch
import torch.nn as nn
from models.mlp_fusion import MultimodalFusionMLP
from transformers import AutoModel, AutoProcessor, AutoConfig

# Fixture for model config
@pytest.fixture(params=[
    # Test case 1: Freeze both encoders, raw audio input
    {
        'mlp_hidden_size': 256,
        'mlp_dropout_rate': 0.3,
        'text_encoder_model_name': 'distilroberta-base',
        'audio_encoder_model_name': 'facebook/wav2vec2-base',
        'text_feature_dim': 768,
        'audio_feature_dim': 768,
        'output_dim': 7,
        'learning_rate': 1e-4,
        'freeze_text_encoder': True,
        'freeze_audio_encoder': True,
        'audio_input_type': 'raw_wav'
    },
    # Test case 2: Don't freeze encoders, pre-extracted features
    {
        'mlp_hidden_size': 256,  # Using same hidden size as model expects
        'mlp_dropout_rate': 0.1,
        'text_encoder_model_name': 'distilroberta-base',
        'audio_encoder_model_name': 'facebook/wav2vec2-base',
        'text_feature_dim': 768,
        'audio_feature_dim': 1024,
        'output_dim': 7,
        'learning_rate': 1e-5,
        'freeze_text_encoder': False,
        'freeze_audio_encoder': False,
        'audio_input_type': 'pre_extracted'
    }
])
def model_config(request):
    return request.param

# Fixture for test batch data
@pytest.fixture
def test_batch(model_config):
    batch_size = 2
    seq_len = 32
    audio_len = 16000
    
    batch = {
        'text_input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'text_attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        'input_values': torch.randn(batch_size, audio_len),
        'attention_mask': torch.ones(batch_size, 49, dtype=torch.long),
        'labels': torch.randint(0, model_config['output_dim'], (batch_size,))
    }
    
    # Adjust batch based on audio input type
    if model_config['audio_input_type'] == 'pre_extracted':
        batch['audio_features'] = torch.randn(batch_size, model_config['audio_feature_dim'])
    else:
        # For raw audio, we need to add the audio processor output
        batch['input_values'] = torch.randn(batch_size, audio_len)
        
    return batch

# Fixture for model
@pytest.fixture
def model(model_config):
    return MultimodalFusionMLP(**model_config)

def test_mlp_fusion_init(model, model_config):
    """Test if we can initialize an MLP Fusion model with different configs."""
    assert model is not None
    
    # Test encoder freezing
    if model_config['freeze_text_encoder']:
        for param in model.text_encoder.parameters():
            assert not param.requires_grad
    else:
        assert any(p.requires_grad for p in model.text_encoder.parameters())
    
    if model_config['freeze_audio_encoder']:
        for param in model.audio_encoder.parameters():
            assert not param.requires_grad
    else:
        assert any(p.requires_grad for p in model.audio_encoder.parameters())
            
    # Test audio processor initialization
    if model_config['audio_input_type'] == 'raw_wav':
        assert model.audio_processor is not None
    else:
        assert model.audio_processor is None
        
    # Test model components
    assert isinstance(model.fusion_mlp, nn.Sequential)
    assert isinstance(model.criterion, nn.CrossEntropyLoss)
    # Check that the model was initialized with the correct config values
    assert model.hparams.mlp_dropout_rate == model_config['mlp_dropout_rate']
    assert model.learning_rate == model_config['learning_rate']

def test_forward_pass(model, model_config, test_batch):
    """Test the forward pass of the model."""
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in test_batch.items()}
    
    # Forward pass
    output = model(batch)
    
    # Check output shape
    assert output.shape == (test_batch['labels'].shape[0], model_config['output_dim'])

def test_training_step(model, model_config, test_batch):
    """Test the training step of the model."""
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in test_batch.items()}
    
    # Training step
    loss = model.training_step(batch, batch_idx=0)
    
    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss) and not torch.isinf(loss)

def test_validation_step(model, model_config, test_batch):
    """Test the validation step of the model."""
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in test_batch.items()}
    
    # Validation step - returns a tensor (loss) directly
    loss = model.validation_step(batch, batch_idx=0)
    
    # Check that the output is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss) and not torch.isinf(loss)

def test_configure_optimizers(model):
    """Test the optimizer configuration."""
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Optimizer)

def test_test_step(model, model_config, test_batch):
    """Test the test step of the model."""
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in test_batch.items()}
    
    # Test step
    loss = model.test_step(batch, batch_idx=0)
    
    # Check that the output is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss) and not torch.isinf(loss)

def test_predict_step(model, model_config, test_batch):
    """Test the predict step of the model."""
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in test_batch.items()}
    
    # Predict step
    logits = model.predict_step(batch, batch_idx=0)
    
    # Check output shape
    assert logits.shape == (test_batch['labels'].shape[0], model_config['output_dim'])

def test_model_save_load(tmp_path, model):
    """Test model saving and loading."""
    # Save the model
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Create a new model with the same config
    new_model = MultimodalFusionMLP(
        mlp_hidden_size=model.hparams.mlp_hidden_size,
        mlp_dropout_rate=model.hparams.mlp_dropout_rate,
        text_encoder_model_name=model.hparams.text_encoder_model_name,
        audio_encoder_model_name=model.hparams.audio_encoder_model_name,
        text_feature_dim=model.hparams.text_feature_dim,
        audio_feature_dim=model.hparams.audio_feature_dim,
        output_dim=model.hparams.output_dim,
        learning_rate=model.learning_rate,
        freeze_text_encoder=model.hparams.freeze_text_encoder,
        freeze_audio_encoder=model.hparams.freeze_audio_encoder,
        audio_input_type=model.hparams.audio_input_type
    )
    
    # Load the saved state dict
    new_model.load_state_dict(torch.load(model_path))
    
    # Check that parameters match
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), new_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2, atol=1e-6)

def test_forward_pass_edge_cases(model, model_config):
    """Test edge cases in forward pass."""
    device = next(model.parameters()).device
    
    # Test with empty batch
    empty_batch = {
        'text_input_ids': torch.empty(0, 10, dtype=torch.long, device=device),
        'text_attention_mask': torch.empty(0, 10, dtype=torch.long, device=device),
        'input_values': torch.empty(0, 16000, device=device),
        'labels': torch.empty(0, dtype=torch.long, device=device)
    }
    if model_config['audio_input_type'] == 'pre_extracted':
        empty_batch['audio_features'] = torch.empty(0, model_config['audio_feature_dim'], device=device)
    
    with torch.no_grad():
        output = model(empty_batch)
        assert output.shape == (0, model_config['output_dim'])
    
    # Test with sequence length 1
    single_batch = {
        'text_input_ids': torch.randint(0, 1000, (1, 1), device=device),
        'text_attention_mask': torch.ones(1, 1, dtype=torch.long, device=device),
        'input_values': torch.randn(1, 16000, device=device),
        'labels': torch.randint(0, model_config['output_dim'], (1,), device=device)
    }
    if model_config['audio_input_type'] == 'pre_extracted':
        single_batch['audio_features'] = torch.randn(1, model_config['audio_feature_dim'], device=device)
    
    with torch.no_grad():
        output = model(single_batch)
        assert output.shape == (1, model_config['output_dim'])

def test_gradient_flow(model, test_batch):
    """Test that gradients flow correctly through the model."""
    # Move model and batch to the same device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in test_batch.items()}
    
    # Forward pass
    output = model(batch)
    
    # Create a dummy loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are flowing
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.all(param.grad == 0.0), f"Zero gradients for {name}"
    
    assert has_grad, "No gradients were computed"
