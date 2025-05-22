"""Tests for the Student GRU model."""
import pytest
import torch
import torch.nn as nn
from models.student import StudentGRU

# Fixture for model config
@pytest.fixture(params=[
    # Test case 1: Freeze both encoders, raw audio input, no distillation
    {
        'hidden_size': 128,
        'gru_layers': 2,
        'dropout_rate': 0.3,
        'text_encoder_model_name': 'distilroberta-base',
        'audio_encoder_model_name': 'facebook/wav2vec2-base',
        'text_feature_dim': 768,
        'audio_feature_dim': 768,
        'freeze_text_encoder': True,
        'freeze_audio_encoder': True,
        'audio_input_type': 'raw_wav',
        'output_dim': 7,
        'learning_rate': 1e-4,
        'use_distillation': False,
        'distillation_temperature': 2.0,
        'distillation_alpha': 0.5
    },
    # Test case 2: Don't freeze encoders, hf features, with distillation
    {
        'hidden_size': 256,
        'gru_layers': 1,
        'dropout_rate': 0.2,
        'text_encoder_model_name': 'distilroberta-base',
        'audio_encoder_model_name': 'facebook/wav2vec2-base',
        'text_feature_dim': 768,
        'audio_feature_dim': 768,
        'freeze_text_encoder': False,
        'freeze_audio_encoder': False,
        'audio_input_type': 'hf_features',
        'output_dim': 7,
        'learning_rate': 2e-5,
        'use_distillation': True,
        'distillation_temperature': 1.5,
        'distillation_alpha': 0.7
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
    
    # Add teacher logits if using distillation
    if model_config['use_distillation']:
        batch['teacher_logits'] = torch.randn(batch_size, model_config['output_dim'])
    
    # Add audio features if using hf_features
    if model_config['audio_input_type'] == 'hf_features':
        batch['audio_features'] = torch.randn(batch_size, model_config['audio_feature_dim'])
    
    return batch

# Fixture for model
@pytest.fixture
def model(model_config):
    return StudentGRU(**model_config)

def test_student_init(model, model_config):
    """Test if we can initialize a StudentGRU model with different configs."""
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
    
    # Test model components
    assert isinstance(model.text_projection, nn.Linear)
    assert isinstance(model.audio_projection, nn.Linear)
    assert isinstance(model.gru, nn.GRU)
    assert isinstance(model.classifier, nn.Sequential)
    assert isinstance(model.criterion, nn.CrossEntropyLoss)
    
    # Check hyperparameters
    assert model.hparams.hidden_size == model_config['hidden_size']
    assert model.hparams.dropout_rate == model_config['dropout_rate']
    assert model.learning_rate == model_config['learning_rate']
    assert model.hparams.use_distillation == model_config['use_distillation']

def test_forward_pass(model, model_config, test_batch):
    """Test the forward pass of the model."""
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Forward pass
    logits = model(test_batch)
    
    # Check output shape
    assert logits.shape == (test_batch['labels'].shape[0], model_config['output_dim'])

def test_training_step(model, model_config, test_batch):
    """Test the training step of the model."""
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Training step
    loss = model.training_step(test_batch, batch_idx=0)
    
    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar

def test_validation_step(model, model_config, test_batch):
    """Test the validation step of the model."""
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Validation step
    loss = model.validation_step(test_batch, batch_idx=0)
    
    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar

def test_distillation_loss(model, model_config, test_batch):
    """Test the distillation loss calculation."""
    if not model_config['use_distillation']:
        return
        
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Get student logits
    student_logits = model(test_batch)
    
    # Calculate distillation loss
    loss = model._distillation_loss(
        student_logits,
        test_batch['teacher_logits'],
        test_batch['labels']
    )
    
    # Check loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar

def test_configure_optimizers(model):
    """Test the optimizer configuration."""
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults['lr'] == model.learning_rate

def test_test_step(model, model_config, test_batch):
    """Test the test step of the model."""
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Test step
    loss = model.test_step(test_batch, batch_idx=0)
    
    # Check that the output is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss) and not torch.isinf(loss)

def test_predict_step(model, model_config, test_batch):
    """Test the predict step of the model."""
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Predict step
    logits = model.predict_step(test_batch, batch_idx=0)
    
    # Check output shape
    assert logits.shape == (test_batch['labels'].shape[0], model_config['output_dim'])

def test_model_save_load(tmp_path, model, model_config):
    """Test model saving and loading."""
    # Save the model
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), model_path)
    
    # Create a new model with the same config
    new_model = StudentGRU(**model_config)
    
    # Load the saved state dict
    new_model.load_state_dict(torch.load(model_path, weights_only=False))
    
    # Check that parameters match
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), new_model.named_parameters()
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2, atol=1e-6)

def test_forward_pass_edge_cases(model, model_config):
    """Test edge cases in forward pass."""
    device = next(model.parameters()).device
    
    # Skip empty batch test for raw audio input as Wav2Vec2 doesn't handle empty tensors
    if model_config['audio_input_type'] == 'hf_features':
        # Test with empty batch (only for HF features)
        empty_batch = {
            'text_input_ids': torch.empty(0, 10, dtype=torch.long, device=device),
            'text_attention_mask': torch.empty(0, 10, dtype=torch.long, device=device),
            'audio_features': torch.empty(0, model_config['audio_feature_dim'], device=device),
            'labels': torch.empty(0, dtype=torch.long, device=device)
        }
        
        if model_config['use_distillation']:
            empty_batch['teacher_logits'] = torch.empty(0, model_config['output_dim'], device=device)
        
        with torch.no_grad():
            output = model(empty_batch)
            assert output.shape == (0, model_config['output_dim'])
    
    # Test with sequence length 1
    single_batch = {
        'text_input_ids': torch.randint(0, 1000, (1, 1), device=device),
        'text_attention_mask': torch.ones(1, 1, dtype=torch.long, device=device),
        'labels': torch.randint(0, model_config['output_dim'], (1,), device=device)
    }
    
    # Add appropriate audio input based on audio_input_type
    if model_config['audio_input_type'] == 'hf_features':
        single_batch['audio_features'] = torch.randn(1, model_config['audio_feature_dim'], device=device)
    else:  # raw_wav
        single_batch['input_values'] = torch.randn(1, 16000, device=device)
    
    if model_config['use_distillation']:
        single_batch['teacher_logits'] = torch.randn(1, model_config['output_dim'], device=device)
    if model_config['audio_input_type'] == 'hf_features':
        single_batch['audio_features'] = torch.randn(1, model_config['audio_feature_dim'], device=device)
    
    with torch.no_grad():
        output = model(single_batch)
        assert output.shape == (1, model_config['output_dim'])

def test_gradient_flow(model, model_config, test_batch):
    """Test that gradients flow correctly through the model."""
    # Move model to CPU for testing
    model = model.to('cpu')
    
    # Forward pass
    output = model(test_batch)
    
    # Create a dummy loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are flowing
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            # Skip audio_projection if using raw audio
            if name == 'audio_projection.weight' and model_config['audio_input_type'] != 'hf_features':
                continue
            assert not torch.all(param.grad == 0.0), f"Zero gradients for {name}"
    
    assert has_grad, "No gradients were computed"

def test_distillation_handling(model, model_config):
    """Test that distillation is handled correctly."""
    device = next(model.parameters()).device
    
    # Create a batch with and without teacher logits
    batch_size = 2
    batch = {
        'text_input_ids': torch.randint(0, 1000, (batch_size, 10), device=device),
        'text_attention_mask': torch.ones(batch_size, 10, dtype=torch.long, device=device),
        'input_values': torch.randn(batch_size, 16000, device=device),
        'labels': torch.randint(0, model_config['output_dim'], (batch_size,), device=device)
    }
    
    if model_config['audio_input_type'] == 'hf_features':
        batch['audio_features'] = torch.randn(batch_size, model_config['audio_feature_dim'], device=device)
    
    # Test with teacher logits
    if model_config['use_distillation']:
        batch['teacher_logits'] = torch.randn(batch_size, model_config['output_dim'], device=device)
        with torch.no_grad():
            output = model(batch)
            assert output.shape == (batch_size, model_config['output_dim'])
    
    # Test without teacher logits (should still work)
    if 'teacher_logits' in batch:
        del batch['teacher_logits']
    with torch.no_grad():
        output = model(batch)
        assert output.shape == (batch_size, model_config['output_dim'])
