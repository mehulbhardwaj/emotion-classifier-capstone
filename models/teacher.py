"""Teacher model for emotion classification.

A simplified implementation of a transformer-based model for emotion classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional


class TeacherTransformer(pl.LightningModule):
    """Teacher model for emotion classification using a transformer architecture.
    
    Uses pre-trained encoders for audio and text, with a transformer-based fusion mechanism.
    """
    
    def __init__(
        self, 
        hidden_size: int = 256,
        num_transformer_layers: int = 2,
        num_transformer_heads: int = 4,
        dropout_rate: float = 0.3,
        text_encoder_model_name: str = "distilroberta-base",
        audio_encoder_model_name: str = "facebook/wav2vec2-base-960h",
        text_feature_dim: int = 768,
        audio_feature_dim: int = 768,
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = True,
        audio_input_type: str = "hf_features",
        output_dim: int = 7,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        """Initialize the model.
        
        Args:
            hidden_size: Dimension of the hidden layer.
            num_transformer_layers: Number of transformer layers for fusion.
            num_transformer_heads: Number of attention heads in transformer.
            dropout_rate: Dropout rate.
            text_encoder_model_name: Name of the pre-trained text encoder model.
            audio_encoder_model_name: Name of the pre-trained audio encoder model.
            text_feature_dim: Dimension of text features.
            audio_feature_dim: Dimension of audio features.
            freeze_text_encoder: Whether to freeze the text encoder parameters.
            freeze_audio_encoder: Whether to freeze the audio encoder parameters.
            audio_input_type: Type of audio input ("hf_features" or "raw_wav").
            output_dim: Number of output classes.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        
        # Save arguments as attributes
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.audio_input_type = audio_input_type
        self.text_feature_dim = text_feature_dim
        self.audio_feature_dim = audio_feature_dim
        
        # Load text encoder
        self.text_encoder = self._load_text_encoder(text_encoder_model_name)
        if freeze_text_encoder:
            self._freeze_encoder(self.text_encoder)

        # Load audio encoder
        self.audio_encoder = self._load_audio_encoder(audio_encoder_model_name)
        if freeze_audio_encoder:
            self._freeze_encoder(self.audio_encoder)
            
        # Feature dimensionality reduction (optional)
        self.text_projection = nn.Linear(text_feature_dim, hidden_size)
        self.audio_projection = nn.Linear(audio_feature_dim, hidden_size)
        
        # Fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_transformer_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _load_text_encoder(self, model_name: str) -> nn.Module:
        """Load text encoder model."""
        return AutoModel.from_pretrained(model_name)

    def _load_audio_encoder(self, model_name: str) -> nn.Module:
        """Load audio encoder model."""
        return AutoModel.from_pretrained(model_name)

    def _freeze_encoder(self, encoder: nn.Module):
        """Freeze encoder parameters."""
        for param in encoder.parameters():
            param.requires_grad = False

    def forward(self, batch: Dict[str, Any]):
        """Forward pass."""
        device = self.device
        batch_size = next(iter(batch.values())).size(0)
        
        # Text encoding
        if "text_input_ids" in batch and batch["text_input_ids"] is not None:
            text_inputs = batch["text_input_ids"]
            text_attention_mask = batch.get("text_attention_mask")
            text_outputs = self.text_encoder(
                input_ids=text_inputs,
                attention_mask=text_attention_mask,
                return_dict=True
            )
            if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
                text_features = text_outputs.pooler_output
            else:
                # Mean pooling of the last hidden state
                text_features = self._mean_pooling(
                    text_outputs.last_hidden_state,
                    text_attention_mask
                )
        else:
            # Create zero tensor if no text input
            text_features = torch.zeros(batch_size, self.text_feature_dim, device=device)
        
        # Project text features to hidden size
        text_projected = self.text_projection(text_features)

        # Audio encoding
        if "input_values" in batch and batch["input_values"] is not None:
            audio_inputs = batch["input_values"]
            audio_attention_mask = batch.get("attention_mask")
            
            # For raw audio input, we need to pass it directly to the model
            audio_outputs = self.audio_encoder(
                input_values=audio_inputs,
                attention_mask=audio_attention_mask,
                return_dict=True
            )
            
            # Extract features from the output
            if hasattr(audio_outputs, 'last_hidden_state'):
                # For models that return last_hidden_state (like Wav2Vec2)
                audio_features = self._mean_pooling(
                    audio_outputs.last_hidden_state,
                    audio_attention_mask
                )
            else:
                # For models that return pooled_output (like some Wav2Vec2 variants)
                audio_features = audio_outputs.pooler_output
        else:
            # Create zero tensor if no audio input
            audio_features = torch.zeros(batch_size, self.audio_feature_dim, device=device)
        
        # Ensure audio features have the correct dimension before projection
        if audio_features.dim() > 2:
            # If we have sequence dimension, take mean over it
            audio_features = audio_features.mean(dim=1)
            
        # If using pre-extracted features, they should already be in the correct dimension
        # If using raw audio, we need to project to the expected dimension
        if audio_features.size(-1) != self.audio_feature_dim:
            # If dimensions don't match, use a projection
            projection = nn.Linear(audio_features.size(-1), self.audio_feature_dim).to(device)
            audio_features = projection(audio_features)
            
        # Project to hidden size
        audio_projected = self.audio_projection(audio_features)

        # Combine modalities for transformer input (batch_size, 2, hidden_size)
        # Adding positional tokens for text and audio
        multimodal_input = torch.stack([text_projected, audio_projected], dim=1)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(multimodal_input)
        
        # Use [CLS] token (first token) for classification
        cls_representation = transformer_output[:, 0, :]
        
        # Pass through classifier
        logits = self.classifier(cls_representation)
        
        return logits

    def _mean_pooling(self, hidden_states, attention_mask=None):
        """Mean pooling of hidden states with attention mask."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
            
        # Ensure attention_mask has the same device as hidden_states
        attention_mask = attention_mask.to(hidden_states.device)
        
        # If attention_mask has fewer dimensions than hidden_states, unsqueeze it
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(-1)
            
        # If dimensions still don't match, adjust the attention mask
        if attention_mask.size(1) != hidden_states.size(1):
            # Resize attention_mask to match the sequence length of hidden_states
            attention_mask = attention_mask[:, :hidden_states.size(1)]
        
        # Expand attention mask to match hidden states dimensions
        attention_mask = attention_mask.expand_as(hidden_states)
        
        # Apply mask and sum
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        
        # Compute mean, avoiding division by zero
        seq_length = attention_mask.sum(dim=1).clamp(min=1e-9)  # Ensure no division by zero
        mean_hidden = sum_hidden / seq_length
        
        return mean_hidden

    def _shared_step(self, batch):
        """Shared step used in training, validation, and testing."""
        logits = self(batch)
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, preds, labels = self._shared_step(batch)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, preds, labels = self._shared_step(batch)
        
        # Calculate metrics
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        acc = accuracy_score(labels_cpu, preds_cpu)
        f1 = f1_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, preds, labels = self._shared_step(batch)
        
        # Calculate metrics
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        acc = accuracy_score(labels_cpu, preds_cpu)
        f1 = f1_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
        
        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_f1", f1)
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        # Only optimize parameters that require gradients
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate
        )
        return optimizer
