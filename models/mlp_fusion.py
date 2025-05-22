"""MLP Fusion model for emotion classification.

A simplified implementation that fuses audio and text features using a simple MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional


class MultimodalFusionMLP(pl.LightningModule):
    """MLP fusion model for emotion classification from audio and text.
    
    Uses pre-trained encoders for audio and text, with their outputs fused through an MLP.
    """    
    def __init__(
        self, 
        mlp_hidden_size: int = 256,
        mlp_dropout_rate: float = 0.3,
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
            mlp_hidden_size: Dimension of the hidden layer in the MLP.
            mlp_dropout_rate: Dropout rate for the MLP.
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
        
        # Load audio processor if using raw audio
        self.audio_processor = None
        if audio_input_type == "raw_wav":
            try:
                self.audio_processor = AutoProcessor.from_pretrained(audio_encoder_model_name)
            except Exception as e:
                print(f"Error loading audio processor: {e}")
        
        # Fusion MLP
        fusion_input_dim = text_feature_dim + audio_feature_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(mlp_dropout_rate),
            nn.Linear(mlp_hidden_size, output_dim)
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
        # Text encoding
        text_inputs = batch.get("text_input_ids")
        text_attention_mask = batch.get("text_attention_mask")
        if text_inputs is not None:
            text_outputs = self.text_encoder(
                input_ids=text_inputs,
                attention_mask=text_attention_mask,
                return_dict=True
            )
            # Use pooled output or mean of last hidden state
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
            device = self.device
            batch_size = next(iter(batch.values())).size(0)
            text_features = torch.zeros(batch_size, self.text_feature_dim, device=device)

        # Audio encoding
        if self.audio_input_type == "raw_wav" and "raw_audio" in batch:
            # Process raw audio with audio_processor
            if self.audio_processor is not None:
                audio_inputs = self.audio_processor(
                    batch["raw_audio"], 
                    sampling_rate=batch.get("sampling_rate", 16000),
                    return_tensors="pt"
                ).to(self.device)
                audio_outputs = self.audio_encoder(**audio_inputs, return_dict=True)
                audio_features = audio_outputs.last_hidden_state.mean(dim=1)  # Mean pooling
            else:
                # Fallback if processor not available
                device = self.device
                batch_size = next(iter(batch.values())).size(0)
                audio_features = torch.zeros(batch_size, self.audio_feature_dim, device=device)
        elif "audio_input_values" in batch:
            # Use pre-extracted features
            audio_inputs = batch["audio_input_values"]
            audio_attention_mask = batch.get("audio_attention_mask")
            # Handle differently based on whether we're using features or raw waveform
            if self.audio_input_type == "raw_wav":
                audio_outputs = self.audio_encoder(
                    input_values=audio_inputs,
                    attention_mask=audio_attention_mask,
                    return_dict=True
                )
            else:
                # For pre-computed features, we need to adapt to expected format
                audio_outputs = self.audio_encoder(
                    input_values=audio_inputs.squeeze(1) if audio_inputs.dim() > 2 else audio_inputs,
                    attention_mask=audio_attention_mask,
                    return_dict=True
                )
            # Mean pooling of the last hidden state
            audio_features = self._mean_pooling(
                audio_outputs.last_hidden_state,
                audio_attention_mask
            )
        else:
            # Create zero tensor if no audio input
            device = self.device
            batch_size = next(iter(batch.values())).size(0)
            audio_features = torch.zeros(batch_size, self.audio_feature_dim, device=device)

        # Concatenate features
        fused_features = torch.cat([text_features, audio_features], dim=1)
        
        # Pass through MLP
        logits = self.fusion_mlp(fused_features)
        
        return logits

    def _mean_pooling(self, hidden_states, attention_mask=None):
        """Mean pooling of hidden states with attention mask."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        # Expand attention mask to match hidden states dimensions
        attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        
        # Apply mask and compute mean over sequence length
        masked_hidden = hidden_states * attention_mask.float()
        sum_embeddings = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.float().sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Prevent division by zero
        
        return sum_embeddings / sum_mask

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
