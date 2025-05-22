"""PaNNs Fusion model for emotion classification.

A simplified implementation that fuses PaNNs audio features with text features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional


class PaNNsFusion(pl.LightningModule):
    """PaNNs Fusion model for emotion classification.
    
    Uses PaNNs audio features and text features with a fusion network.
    """
    
    def __init__(
        self, 
        hidden_size: int = 256,
        dropout_rate: float = 0.3,
        text_encoder_model_name: str = "distilroberta-base",
        use_panns_features: bool = True,
        panns_feature_dim: int = 2048,
        text_feature_dim: int = 768,
        freeze_text_encoder: bool = True,
        output_dim: int = 7,
        learning_rate: float = 1e-4,
        **kwargs
    ):
        """Initialize the model.
        
        Args:
            hidden_size: Dimension of the hidden layer.
            dropout_rate: Dropout rate.
            text_encoder_model_name: Name of the pre-trained text encoder model.
            use_panns_features: Whether to use pre-extracted PaNNs features.
            panns_feature_dim: Dimension of PaNNs audio features.
            text_feature_dim: Dimension of text features.
            freeze_text_encoder: Whether to freeze the text encoder parameters.
            output_dim: Number of output classes.
            learning_rate: Learning rate for optimizer.
        """
        super().__init__()
        
        # Save arguments as attributes
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.text_feature_dim = text_feature_dim
        self.panns_feature_dim = panns_feature_dim
        self.use_panns_features = use_panns_features
        
        # Load text encoder
        self.text_encoder = self._load_text_encoder(text_encoder_model_name)
        if freeze_text_encoder:
            self._freeze_encoder(self.text_encoder)
            
        # Feature dimensionality reduction
        self.text_projection = nn.Linear(text_feature_dim, hidden_size)
        self.audio_projection = nn.Linear(panns_feature_dim, hidden_size)
        
        # Fusion network
        fusion_input_dim = hidden_size * 2  # Concatenated features
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_dim)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _load_text_encoder(self, model_name: str) -> nn.Module:
        """Load text encoder model."""
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

        # Get PaNNs audio features (pre-extracted)
        if self.use_panns_features and "panns_features" in batch:
            panns_features = batch["panns_features"]
            audio_features = panns_features
        else:
            # Create zero tensor if no audio features
            audio_features = torch.zeros(batch_size, self.panns_feature_dim, device=device)
        
        # Project audio features to hidden size
        audio_projected = self.audio_projection(audio_features)

        # Concatenate features
        fused_features = torch.cat([text_projected, audio_projected], dim=1)
        
        # Pass through fusion network
        logits = self.fusion_network(fused_features)
        
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
