"""Student model for emotion classification.

A simplified implementation of a distilled GRU-based model for emotion classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, Any, Optional


class StudentGRU(pl.LightningModule):
    """Student model for emotion classification using a GRU architecture.
    
    A lightweight model that can be distilled from the Teacher model.
    """
    
    def __init__(
        self, 
        hidden_size: int = 128,
        gru_layers: int = 2,
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
        use_distillation: bool = False,
        distillation_temperature: float = 2.0,
        distillation_alpha: float = 0.5,
        **kwargs
    ):
        """Initialize the model.
        
        Args:
            hidden_size: Dimension of the hidden layer.
            gru_layers: Number of GRU layers.
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
            use_distillation: Whether to use knowledge distillation.
            distillation_temperature: Temperature for softening distribution.
            distillation_alpha: Weight for distillation loss vs hard loss.
        """
        super().__init__()
        
        # Save arguments as attributes
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.audio_input_type = audio_input_type
        self.use_distillation = use_distillation
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
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
            
        # Feature dimensionality reduction
        self.text_projection = nn.Linear(text_feature_dim, hidden_size)
        self.audio_projection = nn.Linear(audio_feature_dim, hidden_size)
        
        # Fusion GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            dropout=dropout_rate if gru_layers > 1 else 0,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_dim)
        )

        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        if use_distillation:
            self.distillation_criterion = nn.KLDivLoss(reduction="batchmean")

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
        if self.audio_input_type == "hf_features" and "audio_features" in batch and batch["audio_features"] is not None:
            # Use pre-extracted HF features directly
            audio_features = batch["audio_features"]
            # Ensure the features have the right dimension
            if audio_features.dim() > 2:
                audio_features = audio_features.squeeze(1)  # Remove sequence dimension if present
        elif "input_values" in batch and batch["input_values"] is not None and self.audio_input_type == "raw_wav":
            # Process raw audio with the audio encoder
            audio_inputs = batch["input_values"]
            audio_attention_mask = batch.get("attention_mask")
            audio_outputs = self.audio_encoder(
                input_values=audio_inputs,
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
            audio_features = torch.zeros(batch_size, self.audio_feature_dim, device=device)
        
        # Project audio features to hidden size
        audio_projected = self.audio_projection(audio_features)

        # Combine modalities for GRU input (batch_size, 2, hidden_size)
        # Using text and audio as a sequence
        multimodal_input = torch.stack([text_projected, audio_projected], dim=1)
        
        # Pass through GRU
        gru_output, _ = self.gru(multimodal_input)
        
        # Use the last hidden state
        last_hidden = gru_output[:, -1, :]
        
        # Pass through classifier
        logits = self.classifier(last_hidden)
        
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
        if attention_mask.dim() < hidden_states.dim():
            attention_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        
        # Apply mask and compute mean
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask

    def _distillation_loss(self, student_logits, teacher_logits, labels):
        """Compute the distillation loss.
        
        Args:
            student_logits: Logits from the student model.
            teacher_logits: Logits from the teacher model.
            labels: Ground truth labels.
            
        Returns:
            Combined loss using both hard and soft targets.
        """
        # Hard loss (standard cross-entropy with ground truth labels)
        hard_loss = self.criterion(student_logits, labels)
        
        # Soft loss (KL divergence between teacher and student logits)
        soft_loss = self.distillation_criterion(
            torch.log_softmax(student_logits / self.distillation_temperature, dim=1),
            torch.softmax(teacher_logits / self.distillation_temperature, dim=1).detach()
        ) * (self.distillation_temperature ** 2)
        
        # Combine losses
        return (1 - self.distillation_alpha) * hard_loss + self.distillation_alpha * soft_loss

    def _shared_step(self, batch):
        """Shared step used in training, validation, and testing."""
        logits = self(batch)
        labels = batch["labels"]
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        # Handle knowledge distillation if enabled
        if self.use_distillation and "teacher_logits" in batch:
            teacher_logits = batch["teacher_logits"]
            T = self.distillation_temperature
            
            # Soften distributions using temperature
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=1)
            soft_predictions = nn.functional.log_softmax(logits / T, dim=1)
            
            # Calculate KL divergence loss
            distillation_loss = self.distillation_criterion(soft_predictions, soft_targets) * (T * T)
            
            # Combine losses
            loss = (1 - self.distillation_alpha) * loss + self.distillation_alpha * distillation_loss
        
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
