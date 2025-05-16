"""
MLP Fusion model for emotion classification.

This model fuses audio and text features using a simple MLP.
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WavLMModel
from typing import Optional

from configs.base_config import BaseConfig


class MultimodalFusionMLP(nn.Module):
    """
    Simple MLP fusion model for emotion classification from audio and text.
    Uses frozen WavLM for audio and frozen Whisper/RoBERTa for text.
    """
    
    def __init__(self, cfg):
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg
        
        # Text Encoder
        self.text_encoder = self._get_text_encoder(cfg.text_encoder_model_name)
        
        # Audio Encoder
        self.audio_encoder = self._get_audio_encoder(cfg.audio_encoder_model_name)
        
        # Freeze encoders
        self._freeze_encoder(self.text_encoder)
        self._freeze_encoder(self.audio_encoder)
        
        # Fusion MLP
        # Input: Concatenated text and audio features
        fusion_input_dim = cfg.text_feature_dim + cfg.audio_feature_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.mlp_dropout_rate),
            nn.Linear(cfg.mlp_hidden_dim, cfg.output_dim)
        )
        
        # Report model info
        self._report_model_info()
    
    def _get_text_encoder(self, model_name):
        """
        Load text encoder.
        
        Args:
            model_name (str): Model name or path
            
        Returns:
            nn.Module: Text encoder model
        """
        print(f"Loading text encoder: {model_name}")
        
        if "whisper" in model_name.lower():
            return WhisperModel.from_pretrained(model_name)
        else:
            # Default to AutoModel for RoBERTa, DistilBERT, etc.
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_name)
    
    def _get_audio_encoder(self, model_name):
        """
        Load audio encoder.
        
        Args:
            model_name (str): Model name or path
            
        Returns:
            nn.Module: Audio encoder model
        """
        print(f"Loading audio encoder: {model_name}")
        
        if "wavlm" in model_name.lower():
            return WavLMModel.from_pretrained(model_name)
        else:
            # For other audio models like Wav2Vec2
            from transformers import AutoModel
            return AutoModel.from_pretrained(model_name)
    
    def _freeze_encoder(self, encoder):
        """
        Freeze encoder parameters.
        
        Args:
            encoder (nn.Module): Encoder to freeze
        """
        for param in encoder.parameters():
            param.requires_grad = False
    
    def _report_model_info(self):
        """Report model information."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Initialized MultimodalFusionMLP:")
        print(f"  Text Encoder: {self.cfg.text_encoder_model_name}")
        print(f"  Audio Encoder: {self.cfg.audio_encoder_model_name}")
        print(f"  Fusion MLP: {self.cfg.text_feature_dim + self.cfg.audio_feature_dim} → {self.cfg.mlp_hidden_dim} → {self.cfg.output_dim}")
        print(f"  Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch (dict): Batch dictionary containing:
                - text_input_ids: [batch_size, seq_len]
                - text_attention_mask: [batch_size, seq_len]
                - audio_input_values: [batch_size, audio_len]
                - audio_attention_mask: [batch_size, audio_len]
                
        Returns:
            torch.Tensor: Logits [batch_size, num_classes]
        """
        # Get text features
        if "whisper" in self.cfg.text_encoder_model_name.lower():
            # Whisper takes input_features preprocessed by WhisperProcessor
            if hasattr(batch, 'whisper_input_features'):
                text_outputs = self.text_encoder.encoder(
                    input_features=batch['whisper_input_features'],
                    attention_mask=batch.get('whisper_attention_mask')
                )
            else:
                # Use text_input_ids from tokenizer instead
                text_outputs = self.text_encoder.encoder(
                    input_ids=batch['text_input_ids'],
                    attention_mask=batch.get('text_attention_mask')
                )
        else:
            # Other text models like RoBERTa
            text_outputs = self.text_encoder(
                input_ids=batch['text_input_ids'],
                attention_mask=batch.get('text_attention_mask')
            )
        
        # Mean pooling over sequence dimension
        if hasattr(text_outputs, 'last_hidden_state'):
            text_features = text_outputs.last_hidden_state.mean(dim=1)
        else:
            text_features = text_outputs[0].mean(dim=1)
        
        # Get audio features
        audio_outputs = self.audio_encoder(
            input_values=batch['audio_input_values'],
            attention_mask=batch.get('audio_attention_mask')
        )
        
        # Mean pooling over sequence dimension
        if hasattr(audio_outputs, 'last_hidden_state'):
            audio_features = audio_outputs.last_hidden_state.mean(dim=1)
        else:
            audio_features = audio_outputs[0].mean(dim=1)
        
        # Concatenate features
        fused_features = torch.cat([text_features, audio_features], dim=1)
        
        # Apply MLP
        logits = self.fusion_mlp(fused_features)
        
        return logits


class MLPFusionConfig(BaseConfig):
    """Configuration specific to MLP Fusion model."""
    
    def __init__(self, architecture_name_override: Optional[str] = None):
        super().__init__(architecture_name_override=architecture_name_override)
        self.architecture_name = "mlp_fusion"
        
        # Model-specific parameters
        self.mlp_hidden_dim = 256
        self.mlp_dropout_rate = 0.3 