"""
Teacher TODKAT-lite model for emotion classification.

This model uses RoBERTa-Large with topic head and COMET triples,
plus a 2-layer encoder-decoder fusion architecture.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, WavLMModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional

from configs.base_config import BaseConfig

class TeacherFusionModel(nn.Module):
    """
    Teacher TODKAT-lite model for emotion classification.
    
    Uses RoBERTa-Large, topic head, COMET triples, and a 2-layer encoder-decoder
    for fusion with WavLM audio features.
    """
    
    def __init__(self, cfg):
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg
        
        # Text Encoder (RoBERTa-Large)
        self.text_encoder = RobertaModel.from_pretrained(cfg.teacher_text_model_name)
        
        # Audio Encoder (WavLM)
        self.audio_encoder = WavLMModel.from_pretrained(cfg.audio_encoder_model_name)
        
        # Topic Head (placeholder - needs implementation)
        # This would extract topic information from text
        self.topic_projector = nn.Linear(cfg.teacher_text_feature_dim, cfg.teacher_topic_dim)
        
        # COMET Head (placeholder - needs implementation)
        # This would process COMET triples or integrate with a COMET model
        self.comet_projector = nn.Linear(cfg.teacher_text_feature_dim, cfg.teacher_comet_dim)
        
        # Calculate fusion input dimension
        fusion_input_dim = (cfg.teacher_text_feature_dim +  # RoBERTa
                           cfg.teacher_topic_dim +         # Topic
                           cfg.teacher_comet_dim +         # COMET
                           cfg.teacher_audio_feature_dim)  # WavLM
        
        # Project to transformer dimension
        self.fusion_input_projector = nn.Linear(fusion_input_dim, cfg.teacher_fusion_hidden_dim)
        
        # Transformer Encoder for fusion
        encoder_layer = TransformerEncoderLayer(
            d_model=cfg.teacher_fusion_hidden_dim,
            nhead=8,  # 8 attention heads
            dim_feedforward=cfg.teacher_fusion_hidden_dim * 4,  # Standard 4x hidden dim
            dropout=0.1,
            batch_first=True
        )
        self.fusion_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=cfg.teacher_fusion_layers
        )
        
        # Classification head
        self.classifier = nn.Linear(cfg.teacher_fusion_hidden_dim, cfg.output_dim)
        
        # Report model info
        self._report_model_info()
    
    def _report_model_info(self):
        """Report model information."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Initialized TeacherFusionModel (TODKAT-lite):")
        print(f"  Text Encoder: {self.cfg.teacher_text_model_name}")
        print(f"  Audio Encoder: {self.cfg.audio_encoder_model_name}")
        print(f"  Topic Dimension: {self.cfg.teacher_topic_dim}")
        print(f"  COMET Dimension: {self.cfg.teacher_comet_dim}")
        print(f"  Fusion Transformer: {self.cfg.teacher_fusion_layers} layers, {self.cfg.teacher_fusion_hidden_dim} hidden dim")
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
        # Get text features from RoBERTa
        text_outputs = self.text_encoder(
            input_ids=batch['text_input_ids'],
            attention_mask=batch.get('text_attention_mask')
        )
        
        # Mean pooling over sequence dimension (or use [CLS] token)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Get topic features (placeholder)
        # In a full implementation, this would use a topic model or extract topic features
        topic_features = self.topic_projector(text_features)
        
        # Get COMET features (placeholder)
        # In a full implementation, this would use COMET to get commonsense features
        comet_features = self.comet_projector(text_features)
        
        # Get audio features
        audio_outputs = self.audio_encoder(
            input_values=batch['audio_input_values'],
            attention_mask=batch.get('audio_attention_mask')
        )
        
        # Mean pooling over sequence dimension
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)
        
        # Concatenate all features
        combined_features = torch.cat([
            text_features,
            topic_features,
            comet_features,
            audio_features
        ], dim=1)
        
        # Project to transformer dimension
        projected_features = self.fusion_input_projector(combined_features)
        
        # Add batch dimension for transformer (if needed)
        if projected_features.dim() == 2:
            projected_features = projected_features.unsqueeze(1)
            
        # Apply transformer encoder
        transformed_features = self.fusion_encoder(projected_features)
        
        # Remove batch dimension (if added)
        if transformed_features.dim() == 3 and transformed_features.size(1) == 1:
            transformed_features = transformed_features.squeeze(1)
            
        # Apply classifier
        logits = self.classifier(transformed_features)
        
        return logits


class TeacherConfig(BaseConfig):
    """Configuration specific to Teacher TODKAT-lite model."""
    
    def __init__(self, architecture_name_override: Optional[str] = None):
        super().__init__(architecture_name_override=architecture_name_override)
        self.architecture_name = "teacher_transformer"
        
        # Model-specific parameters
        self.teacher_text_model_name = "roberta-large"
        self.teacher_text_feature_dim = 1024  # RoBERTa-Large output
        self.teacher_audio_feature_dim = self.audio_feature_dim  # WavLM output
        self.teacher_fusion_hidden_dim = 768  # Fusion transformer hidden dim
        self.teacher_fusion_layers = 2  # Number of transformer layers
        self.teacher_topic_dim = 100  # Topic embedding dimension
        self.teacher_comet_dim = 768  # COMET triples dimension 