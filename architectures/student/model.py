"""
Student Distilled model for emotion classification.

This model uses DistilRoBERTa with WavLM-Base and GRU party-tracker,
designed to be distilled from the Teacher model.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, WavLMModel
from typing import Optional

from configs.base_config import BaseConfig

class StudentDistilledModel(nn.Module):
    """
    Student Distilled model for emotion classification.
    
    Uses DistilRoBERTa, WavLM-Base, and GRU party-tracker
    with multi-task learning heads.
    """
    
    def __init__(self, cfg):
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object
        """
        super().__init__()
        self.cfg = cfg
        
        # Text Encoder (DistilRoBERTa)
        self.text_encoder = AutoModel.from_pretrained(cfg.student_text_model_name)
        
        # Audio Encoder (WavLM)
        self.audio_encoder = WavLMModel.from_pretrained(cfg.audio_encoder_model_name)
        
        # Speaker/Party Embedding
        # This would be populated with speaker IDs from the MELD dataset
        # Assuming a fixed number of speakers for simplicity
        self.num_speakers = 20  # MELD has approximately ~400 speakers, but many are rare
        self.speaker_embedding = nn.Embedding(
            num_embeddings=self.num_speakers,
            embedding_dim=cfg.student_party_embed_dim
        )
        
        # GRU Party Tracker for dialogue context
        # Input: Current text + audio features + previous speaker embedding
        gru_input_dim = (cfg.student_text_feature_dim +  # Text
                        cfg.student_audio_feature_dim +  # Audio
                        cfg.student_party_embed_dim)     # Speaker embedding
        
        self.party_tracker = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=cfg.student_gru_hidden_dim,
            num_layers=cfg.student_gru_layers,
            batch_first=True
        )
        
        # Fusion layer
        fusion_dim = (cfg.student_text_feature_dim +  # Text
                     cfg.student_audio_feature_dim +  # Audio
                     cfg.student_gru_hidden_dim)      # GRU hidden state
        
        # Multi-task learning heads
        # Primary task: Emotion classification
        self.emotion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg.student_gru_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.student_gru_hidden_dim, cfg.output_dim)
        )
        
        # Auxiliary task: Speaker prediction (optional)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg.student_gru_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.student_gru_hidden_dim, self.num_speakers)
        )
        
        # Report model info
        self._report_model_info()
    
    def _report_model_info(self):
        """Report model information."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Initialized StudentDistilledModel:")
        print(f"  Text Encoder: {self.cfg.student_text_model_name}")
        print(f"  Audio Encoder: {self.cfg.audio_encoder_model_name}")
        print(f"  GRU Party Tracker: {self.cfg.student_gru_layers} layers, {self.cfg.student_gru_hidden_dim} hidden dim")
        print(f"  Speaker Embedding: {self.num_speakers} speakers, {self.cfg.student_party_embed_dim} dim")
        print(f"  Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    def forward(self, batch, hidden_state=None):
        """
        Forward pass.
        
        Args:
            batch (dict): Batch dictionary containing:
                - text_input_ids: [batch_size, seq_len]
                - text_attention_mask: [batch_size, seq_len]
                - audio_input_values: [batch_size, audio_len]
                - audio_attention_mask: [batch_size, audio_len]
                - speaker_ids (optional): [batch_size]
            hidden_state (torch.Tensor, optional): Previous GRU hidden state
                
        Returns:
            dict: Dictionary with:
                - emotion_logits: Emotion classification logits
                - speaker_logits: Speaker classification logits
                - hidden_state: Updated GRU hidden state
        """
        # Get text features
        text_outputs = self.text_encoder(
            input_ids=batch['text_input_ids'],
            attention_mask=batch.get('text_attention_mask')
        )
        
        # Mean pooling over sequence dimension
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        # Get audio features
        audio_outputs = self.audio_encoder(
            input_values=batch['audio_input_values'],
            attention_mask=batch.get('audio_attention_mask')
        )
        
        # Mean pooling over sequence dimension
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)
        
        # Get speaker embeddings (if available)
        if 'speaker_ids' in batch:
            speaker_ids = batch['speaker_ids']
            speaker_features = self.speaker_embedding(speaker_ids)
        else:
            # Use a default embedding if speaker info not available
            batch_size = text_features.size(0)
            speaker_features = torch.zeros(
                batch_size,
                self.cfg.student_party_embed_dim,
                device=text_features.device
            )
        
        # Concatenate features for GRU input
        gru_input = torch.cat([
            text_features,
            audio_features,
            speaker_features
        ], dim=1)
        
        # Reshape for GRU (add sequence dimension if needed)
        if gru_input.dim() == 2:
            gru_input = gru_input.unsqueeze(1)  # [batch_size, 1, feature_dim]
            
        # Process with GRU party tracker
        if hidden_state is None:
            # Initialize hidden state
            batch_size = gru_input.size(0)
            hidden_state = torch.zeros(
                self.cfg.student_gru_layers,
                batch_size,
                self.cfg.student_gru_hidden_dim,
                device=gru_input.device
            )
            
        # Forward pass through GRU
        gru_output, hidden_state = self.party_tracker(gru_input, hidden_state)
        
        # Get final GRU output
        gru_features = gru_output[:, -1, :]  # Last time step
        
        # Concatenate all features for fusion
        fused_features = torch.cat([
            text_features,
            audio_features,
            gru_features
        ], dim=1)
        
        # Multi-task heads
        emotion_logits = self.emotion_classifier(fused_features)
        speaker_logits = self.speaker_classifier(fused_features)
        
        return {
            'emotion_logits': emotion_logits,
            'speaker_logits': speaker_logits,
            'hidden_state': hidden_state
        }
    
    def distill_from_teacher(self, teacher_logits, alpha=0.5, temperature=2.0):
        """
        Compute distillation loss.
        
        Args:
            teacher_logits (torch.Tensor): Logits from teacher model
            alpha (float): Weight for distillation loss (vs. hard loss)
            temperature (float): Temperature for softening probabilities
            
        Returns:
            torch.Tensor: Distillation loss
        """
        # Not fully implemented - placeholder for the distillation mechanism
        # This would be called by the training loop
        pass


class StudentConfig(BaseConfig):
    """Configuration specific to Student Distilled model."""
    
    def __init__(self, architecture_name_override: Optional[str] = None):
        super().__init__(architecture_name_override=architecture_name_override)
        self.architecture_name = "student_distilled_gru"
        
        # Model-specific parameters
        self.student_text_model_name = "distilroberta-base"
        self.student_text_feature_dim = 768  # DistilRoBERTa output
        self.student_audio_feature_dim = self.audio_feature_dim  # WavLM output
        self.student_gru_hidden_dim = 256  # GRU hidden dimension
        self.student_gru_layers = 1  # Number of GRU layers
        self.student_party_embed_dim = 64  # Speaker embedding dimension 