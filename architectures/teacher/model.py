"""
Teacher TODKAT-lite model for emotion classification.

This model uses RoBERTa-Large with topic head and COMET triples,
plus a 2-layer encoder-decoder fusion architecture.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from transformers import RobertaModel, WavLMModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional, Dict
from pathlib import Path

from configs.base_config import BaseConfig

class TeacherFusionModel(pl.LightningModule):
    """
    Teacher TODKAT-lite model for emotion classification (LightningModule).
    """
    
    def __init__(self, cfg: BaseConfig):
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object (TeacherConfig)
        """
        super().__init__()
        self.cfg = cfg
        
        # Save hyperparameters - adapt attributes from TeacherConfig
        hparams_to_save = {
            attr: getattr(cfg, attr) for attr in [
                'teacher_text_model_name', 'audio_encoder_model_name', 
                'teacher_text_feature_dim', 'teacher_audio_feature_dim', 
                'teacher_fusion_hidden_dim', 'teacher_fusion_layers',
                'teacher_topic_dim', 'teacher_comet_dim',
                'output_dim', 'learning_rate', 'input_mode', 'dataset_name'
            ] if hasattr(cfg, attr)
        }
        self.save_hyperparameters(hparams_to_save)
        
        # Text Encoder (RoBERTa-Large)
        self.text_encoder = RobertaModel.from_pretrained(self.cfg.teacher_text_model_name)
        
        # Audio Encoder (WavLM or other specified in cfg)
        self.audio_encoder = WavLMModel.from_pretrained(self.cfg.audio_encoder_model_name)
        
        # Topic Head (placeholder - needs implementation or removal if not used)
        # If these are just linear layers, they are fine. If they need complex logic,
        # they should be separate modules.
        self.topic_projector = nn.Linear(self.cfg.teacher_text_feature_dim, self.cfg.teacher_topic_dim)
        
        # COMET Head (placeholder - needs implementation or removal if not used)
        self.comet_projector = nn.Linear(self.cfg.teacher_text_feature_dim, self.cfg.teacher_comet_dim)
        
        # Calculate fusion input dimension
        fusion_input_dim = (self.cfg.teacher_text_feature_dim +
                           self.cfg.teacher_topic_dim +
                           self.cfg.teacher_comet_dim +
                           self.cfg.teacher_audio_feature_dim)
        
        self.fusion_input_projector = nn.Linear(fusion_input_dim, self.cfg.teacher_fusion_hidden_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=self.cfg.teacher_fusion_hidden_dim,
            nhead=getattr(self.cfg, 'teacher_fusion_nhead', 8), # Use getattr for new hparam
            dim_feedforward=self.cfg.teacher_fusion_hidden_dim * getattr(self.cfg, 'teacher_fusion_ff_expansion', 4),
            dropout=getattr(self.cfg, 'teacher_fusion_dropout', 0.1),
            batch_first=True
        )
        self.fusion_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.teacher_fusion_layers
        )
        
        self.classifier = nn.Linear(self.cfg.teacher_fusion_hidden_dim, self.cfg.output_dim)

        self.criterion = nn.CrossEntropyLoss()
        
        self._report_model_info_on_init() # Call after all components are defined
    
    def _report_model_info_on_init(self): # Renamed for clarity
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Initialized TeacherFusionModel (LightningModule):")
        print(f"  Text Encoder: {self.cfg.teacher_text_model_name}")
        print(f"  Audio Encoder: {self.cfg.audio_encoder_model_name}")
        print(f"  Topic Dimension: {self.cfg.teacher_topic_dim}")
        print(f"  COMET Dimension: {self.cfg.teacher_comet_dim}")
        print(f"  Fusion Transformer: {self.cfg.teacher_fusion_layers} layers, {self.cfg.teacher_fusion_hidden_dim} hidden dim")
        print(f"  Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        """
        text_outputs = self.text_encoder(
            input_ids=batch['text_input_ids'],
            attention_mask=batch.get('text_attention_mask')
        )
        text_features = text_outputs.last_hidden_state[:, 0, :] 
        
        topic_features = self.topic_projector(text_features)
        comet_features = self.comet_projector(text_features)
        
        audio_outputs = self.audio_encoder(
            input_values=batch['audio_input_values'],
            attention_mask=batch.get('audio_attention_mask')
        )
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)
        
        combined_features = torch.cat([
            text_features, topic_features, comet_features, audio_features
        ], dim=1)
        
        projected_features = self.fusion_input_projector(combined_features)
        
        if projected_features.dim() == 2:
            projected_features = projected_features.unsqueeze(1)
            
        transformed_features = self.fusion_encoder(projected_features)
        
        if transformed_features.dim() == 3 and transformed_features.size(1) == 1:
            transformed_features = transformed_features.squeeze(1)
            
        logits = self.classifier(transformed_features)
        return logits

    def _shared_step(self, batch: Dict[str, torch.Tensor]):
        labels = batch.get('labels', batch.get('label'))
        if labels is None:
            raise KeyError("Batch dictionary must contain 'labels' or 'label' for ground truth.")
        
        logits = self(batch)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self._shared_step(batch)
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        acc = accuracy_score(labels_cpu, preds_cpu)
        f1 = f1_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self._shared_step(batch)
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        acc = accuracy_score(labels_cpu, preds_cpu)
        f1 = f1_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_wf1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True) # For checkpoint
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self._shared_step(batch)
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        acc = accuracy_score(labels_cpu, preds_cpu)
        f1 = f1_score(labels_cpu, preds_cpu, average='weighted', zero_division=0)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.cfg.learning_rate # Ensure learning_rate is in TeacherConfig or BaseConfig
        )
        return optimizer


class TeacherConfig(BaseConfig):
    """Configuration specific to Teacher TODKAT-lite model."""
    
    def __init__(self,
                 data_root_override: Optional[Path] = None,
                 output_dir_root_override: Optional[Path] = None,
                 dataset_name: str = "meld",
                 input_mode: str = "audio_text",
                 architecture_name_override: Optional[str] = "teacher_transformer",
                 # Teacher-specific parameters
                 teacher_text_model_name: str = "roberta-large",
                 teacher_text_feature_dim: int = 1024, # For roberta-large
                 # audio_encoder_model_name & audio_feature_dim are inherited from BaseConfig defaults or YAML
                 # If Teacher needs specific *different* defaults for audio, define them here.
                 teacher_fusion_hidden_dim: int = 768,
                 teacher_fusion_layers: int = 2,
                 teacher_fusion_nhead: int = 8, 
                 teacher_fusion_ff_expansion: int = 4, 
                 teacher_fusion_dropout: float = 0.1, 
                 teacher_topic_dim: int = 100,
                 teacher_comet_dim: int = 768, 
                 # learning_rate is inherited from BaseConfig defaults or YAML
                 **kwargs):

        super().__init__(
            data_root_override=data_root_override,
            output_dir_root_override=output_dir_root_override,
            dataset_name=dataset_name,
            input_mode=input_mode,
            architecture_name_override=architecture_name_override,
            **kwargs
        )

        self.architecture_name = "teacher_transformer"
        
        # Teacher-specific attributes are set here.
        # Attributes like audio_encoder_model_name, audio_feature_dim, learning_rate are 
        # already set on `self` by super().__init__() using BaseConfig defaults, 
        # which are then potentially overridden by YAML or CLI args via BaseConfig.from_args().
        # Only use getattr here if TeacherConfig defines its OWN default for a parameter that might also be in BaseConfig,
        # and you want TeacherConfig's default to take precedence if YAML/CLI doesn't specify it.

        self.teacher_text_model_name = getattr(self, 'teacher_text_model_name', teacher_text_model_name)
        self.teacher_text_feature_dim = getattr(self, 'teacher_text_feature_dim', teacher_text_feature_dim)
        
        # For audio features, self.audio_encoder_model_name and self.audio_feature_dim are already populated by BaseConfig.
        # If Teacher model specifically needs a different audio feature dimension based on its fusion logic,
        # distinct from the direct output of the base audio encoder, define a new attribute e.g.,
        # self.teacher_specific_audio_dim = getattr(self, 'teacher_specific_audio_dim', self.audio_feature_dim)
        # For now, we assume the Teacher model uses self.audio_feature_dim directly from BaseConfig.
        # Let's add teacher_audio_feature_dim for clarity if it can be distinct for the teacher model
        self.teacher_audio_feature_dim = getattr(self, 'teacher_audio_feature_dim', self.audio_feature_dim)

        self.teacher_fusion_hidden_dim = getattr(self, 'teacher_fusion_hidden_dim', teacher_fusion_hidden_dim)
        self.teacher_fusion_layers = getattr(self, 'teacher_fusion_layers', teacher_fusion_layers)
        self.teacher_fusion_nhead = getattr(self, 'teacher_fusion_nhead', teacher_fusion_nhead)
        self.teacher_fusion_ff_expansion = getattr(self, 'teacher_fusion_ff_expansion', teacher_fusion_ff_expansion)
        self.teacher_fusion_dropout = getattr(self, 'teacher_fusion_dropout', teacher_fusion_dropout)
        self.teacher_topic_dim = getattr(self, 'teacher_topic_dim', teacher_topic_dim)
        self.teacher_comet_dim = getattr(self, 'teacher_comet_dim', teacher_comet_dim)
        
        # self.learning_rate is set by BaseConfig. No need to re-set with DEFAULT_LEARNING_RATE here.
        # self.output_dim is a property in BaseConfig (self.num_classes). No need to set it here. 