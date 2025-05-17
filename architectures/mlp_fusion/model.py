"""
MLP Fusion model for emotion classification.

This model fuses audio and text features using a simple MLP.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from transformers import WhisperModel, WavLMModel, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional, Dict
from pathlib import Path

from configs.base_config import BaseConfig


class MultimodalFusionMLP(pl.LightningModule):
    """
    MLP fusion model for emotion classification from audio and text,
    implemented as a PyTorch LightningModule.
    Uses frozen WavLM for audio and frozen Whisper/RoBERTa for text.
    """
    
    def __init__(self, cfg: BaseConfig): # cfg is expected to be an instance of MLPFusionConfig or BaseConfig
        """
        Initialize the model.
        
        Args:
            cfg: Configuration object (MLPFusionConfig)
        """
        super().__init__()
        self.cfg = cfg
        # self.save_hyperparameters(cfg.__dict__ if hasattr(cfg, '__dict__') else dict(cfg)) # Safely convert cfg to dict
        # More targeted saving of hyperparameters:
        # List relevant attributes from your cfg that define the model architecture and training.
        # This helps with reloading checkpoints and tracking experiments.
        hparams_to_save = {
            attr: getattr(cfg, attr) for attr in [
                'text_encoder_model_name', 'audio_encoder_model_name',
                'text_feature_dim', 'audio_feature_dim', 'mlp_hidden_size',
                'mlp_dropout_rate', 'output_dim', 'learning_rate', 'input_mode', 'dataset_name'
            ] if hasattr(cfg, attr)
        }
        self.save_hyperparameters(hparams_to_save)

        # Text Encoder
        self.text_encoder = self._get_text_encoder(self.cfg.text_encoder_model_name)
        
        # Audio Encoder
        self.audio_encoder = self._get_audio_encoder(self.cfg.audio_encoder_model_name)
        
        # Freeze encoders
        self._freeze_encoder(self.text_encoder)
        self._freeze_encoder(self.audio_encoder)
        
        # Fusion MLP
        fusion_input_dim = self.cfg.text_feature_dim + self.cfg.audio_feature_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, self.cfg.mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.cfg.mlp_dropout_rate),
            nn.Linear(self.cfg.mlp_hidden_size, self.cfg.output_dim)
        )

        self.criterion = nn.CrossEntropyLoss()

        # Metrics - using sklearn.metrics for simplicity now
        # For more integrated PL metrics, consider torchmetrics
        # self.train_accuracy = Accuracy(task="multiclass", num_classes=cfg.output_dim) # Example with torchmetrics
        # self.val_f1 = F1Score(task="multiclass", num_classes=cfg.output_dim, average="weighted") # Example

        self._report_model_info_on_init()

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
            return AutoModel.from_pretrained(model_name)
    
    def _freeze_encoder(self, encoder):
        """
        Freeze encoder parameters.
        
        Args:
            encoder (nn.Module): Encoder to freeze
        """
        for param in encoder.parameters():
            param.requires_grad = False
    
    def _report_model_info_on_init(self):
        """Report model information."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Initialized MultimodalFusionMLP (LightningModule):")
        print(f"  Text Encoder: {self.cfg.text_encoder_model_name}")
        print(f"  Audio Encoder: {self.cfg.audio_encoder_model_name}")
        print(f"  Fusion MLP: {self.cfg.text_feature_dim + self.cfg.audio_feature_dim} -> {self.cfg.mlp_hidden_size} -> {self.cfg.output_dim}")
        print(f"  Trainable Parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass. Batch is expected to be a dictionary from DataLoader.
        """
        # Get text features
        if "whisper" in self.cfg.text_encoder_model_name.lower():
            if 'whisper_input_features' in batch and batch['whisper_input_features'] is not None:
                 # This assumes whisper_input_features is correctly preprocessed by a WhisperProcessor outside
                text_outputs = self.text_encoder.encoder(
                    input_features=batch['whisper_input_features'],
                    attention_mask=batch.get('whisper_attention_mask') 
                )
            elif 'text_input_ids' in batch: # Fallback or primary if Whisper encoder takes input_ids
                text_outputs = self.text_encoder.encoder(
                    input_ids=batch['text_input_ids'],
                    attention_mask=batch.get('text_attention_mask')
                )
            else:
                raise ValueError("Whisper model selected, but neither 'whisper_input_features' nor 'text_input_ids' found in batch.")

        else: # Other text models like RoBERTa
            text_outputs = self.text_encoder(
                input_ids=batch['text_input_ids'],
                attention_mask=batch.get('text_attention_mask')
            )
        
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        # Get audio features
        audio_outputs = self.audio_encoder(
            input_values=batch['audio_input_values'],
            attention_mask=batch.get('audio_attention_mask')
        )
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)
        
        fused_features = torch.cat([text_features, audio_features], dim=1)
        logits = self.fusion_mlp(fused_features)
        return logits

    def _shared_step(self, batch: Dict[str, torch.Tensor]):
        # Ensure the key for ground truth labels matches what DataLoader provides
        # The MELDDataset and collate_fn provide 'emotion_ids' (plural for batch)
        labels = batch.get('emotion_ids') 
        if labels is None:
            # As a fallback, also check for 'emotion_id' (singular) in case batch size is 1 or collate_fn changes
            labels = batch.get('emotion_id')
            if labels is None:
                 # Final fallback to original keys for broader compatibility or if other datasets use them
                labels = batch.get('labels', batch.get('label')) 
                if labels is None:
                    raise KeyError("Batch dictionary must contain 'emotion_ids', 'emotion_id', 'labels', or 'label' for ground truth.")
        
        logits = self(batch) # Calls forward
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        loss, preds, labels = self._shared_step(batch)
        
        # Calculate metrics
        # Ensure labels and preds are on CPU for sklearn metrics
        labels_cpu = labels.cpu().numpy()
        preds_cpu = preds.cpu().numpy()
        
        acc = accuracy_score(labels_cpu, preds_cpu)
        # Using weighted F1, can be configured if needed
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
        self.log('val_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True) # ModelCheckpoint will monitor this if named val_f1
        self.log('val_wf1', f1, on_step=False, on_epoch=True, prog_bar=False, logger=True) # Explicitly log val_wf1 for checkpoint
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
        # Filter parameters that require gradients, e.g., only MLP parameters
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=self.cfg.learning_rate
        )
        # Example of adding a scheduler:
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.cfg.lr_patience)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer


class MLPFusionConfig(BaseConfig):
    """Configuration specific to MLP Fusion model."""
    
    def __init__(self,
                 data_root_override: Optional[Path] = None,
                 output_dir_root_override: Optional[Path] = None,
                 dataset_name: str = "meld",
                 input_mode: str = "audio_text",
                 architecture_name_override: Optional[str] = "mlp_fusion",
                 mlp_hidden_size: int = 256,
                 mlp_dropout_rate: float = 0.3,
                 # Default values for text/audio encoders if not overridden by YAML/CLI
                 text_encoder_model_name: str = "distilroberta-base", # Match common default
                 audio_encoder_model_name: str = "facebook/wav2vec2-base-960h", # Match common default
                 text_feature_dim: int = 768, # Default for distilroberta-base
                 audio_feature_dim: int = 768, # Default for wav2vec2-base
                 **kwargs): 

        super().__init__(
            data_root_override=data_root_override,
            output_dir_root_override=output_dir_root_override,
            dataset_name=dataset_name,
            input_mode=input_mode,
            architecture_name_override=architecture_name_override,
            **kwargs
        )

        self.architecture_name = "mlp_fusion" # Ensure it's correctly set
        
        # Set model-specific parameters, allowing overrides from YAML/CLI via BaseConfig.from_args
        # BaseConfig.from_args will setattr for keys found in YAML or args.
        # getattr here ensures that if they were set by from_args, those values are used,
        # otherwise, the defaults from this __init__ signature are used.
        
        # Encoder model names and dims are crucial for model init, try to get from self first (if set by from_args)
        self.text_encoder_model_name = getattr(self, 'text_encoder_model_name', text_encoder_model_name)
        self.audio_encoder_model_name = getattr(self, 'audio_encoder_model_name', audio_encoder_model_name)
        
        # Feature dimensions should ideally correspond to the chosen encoders.
        # These might be dynamically determined or fixed if encoders are fixed for this config.
        # For MLP Fusion, these are determined by the chosen encoders.
        # These defaults are just illustrative.
        self.text_feature_dim = getattr(self, 'text_feature_dim', text_feature_dim)
        self.audio_feature_dim = getattr(self, 'audio_feature_dim', audio_feature_dim)

        self.mlp_hidden_size = getattr(self, 'mlp_hidden_size', mlp_hidden_size)
        self.mlp_dropout_rate = getattr(self, 'mlp_dropout_rate', mlp_dropout_rate)
        
        # output_dim is typically self.num_classes from BaseConfig, which depends on dataset_name
        # Ensure it's available or set. BaseConfig._update_dataset_specifics() sets num_classes.
        # self.output_dim will be available via self.num_classes after BaseConfig finalization.
        # No need to redefine self.output_dim here typically unless it's different from num_classes.