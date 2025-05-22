"""Simplified configuration for emotion classification.

This module provides a streamlined configuration system using dataclasses.
It replaces the complex BaseConfig hierarchy with a simple, flat structure.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Any


@dataclass
class Config:
    """Central configuration class for emotion classification project."""
    # Project and experiment settings
    experiment_name: str = "default_experiment"
    run_name: Optional[str] = None
    random_seed: int = 42
    
    # Dataset settings
    dataset_name: str = "meld"
    input_mode: str = "audio_text"  # audio_text, audio_only, text_only
    
    # Path settings
    data_root: Path = field(default_factory=lambda: Path("meld_data"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    
    # Model architecture settings
    architecture_name: str = "mlp_fusion"  # mlp_fusion, teacher, student, panns_fusion
    
    # Audio settings
    audio_encoder_model_name: str = "facebook/wav2vec2-base-960h"
    audio_feature_dim: int = 768
    sampling_rate: int = 16000
    max_audio_duration_seconds: float = 15.0
    audio_input_type: str = "hf_features"  # raw_wav, hf_features
    
    # Text settings
    text_encoder_model_name: str = "distilroberta-base"
    text_feature_dim: int = 768
    text_max_length: int = 128
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 20
    early_stopping_patience: int = 5
    optimizer_name: str = "AdamW"
    grad_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    
    # MLP Fusion specific settings
    mlp_hidden_size: int = 256
    mlp_dropout_rate: float = 0.3
    freeze_text_encoder: bool = True
    freeze_audio_encoder: bool = True
    
    # Output dimensions (emotion classes)
    output_dim: int = 7  # MELD has 7 emotion classes
    
    # Data preprocessing flags
    force_wav_conversion: bool = False
    force_reprocess_hf_dataset: bool = False
    limit_dialogues_train: Optional[int] = None
    limit_dialogues_dev: Optional[int] = None
    limit_dialogues_test: Optional[int] = None
    
    def __post_init__(self):
        """Perform post-initialization setup."""
        # Convert string paths to Path objects
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        # Create absolute paths
        self.data_root = self.data_root.absolute()
        self.output_dir = self.output_dir.absolute()
        
        # Set derived paths - use direct paths without dataset_name prefix
        self.raw_data_dir = self.data_root / "raw"
        self.processed_data_dir = self.data_root / "processed"
        self.processed_audio_dir = self.processed_data_dir / "audio_16kHz_mono"  # Match actual directory name
        self.processed_features_dir = self.processed_data_dir / "features"
        self.hf_dataset_dir = self.processed_features_dir / "hf_datasets"
        
        # Set experiment output directories
        self.experiment_output_dir = self.output_dir / self.experiment_name / self.architecture_name
        self.model_save_dir = self.experiment_output_dir / "checkpoints"
        self.logs_dir = self.experiment_output_dir / "logs"
        
        # Set device automatically
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Determine the best available device for training."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # For Apple Silicon
        else:
            return torch.device("cpu")
    
    def create_directories(self):
        """Create all necessary directories."""
        # Data directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_audio_dir.mkdir(parents=True, exist_ok=True)
        self.processed_features_dir.mkdir(parents=True, exist_ok=True)
        self.hf_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Output directories
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config instance from a dictionary."""
        # Filter out keys that aren't part of the dataclass
        valid_keys = {f.name for f in fields(cls)}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Create a Config instance from a YAML file."""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Config to a dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
    
    def save_to_yaml(self, yaml_path: str):
        """Save Config to a YAML file."""
        import yaml
        # Convert Path objects to strings for YAML serialization
        config_dict = self.to_dict()
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
