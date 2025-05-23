"""ULTRA-VERBOSE DEBUG VERSION OF CONFIG"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

print("🔧 LOADING BASE_CONFIG_DEBUG MODULE")

class Config:
    """Configuration class with ultra-verbose debugging."""
    
    def __init__(self, **kwargs):
        print(f"⚙️ CREATING Config object")
        print(f"   Kwargs: {kwargs}")
        
        # Model architecture
        self.architecture_name = kwargs.get("architecture_name", "mlp_fusion")
        print(f"   Set architecture_name: '{self.architecture_name}' (type: {type(self.architecture_name)})")
        
        # Data settings
        self.data_dir = kwargs.get("data_dir", "data/MELD")
        self.output_dim = kwargs.get("output_dim", 7)
        self.batch_size = kwargs.get("batch_size", 4)
        self.max_sequence_length = kwargs.get("max_sequence_length", 20)
        print(f"   Data settings: data_dir={self.data_dir}, output_dim={self.output_dim}")
        print(f"   Batch settings: batch_size={self.batch_size}, max_seq_len={self.max_sequence_length}")
        
        # Training settings
        self.num_epochs = kwargs.get("num_epochs", 10)
        self.learning_rate = kwargs.get("learning_rate", 2e-4)
        self.weight_decay = kwargs.get("weight_decay", 1e-4)
        self.early_stopping_patience = kwargs.get("early_stopping_patience", 5)
        self.use_mixed_precision = kwargs.get("use_mixed_precision", False)
        self.grad_accumulation_steps = kwargs.get("grad_accumulation_steps", 1)
        self.random_seed = kwargs.get("random_seed", 42)
        print(f"   Training: epochs={self.num_epochs}, lr={self.learning_rate}, wd={self.weight_decay}")
        print(f"   Training: patience={self.early_stopping_patience}, mixed_precision={self.use_mixed_precision}")
        
        # Model-specific settings
        self.mlp_hidden_size = kwargs.get("mlp_hidden_size", 2048)
        self.gru_hidden_size = kwargs.get("gru_hidden_size", 128)
        self.context_window = kwargs.get("context_window", 0)
        print(f"   Model: mlp_hidden={self.mlp_hidden_size}, gru_hidden={self.gru_hidden_size}")
        print(f"   Model: context_window={self.context_window}")
        
        # Loss settings
        self.focal_gamma = kwargs.get("focal_gamma", 2.0)
        self.class_weights = kwargs.get("class_weights", [1.0] * self.output_dim)
        print(f"   Loss: focal_gamma={self.focal_gamma}, class_weights={self.class_weights}")
        
        # Scheduler settings
        self.eta_min = kwargs.get("eta_min", 1e-7)
        print(f"   Scheduler: eta_min={self.eta_min}")
        
        # Fine-tuning settings (optional)
        if "fine_tune" in kwargs:
            print(f"   Fine-tuning settings found: {kwargs['fine_tune']}")
            self.fine_tune = self._create_fine_tune_config(kwargs["fine_tune"])
        else:
            print(f"   No fine-tuning settings")
        
        # Output directories
        self.experiment_name = kwargs.get("experiment_name", f"experiment_{self.architecture_name}")
        self.run_name = kwargs.get("run_name", "run_001")
        self.output_dir = Path(kwargs.get("output_dir", "outputs"))
        print(f"   Experiment: name={self.experiment_name}, run={self.run_name}")
        print(f"   Output dir: {self.output_dir}")
        
        # Derived paths
        self.experiment_output_dir = self.output_dir / self.experiment_name / self.run_name
        self.model_save_dir = self.experiment_output_dir / "models"
        self.logs_dir = self.experiment_output_dir / "logs"
        print(f"   Derived paths:")
        print(f"      experiment_output_dir: {self.experiment_output_dir}")
        print(f"      model_save_dir: {self.model_save_dir}")
        print(f"      logs_dir: {self.logs_dir}")
        
        print(f"   ✅ Config object created")
        
        # Final verification
        print(f"   🔍 FINAL VERIFICATION:")
        print(f"      self.architecture_name = '{self.architecture_name}'")
        print(f"      type(self.architecture_name) = {type(self.architecture_name)}")
        print(f"      repr(self.architecture_name) = {repr(self.architecture_name)}")
    
    def _create_fine_tune_config(self, fine_tune_dict):
        """Create fine-tuning configuration."""
        print(f"   🔧 Creating fine-tune config from: {fine_tune_dict}")
        
        class FineTuneConfig:
            def __init__(self, config_dict):
                self.audio_encoder = self._create_encoder_config(config_dict.get("audio_encoder", {}))
                self.text_encoder = self._create_encoder_config(config_dict.get("text_encoder", {}))
            
            def _create_encoder_config(self, encoder_dict):
                class EncoderConfig:
                    def __init__(self, enc_dict):
                        self.unfreeze_top_n_layers = enc_dict.get("unfreeze_top_n_layers", 0)
                        self.lr_mul = enc_dict.get("lr_mul", 0.1)
                        print(f"         Encoder config: unfreeze={self.unfreeze_top_n_layers}, lr_mul={self.lr_mul}")
                return EncoderConfig(encoder_dict)
        
        return FineTuneConfig(fine_tune_dict)
    
    def create_directories(self):
        """Create output directories."""
        print(f"📁 CREATING DIRECTORIES...")
        print(f"   experiment_output_dir: {self.experiment_output_dir}")
        print(f"   model_save_dir: {self.model_save_dir}")
        print(f"   logs_dir: {self.logs_dir}")
        
        self.experiment_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"   ✅ Directories created")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file with verbose debugging."""
        print(f"📋 LOADING CONFIG FROM YAML: {yaml_path}")
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            error = f"Config file not found: {yaml_path}"
            print(f"   ❌ ERROR: {error}")
            raise FileNotFoundError(error)
        
        print(f"   ✅ File exists: {yaml_path}")
        
        # Read and parse YAML
        with open(yaml_path, 'r') as f:
            raw_content = f.read()
        
        print(f"   📄 Raw YAML content:")
        print("   " + "="*50)
        for i, line in enumerate(raw_content.split('\n'), 1):
            print(f"   {i:2}: {line}")
        print("   " + "="*50)
        
        try:
            yaml_data = yaml.safe_load(raw_content)
            print(f"   ✅ YAML parsed successfully")
            print(f"   📊 Parsed data type: {type(yaml_data)}")
            print(f"   📊 Parsed data: {yaml_data}")
        except yaml.YAMLError as e:
            error = f"Failed to parse YAML: {e}"
            print(f"   ❌ YAML ERROR: {error}")
            raise ValueError(error)
        
        if not isinstance(yaml_data, dict):
            error = f"YAML root must be a dictionary, got {type(yaml_data)}"
            print(f"   ❌ TYPE ERROR: {error}")
            raise ValueError(error)
        
        # Check for architecture_name specifically
        if 'architecture_name' in yaml_data:
            arch_name = yaml_data['architecture_name']
            print(f"   🎯 Found architecture_name: '{arch_name}' (type: {type(arch_name)})")
            print(f"   🎯 architecture_name repr: {repr(arch_name)}")
        else:
            print(f"   ⚠️  architecture_name not found in YAML data")
            print(f"   Available keys: {list(yaml_data.keys())}")
        
        # Create config object
        print(f"   🏗️  Creating Config object with YAML data...")
        config = cls(**yaml_data)
        
        # Verify architecture_name was set correctly
        print(f"   🔍 POST-CREATION VERIFICATION:")
        print(f"      config.architecture_name = '{config.architecture_name}'")
        print(f"      type = {type(config.architecture_name)}")
        print(f"      repr = {repr(config.architecture_name)}")
        
        print(f"   ✅ Config loaded successfully from YAML")
        return config
    
    def save_to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        print(f"💾 SAVING CONFIG TO YAML: {yaml_path}")
        
        # Convert to dictionary (excluding methods and private attrs)
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and not callable(value):
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        
        print(f"   Config dict: {config_dict}")
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"   ✅ Config saved to: {yaml_path}")
    
    def __repr__(self):
        return f"Config(architecture_name='{self.architecture_name}', experiment_name='{self.experiment_name}')"

print("✅ BASE_CONFIG_DEBUG MODULE LOADED") 