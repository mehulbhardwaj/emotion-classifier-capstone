### common/config.py content starts ###
"""
Base configuration for emotion classification project.
Contains common settings for all architectures.
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml # Added for YAML loading
import argparse

# --- Base Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# SINGLE DATA ROOT for the active dataset (e.g., MELD)
# This can be changed if you switch to a new dataset by modifying this global or using an env var.
# For simplicity in this refactor, we'll assume it's set here.
# A more dynamic approach might involve passing dataset_name to determine this path.
ACTIVE_DATASET_NAME_FOR_PATHS = "meld" # Global for defining paths, BaseConfig instance will use its own dataset_name for logic
DATASET_ROOT = PROJECT_ROOT / f"{ACTIVE_DATASET_NAME_FOR_PATHS}_data" # e.g., project_root/meld_data

RAW_DATA_DIR_BASE = DATASET_ROOT / "raw" # For both CSVs and raw media like videos
PROCESSED_DATA_DIR_BASE = DATASET_ROOT / "processed" # For audio, features

# These are specific to the ACTIVE_DATASET_NAME_FOR_PATHS for clarity at module level
# BaseConfig instances will use properties derived from their own self.dataset_name
DEFAULT_PROCESSED_AUDIO_BASE_DIR = PROCESSED_DATA_DIR_BASE / "audio"
DEFAULT_FEATURES_BASE_DIR = PROCESSED_DATA_DIR_BASE / "features"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Default Feature Extraction Models & Parameters (can be overridden) ---
DEFAULT_TEXT_ENCODER_MODEL_NAME = "openai/whisper-base" 
DEFAULT_TEXT_FEATURE_DIM = 512
DEFAULT_AUDIO_ENCODER_MODEL_NAME = "microsoft/wavlm-base-plus"
DEFAULT_AUDIO_FEATURE_DIM = 768
DEFAULT_ASR_MODEL_NAME = "openai/whisper-base" # For ASR tasks

DEFAULT_SAMPLE_RATE = 16000

# --- Default Training Hyperparameters (can be overridden) ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_MAX_SEQ_LENGTH_TEXT = 128
DEFAULT_MAX_AUDIO_DURATION_SECONDS = 15

# --- Default Technical Settings (can be overridden) ---
DEFAULT_DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_DATALOADER_WORKERS = min(os.cpu_count(), 4)
DEFAULT_MIXED_PRECISION_TRAINING = True

# --- Default Data Preparation Switches ---
DEFAULT_RUN_MP4_TO_WAV_CONVERSION = True
DEFAULT_RUN_HF_DATASET_CREATION = True
DEFAULT_FFMPEG_PATH = "ffmpeg"

# --- Default Audio Feature Extraction Parameters ---
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512  # samples
DEFAULT_WIN_LENGTH = 2048 # samples
DEFAULT_N_MELS = 128      # Number of Mel bands
DEFAULT_F_MIN = 0.0       # Minimum Hz frequency
DEFAULT_F_MAX = None      # Maximum Hz frequency (None defaults to sample_rate / 2)
DEFAULT_POWER = 2.0       # Exponent for the magnitude spectrogram
DEFAULT_NORMALIZE_SPECTROGRAM = True # Custom flag, not a direct torchaudio param
DEFAULT_MEL_SCALE = "htk" # "htk" or "slaney" for torchaudio

# --- Default CSV Column Maps (Extensible for other datasets) ---
DEFAULT_CSV_COLUMN_MAPS = {
    "meld": {
        "text_col": "Utterance", 
        "dialogue_id_col": "Dialogue_ID", 
        "utterance_id_col": "Utterance_ID", 
        "emotion_col": "Emotion",
        "audio_path_col": "audio_path" 
    },
    "iemocap": { # Example for another dataset
        "text_col": "transcription", 
        "dialogue_id_col": "session_id",
        "utterance_id_col": "utt_id", 
        "emotion_col": "emotion_label",
        "audio_path_col": "audio_file_path" 
    }
}

class BaseConfig:
    """Base configuration class that can be extended by architecture-specific configs."""
    
    def __init__(self, dataset_name: str = "meld", input_mode: str = "audio_text", architecture_name_override: Optional[str] = None):
        """
        Args:
            dataset_name (str): Name of the dataset (e.g., "meld", "iemocap").
            input_mode (str): Input modality combination 
                              (e.g., "audio_text", "audio_only_asr", "video_audio_text").
            architecture_name_override (Optional[str]): Overrides the architecture name if provided (e.g. from CLI).
        """
        self.dataset_name = dataset_name.lower()
        self.input_mode = input_mode.lower()
        
        # Determine architecture name: CLI override > subclass default > base default
        if architecture_name_override:
            self.architecture_name = architecture_name_override
        elif not hasattr(self, 'architecture_name'): # If subclass hasn't set it yet
            self.architecture_name = "base" 
        # If hasattr(self, 'architecture_name') is True, it means a subclass already set it, so we keep that.

        self.project_root = PROJECT_ROOT
        # self.dataset_data_root is now a property based on self.dataset_name

        self.models_dir = MODELS_DIR
        self.results_dir = RESULTS_DIR
        
        self.text_encoder_model_name = DEFAULT_TEXT_ENCODER_MODEL_NAME
        self.text_feature_dim = DEFAULT_TEXT_FEATURE_DIM
        self.audio_encoder_model_name = DEFAULT_AUDIO_ENCODER_MODEL_NAME
        self.audio_feature_dim = DEFAULT_AUDIO_FEATURE_DIM
        self.sample_rate = DEFAULT_SAMPLE_RATE
        
        self.asr_model_name = DEFAULT_ASR_MODEL_NAME
        self.load_asr_model_on_init = (self.input_mode == "audio_only_asr")

        self.batch_size = DEFAULT_BATCH_SIZE
        self.learning_rate = DEFAULT_LEARNING_RATE
        self.num_epochs = DEFAULT_NUM_EPOCHS
        self.early_stopping_patience = DEFAULT_EARLY_STOPPING_PATIENCE
        self.max_seq_length_text = DEFAULT_MAX_SEQ_LENGTH_TEXT
        self.max_audio_duration_seconds = DEFAULT_MAX_AUDIO_DURATION_SECONDS

        self.device_name = DEFAULT_DEVICE_NAME
        self.device = torch.device(self.device_name)
        self.random_seed = DEFAULT_RANDOM_SEED
        self.num_dataloader_workers = DEFAULT_NUM_DATALOADER_WORKERS
        self.mixed_precision_training = DEFAULT_MIXED_PRECISION_TRAINING
        
        self.run_mp4_to_wav_conversion = DEFAULT_RUN_MP4_TO_WAV_CONVERSION
        self.run_hf_dataset_creation = DEFAULT_RUN_HF_DATASET_CREATION
        self.ffmpeg_path = DEFAULT_FFMPEG_PATH
        
        dataset_specific_csv_cols = DEFAULT_CSV_COLUMN_MAPS.get(self.dataset_name, DEFAULT_CSV_COLUMN_MAPS.get("meld"))
        self.csv_text_col_name = dataset_specific_csv_cols.get("text_col", "Utterance")
        self.csv_dialogue_id_col_name = dataset_specific_csv_cols.get("dialogue_id_col", "Dialogue_ID")
        self.csv_utterance_id_col_name = dataset_specific_csv_cols.get("utterance_id_col", "Utterance_ID")
        self.csv_emotion_col_name = dataset_specific_csv_cols.get("emotion_col", "Emotion")
        self.csv_audio_path_col_name = dataset_specific_csv_cols.get("audio_path_col", "audio_path")

        # Audio feature parameters
        self.n_fft = DEFAULT_N_FFT
        self.hop_length = DEFAULT_HOP_LENGTH
        self.win_length = DEFAULT_WIN_LENGTH
        self.n_mels = DEFAULT_N_MELS
        self.f_min = DEFAULT_F_MIN
        self.f_max = DEFAULT_F_MAX # Will be cfg.sample_rate / 2 if None and used in MelSpectrogram
        self.power = DEFAULT_POWER
        self.normalize_spectrogram = DEFAULT_NORMALIZE_SPECTROGRAM 
        self.mel_scale = DEFAULT_MEL_SCALE

    def _update_dataset_specifics(self):
        """Update dataset-specific settings after all configuration is loaded."""
        # Update CSV column names based on dataset
        dataset_specific_csv_cols = DEFAULT_CSV_COLUMN_MAPS.get(self.dataset_name, DEFAULT_CSV_COLUMN_MAPS.get("meld"))
        self.csv_text_col_name = dataset_specific_csv_cols.get("text_col", "Utterance")
        self.csv_dialogue_id_col_name = dataset_specific_csv_cols.get("dialogue_id_col", "Dialogue_ID")
        self.csv_utterance_id_col_name = dataset_specific_csv_cols.get("utterance_id_col", "Utterance_ID")
        self.csv_emotion_col_name = dataset_specific_csv_cols.get("emotion_col", "Emotion")
        self.csv_audio_path_col_name = dataset_specific_csv_cols.get("audio_path_col", "audio_path")

    def _setup_paths(self):
        """Set up paths based on the current configuration."""
        # Model and results paths
        self.model_save_dir = self.get_specific_model_dir(self.architecture_name)
        self.results_save_dir = self.get_specific_results_dir(self.architecture_name)
        
        # Create directories if they don't exist
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.results_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model save path
        self.model_save_path = self.model_save_dir / f"{self.architecture_name}_model.pt"

    def _set_device(self):
        """Set up the device based on configuration."""
        if self.device_name == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device_name = "cpu"
        elif self.device_name == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            self.device_name = "cpu"
        
        self.device = torch.device(self.device_name)
        
    @property
    def dataset_data_root(self) -> Path:
        """Root directory for the current dataset's data (e.g., project_root/meld_data)."""
        return self.project_root / f"{self.dataset_name}_data"

    @property
    def raw_data_dir(self) -> Path:
        """Base directory for raw data (CSVs, media) for the current dataset (e.g., meld_data/raw)."""
        return self.dataset_data_root / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Base directory for processed data (audio, features) for the current dataset (e.g., meld_data/processed)."""
        return self.dataset_data_root / "processed"

    # raw_csv_data_dir and raw_media_dir now simply point to raw_data_dir
    @property
    def raw_csv_data_dir(self) -> Path:
        """Directory where raw CSV data files are expected (e.g., meld_data/raw/)."""
        return self.raw_data_dir

    @property
    def raw_media_dir(self) -> Path:
        """Directory where raw media files (e.g., MELD MP4s in train_videos/) are expected (e.g., meld_data/raw/)."""
        return self.raw_data_dir 

    @property
    def processed_audio_output_dir(self) -> Path:
        """Base directory where processed audio (WAVs) will be stored (e.g., meld_data/processed/audio)."""
        return self.processed_data_dir / "audio"

    @property
    def processed_features_dir(self) -> Path:
        """Base directory where processed features will be stored (e.g., meld_data/processed/features)."""
        return self.processed_data_dir / "features"

    @property
    def processed_hf_dataset_dir(self) -> Path:
        """Directory where Hugging Face datasets are stored (e.g., meld_data/processed/features/hf_datasets)."""
        return self.processed_features_dir / "hf_datasets"

    @property
    def label_encoder(self) -> Dict[str, int]:
        if self.dataset_name == "meld":
            return {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
        elif self.dataset_name == "iemocap": 
            return {"ang": 0, "hap": 1, "neu": 2, "sad": 3, "exc": 4, "fru":5} 
        else:
            print(f"Warning: Label encoder not defined for dataset: {self.dataset_name}. Using MELD's as default.")
            return {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder)

    @property
    def class_names(self) -> List[str]:
        return list(self.label_encoder.keys())
    
    @property
    def output_dim(self) -> int:
        return self.num_classes

    def get_specific_model_dir(self, architecture_name: str) -> Path:
        return self.models_dir / architecture_name

    def get_specific_results_dir(self, architecture_name: str) -> Path:
        return self.results_dir / architecture_name

    def __str__(self):
        config_vars = {attr: getattr(self, attr) for attr in dir(self) 
                       if not callable(getattr(self, attr)) and not attr.startswith("__")}
        config_vars_filtered = {k: v for k,v in config_vars.items() if not isinstance(v, dict) or k == "dataset_csv_columns"}
        return "\n".join(f"{key}: {value}" for key, value in config_vars_filtered.items())
    
    @classmethod
    def from_args(cls, args: argparse.Namespace,
                  architecture_name_cli: Optional[str] = None,
                  yaml_config_path: Optional[str] = None):

        # Instantiate with the architecture_name from CLI (--architecture)
        config_instance = cls(architecture_name_override=architecture_name_cli)

        # 1. Load from YAML
        if yaml_config_path:
            yaml_path = Path(yaml_config_path)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    try:
                        yaml_data = yaml.safe_load(f)
                        if yaml_data:
                            for key, value in yaml_data.items():
                                if key == 'architecture_name' and value != architecture_name_cli and architecture_name_cli is not None:
                                    print(f"Warning: YAML 'architecture_name' ('{value}') differs from CLI "
                                          f"(--architecture '{architecture_name_cli}'). Using CLI value for paths and model loading.")
                                    continue
                                setattr(config_instance, key, value)
                        config_instance.yaml_config_loaded_path = str(yaml_path.resolve())
                        print(f"Loaded configuration from: {yaml_path.resolve()}")
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {yaml_path}: {e}")
            else:
                print(f"Warning: YAML config file not found at {yaml_path}. Using defaults and CLI args.")

        # Ensure architecture_name from CLI is firmly set if it wasn't updated by YAML
        if architecture_name_cli and config_instance.architecture_name != architecture_name_cli:
            config_instance.architecture_name = architecture_name_cli

        # 2. Override with specific CLI arguments
        cli_overrides_map = {
            'dataset_name': getattr(args, 'dataset_name', None),
            'input_mode': getattr(args, 'input_mode', None),
            'random_seed': getattr(args, 'seed', None),
            'device_name': getattr(args, 'device_name', None),
            'num_epochs': getattr(args, 'num_epochs', None),
            'batch_size': getattr(args, 'batch_size', None),
            'learning_rate': getattr(args, 'learning_rate', None),
            'experiment_name': getattr(args, 'experiment_name', None),
            'limit_dialogues_train': getattr(args, 'limit_dialogues_train', None),
            'limit_dialogues_dev': getattr(args, 'limit_dialogues_dev', None),
            'limit_dialogues_test': getattr(args, 'limit_dialogues_test', None),
        }

        for attr_name, cli_value in cli_overrides_map.items():
            if cli_value is not None:
                setattr(config_instance, attr_name, cli_value)

        # After all values are finalized:
        config_instance._update_dataset_specifics()
        config_instance._setup_paths()
        config_instance._set_device()

        return config_instance
    
    def validate(self):
        """Validate that the configuration is valid."""
        csv_dir_to_check = self.raw_csv_data_dir 
        if not csv_dir_to_check.exists():
                print(f"Warning: Raw CSV data directory not found at {csv_dir_to_check}.")
        elif self.dataset_name == "meld" and not all((csv_dir_to_check / f"{s}_sent_emo.csv").exists() for s in ["train", "dev", "test"]):
            print(f"Warning: Not all MELD CSV files (train_sent_emo.csv, etc.) found in {csv_dir_to_check}")
        
        media_dir_to_check = self.raw_media_dir
        if not media_dir_to_check.exists():
            print(f"Warning: Raw media base directory not found at {media_dir_to_check}. Expecting subfolders like 'train_videos'.")

        if self.input_mode not in ["audio_text", "audio_only_asr", "video_audio_text"]:
            print(f"Warning: Unknown input_mode '{self.input_mode}'. Supported modes are audio_text, audio_only_asr, video_audio_text.")
        
        return True

print(f"BaseConfig class defined. Default device can be determined via an instance: BaseConfig().device")

# The old global constants like MELD_LABEL_ENCODER, MELD_NUM_CLASSES, CLASS_NAMES,
# MELD_DATA_SUBDIR, MELD_AUDIO_WAV_DIR, MELD_FEATURES_CACHE_DIR, TEXT_ENCODER_MODEL_NAME, etc.
# are now handled by BaseConfig properties or have defaults that BaseConfig initializes with.
# Scripts should instantiate a BaseConfig (or a derived architecture-specific config)
# and access these values through the config object instance. 
### common/config.py content ends ### 