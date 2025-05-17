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

# --- Default Feature Extraction Models & Parameters (can be overridden) ---
DEFAULT_TEXT_ENCODER_MODEL_NAME = "openai/whisper-base"
DEFAULT_TEXT_FEATURE_DIM = 512
DEFAULT_AUDIO_ENCODER_MODEL_NAME = "microsoft/wavlm-base-plus"
DEFAULT_AUDIO_FEATURE_DIM = 768
DEFAULT_ASR_MODEL_NAME = "openai/whisper-base" # For ASR tasks
DEFAULT_ASR_MODEL_NAME_FOR_HF_DATASET: Optional[str] = None # Explicitly None for data prep

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TEXT_MAX_LENGTH_FOR_HF_DATASET = 128 # Matches old MAX_SEQ_LENGTH_TEXT

# --- Default Training Hyperparameters (can be overridden) ---
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 20
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_OPTIMIZER_NAME = "AdamW"
DEFAULT_LR_SCHEDULER_NAME: Optional[str] = None # e.g., "linear", "cosine"
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1
# DEFAULT_MAX_SEQ_LENGTH_TEXT = 128 # Replaced by DEFAULT_TEXT_MAX_LENGTH_FOR_HF_DATASET for clarity
DEFAULT_MAX_AUDIO_DURATION_SECONDS = 15

# --- Default Technical Settings (can be overridden) ---
DEFAULT_DEVICE_NAME = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
DEFAULT_RANDOM_SEED = 42
DEFAULT_NUM_DATALOADER_WORKERS = min(os.cpu_count(), 4) if os.cpu_count() else 1
DEFAULT_MIXED_PRECISION_TRAINING = True

# --- Default Data Preparation Switches ---
DEFAULT_RUN_MP4_TO_WAV_CONVERSION = True
DEFAULT_RUN_HF_DATASET_CREATION = True
DEFAULT_USE_ASR_FOR_TEXT_GENERATION_IN_HF_DATASET = False
DEFAULT_FFMPEG_PATH = "ffmpeg"

# --- Default Evaluation & Inference Settings ---
DEFAULT_EVAL_SPLIT = "dev" # Split to use for evaluation after training or standalone eval
DEFAULT_INFER_NUM_EXAMPLES: Optional[int] = 10 # Limit examples for console print during inference

# --- Default Dataset Limits for quick testing (can be overridden) ---
DEFAULT_LIMIT_DIALOGUES_TRAIN: Optional[int] = None
DEFAULT_LIMIT_DIALOGUES_DEV: Optional[int] = None
DEFAULT_LIMIT_DIALOGUES_TEST: Optional[int] = None

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

    def __init__(self,
                 data_root_override: Optional[Path] = None,
                 output_dir_root_override: Optional[Path] = None,
                 dataset_name: str = "meld",
                 input_mode: str = "audio_text",
                 architecture_name_override: Optional[str] = None,
                 **kwargs): # Allow other kwargs to be passed for flexibility
        """
        Args:
            data_root_override (Optional[Path]): Explicit override for data_root.
            output_dir_root_override (Optional[Path]): Explicit override for output_dir_root.
            dataset_name (str): Name of the dataset (e.g., "meld", "iemocap").
            input_mode (str): Input modality combination.
            architecture_name_override (Optional[str]): Overrides the architecture name if provided.
        """
        self.project_root = PROJECT_ROOT

        self.dataset_name = dataset_name.lower()
        # Core paths - these will be set by from_args based on YAML or CLI
        # Initialize with defaults or overrides if provided directly to __init__
        self.data_root: Path = data_root_override if data_root_override else self.project_root / f"{self.dataset_name}_data"
        self.output_dir_root: Path = output_dir_root_override if output_dir_root_override else self.project_root

        self.input_mode = input_mode.lower()

        if architecture_name_override:
            self.architecture_name = architecture_name_override
        elif not hasattr(self, 'architecture_name'): # If subclass hasn't set it
            self.architecture_name = "base"
        # If hasattr(self, 'architecture_name') is True, it means a subclass (like MLPFusionConfig) already set it.

        self.text_encoder_model_name = DEFAULT_TEXT_ENCODER_MODEL_NAME
        self.text_feature_dim = DEFAULT_TEXT_FEATURE_DIM
        self.audio_encoder_model_name = DEFAULT_AUDIO_ENCODER_MODEL_NAME
        self.audio_feature_dim = DEFAULT_AUDIO_FEATURE_DIM
        self.sample_rate = DEFAULT_SAMPLE_RATE

        self.asr_model_name = DEFAULT_ASR_MODEL_NAME # For inference if ASR is used
        self.asr_model_name_for_hf_dataset = DEFAULT_ASR_MODEL_NAME_FOR_HF_DATASET # For data prep
        self.use_asr_for_text_generation_in_hf_dataset = DEFAULT_USE_ASR_FOR_TEXT_GENERATION_IN_HF_DATASET
        self.text_max_length_for_hf_dataset = DEFAULT_TEXT_MAX_LENGTH_FOR_HF_DATASET
        self.load_asr_model_on_init = (self.input_mode == "audio_only_asr")

        self.batch_size = DEFAULT_BATCH_SIZE
        self.learning_rate = DEFAULT_LEARNING_RATE
        self.num_epochs = DEFAULT_NUM_EPOCHS
        self.early_stopping_patience = DEFAULT_EARLY_STOPPING_PATIENCE
        self.optimizer_name = DEFAULT_OPTIMIZER_NAME
        self.lr_scheduler_name = DEFAULT_LR_SCHEDULER_NAME
        self.gradient_accumulation_steps = DEFAULT_GRADIENT_ACCUMULATION_STEPS
        # self.max_seq_length_text = DEFAULT_MAX_SEQ_LENGTH_TEXT # Replaced
        self.max_audio_duration_seconds = DEFAULT_MAX_AUDIO_DURATION_SECONDS

        self.device_name = DEFAULT_DEVICE_NAME
        # self.device will be set by _set_device() called at the end of from_args
        self.random_seed = DEFAULT_RANDOM_SEED
        self.num_dataloader_workers = DEFAULT_NUM_DATALOADER_WORKERS
        self.mixed_precision_training = DEFAULT_MIXED_PRECISION_TRAINING

        self.run_mp4_to_wav_conversion = DEFAULT_RUN_MP4_TO_WAV_CONVERSION
        self.run_hf_dataset_creation = DEFAULT_RUN_HF_DATASET_CREATION
        self.ffmpeg_path = DEFAULT_FFMPEG_PATH

        self.eval_split = DEFAULT_EVAL_SPLIT
        self.infer_num_examples = DEFAULT_INFER_NUM_EXAMPLES

        self.limit_dialogues_train = DEFAULT_LIMIT_DIALOGUES_TRAIN
        self.limit_dialogues_dev = DEFAULT_LIMIT_DIALOGUES_DEV
        self.limit_dialogues_test = DEFAULT_LIMIT_DIALOGUES_TEST

        # CSV columns will be set by _update_dataset_specifics()
        self.csv_text_col_name: Optional[str] = None
        self.csv_dialogue_id_col_name: Optional[str] = None
        self.csv_utterance_id_col_name: Optional[str] = None
        self.csv_emotion_col_name: Optional[str] = None
        self.csv_audio_path_col_name: Optional[str] = None

        # Audio feature parameters
        self.n_fft = DEFAULT_N_FFT
        self.hop_length = DEFAULT_HOP_LENGTH
        self.win_length = DEFAULT_WIN_LENGTH
        self.n_mels = DEFAULT_N_MELS
        self.f_min = DEFAULT_F_MIN
        self.f_max = DEFAULT_F_MAX
        self.power = DEFAULT_POWER
        self.normalize_spectrogram = DEFAULT_NORMALIZE_SPECTROGRAM
        self.mel_scale = DEFAULT_MEL_SCALE

        # Placeholder for experiment name, to be set by from_args or main.py
        self.experiment_name: Optional[str] = None
        self.yaml_config_loaded_path: Optional[str] = None # Track which YAML was loaded

        # Apply any other kwargs passed during instantiation
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Info: Kwarg '{key}' not a pre-defined attribute of BaseConfig, adding it.")
                setattr(self, key, value)

        # Initial setup based on defaults (will be re-run after YAML/CLI in from_args)
        # self._update_dataset_specifics()
        # self._setup_paths()
        # self._set_device()


    def _update_dataset_specifics(self):
        """Update dataset-specific settings based on self.dataset_name."""
        dataset_specific_csv_cols = DEFAULT_CSV_COLUMN_MAPS.get(self.dataset_name, DEFAULT_CSV_COLUMN_MAPS.get("meld"))
        self.csv_text_col_name = dataset_specific_csv_cols.get("text_col", "Utterance")
        self.csv_dialogue_id_col_name = dataset_specific_csv_cols.get("dialogue_id_col", "Dialogue_ID")
        self.csv_utterance_id_col_name = dataset_specific_csv_cols.get("utterance_id_col", "Utterance_ID")
        self.csv_emotion_col_name = dataset_specific_csv_cols.get("emotion_col", "Emotion")
        self.csv_audio_path_col_name = dataset_specific_csv_cols.get("audio_path_col", "audio_path")

    def _setup_paths(self):
        """Set up all derived paths based on the finalized self.data_root,
        self.output_dir_root, and self.architecture_name.
        This method MUST be called after these attributes are finalized.
        """
        self.data_root = Path(self.data_root).resolve()
        self.output_dir_root = Path(self.output_dir_root).resolve()

        _arch_name_for_path = self.architecture_name if self.architecture_name else "unknown_arch"

        self.model_save_dir = self.output_dir_root / "models" / _arch_name_for_path
        self.results_dir = self.output_dir_root / "results" / _arch_name_for_path
        self.log_dir = self.output_dir_root / "logs" # General log dir for experiments

        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


    def _set_device(self):
        """Set up the torch device based on self.device_name."""
        if self.device_name.lower() == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device_name = "cpu"
        elif self.device_name.lower() == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
                print("Warning: MPS requested but not available/built. Falling back to CPU.")
                self.device_name = "cpu"
            # else: # MPS is available and built, keep device_name as mps
            #     pass 
        # Implicitly, if neither CUDA nor MPS is specifically requested and available, it might remain/default to CPU
        # or whatever DEFAULT_DEVICE_NAME resolved to initially.
        # The DEFAULT_DEVICE_NAME logic is now smarter too.

        self.device = torch.device(self.device_name)

    @property
    def dataset_data_root(self) -> Path:
        """Root directory for the current dataset's data. Uses the finalized self.data_root."""
        return Path(self.data_root)

    @property
    def raw_data_dir(self) -> Path:
        """Base directory for raw data (CSVs, media) for the current dataset."""
        return self.dataset_data_root / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Base directory for processed data (audio, features) for the current dataset."""
        return self.dataset_data_root / "processed"

    @property
    def raw_csv_data_dir(self) -> Path:
        """Directory where raw CSV data files are expected."""
        return self.raw_data_dir

    @property
    def raw_media_dir(self) -> Path:
        """Directory where raw media files are expected."""
        return self.raw_data_dir

    @property
    def processed_audio_output_dir(self) -> Path:
        """Base directory where processed audio (WAVs) will be stored."""
        return self.processed_data_dir / "audio"

    @property
    def processed_features_dir(self) -> Path:
        """Base directory where processed features will be stored."""
        return self.processed_data_dir / "features"

    @property
    def processed_hf_dataset_dir(self) -> Path:
        """Directory where Hugging Face datasets are stored."""
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

    def __str__(self):
        config_vars = {attr: getattr(self, attr) for attr in dir(self)
                       if not callable(getattr(self, attr)) and not attr.startswith("__")}
        config_vars_filtered = {
            k: (str(v) if isinstance(v, Path) else v)
            for k,v in config_vars.items()
            if not (isinstance(v, dict) and k != "label_encoder")
        }
        return "\n".join(f"{key}: {value}" for key, value in sorted(config_vars_filtered.items()))

    @classmethod
    def from_args(cls, args: argparse.Namespace,
                  architecture_name_cli: Optional[str] = None,
                  yaml_config_path: Optional[str] = None):

        # Initial values for instantiation from CLI or defaults
        # These are passed to __init__
        
        # Correctly fetch from args, using default if CLI arg is None (not provided)
        cli_dataset_name = args.dataset_name if hasattr(args, 'dataset_name') and args.dataset_name is not None else "meld"
        cli_input_mode = args.input_mode if hasattr(args, 'input_mode') and args.input_mode is not None else "audio_text"

        init_kwargs = {
            'dataset_name': cli_dataset_name,
            'input_mode': cli_input_mode,
            'architecture_name_override': architecture_name_cli
        }
        # If CLI provides data_root or output_dir_root, use them for initial instantiation
        # These act as the very first override if present.
        if hasattr(args, 'data_root') and args.data_root is not None:
            init_kwargs['data_root_override'] = Path(args.data_root)
        if hasattr(args, 'output_dir_root') and args.output_dir_root is not None:
            init_kwargs['output_dir_root_override'] = Path(args.output_dir_root)

        config_instance = cls(**init_kwargs)

        # 1. Load from YAML, potentially overriding attributes set by __init__ (including data_root, output_dir_root)
        if yaml_config_path:
            yaml_path = Path(yaml_config_path)
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    try:
                        yaml_data = yaml.safe_load(f)
                        if yaml_data:
                            # Handle nested 'paths' dictionary from YAML first
                            if 'paths' in yaml_data and isinstance(yaml_data['paths'], dict):
                                paths_from_yaml = yaml_data['paths']
                                if 'data_root' in paths_from_yaml:
                                    config_instance.data_root = Path(paths_from_yaml['data_root'])
                                if 'output_dir_root' in paths_from_yaml:
                                    config_instance.output_dir_root = Path(paths_from_yaml['output_dir_root'])
                                del yaml_data['paths'] # Remove so it's not processed again

                            # Update other top-level attributes from YAML
                            for key, value in yaml_data.items():
                                if key == 'architecture_name' and value != architecture_name_cli and architecture_name_cli is not None:
                                    # YAML arch name differs from CLI; CLI arch name takes precedence.
                                    # config_instance.architecture_name is already set by architecture_name_cli.
                                    print(f"Info: YAML 'architecture_name' ('{value}') differs from CLI "
                                          f"(--architecture '{architecture_name_cli}'). Using CLI value.")
                                    continue # Skip setting architecture_name from YAML if CLI provided it.
                                if hasattr(config_instance, key):
                                    setattr(config_instance, key, value)
                                else: # Add new keys from YAML if they don't exist
                                    print(f"Info: YAML key '{key}' not a pre-defined attribute of config, adding it.")
                                    setattr(config_instance, key, value)
                        config_instance.yaml_config_loaded_path = str(yaml_path.resolve())
                        # print(f"Loaded configuration from: {yaml_path.resolve()}") # This is printed in main.py already
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {yaml_path}: {e}")
            # else: # This warning is also handled in main.py
                # print(f"Warning: YAML config file not found at {yaml_path}. Using defaults and CLI args.")

        # Re-affirm architecture_name from CLI if it was provided, as it's crucial for logic and paths.
        if architecture_name_cli and config_instance.architecture_name != architecture_name_cli:
            config_instance.architecture_name = architecture_name_cli

        # 2. Override with specific CLI arguments (these have the highest precedence)
        # Check for attributes that might be on 'args' and update 'config_instance'
        # This loop handles all args that have a direct mapping to config attributes.
        for arg_name, arg_value in vars(args).items():
            if arg_value is not None: # Only override if CLI arg was actually provided
                if hasattr(config_instance, arg_name):
                    if arg_name in ['data_root', 'output_dir_root']:
                        setattr(config_instance, arg_name, Path(arg_value))
                    elif arg_name == 'seed' and hasattr(config_instance, 'random_seed'): # Map CLI 'seed' to 'random_seed'
                        setattr(config_instance, 'random_seed', arg_value)
                    else:
                        setattr(config_instance, arg_name, arg_value)
                # else: # If CLI arg doesn't match a config attribute, it might be handled by specific logic in main.py
                    # print(f"Info: CLI arg '{arg_name}' not a direct attribute of config, may be used by main.py logic.")

        # Ensure experiment_name is set (main.py also has a backfill for this)
        if not config_instance.experiment_name and config_instance.architecture_name:
             config_instance.experiment_name = f"{config_instance.architecture_name}_default_experiment"


        # CRITICAL FINALIZATION STEPS:
        # These must be called after all values (from init, YAML, CLI) are finalized.
        config_instance._update_dataset_specifics() # Uses final self.dataset_name
        config_instance._setup_paths()             # Uses final self.data_root, self.output_dir_root, self.architecture_name
        config_instance._set_device()              # Uses final self.device_name

        return config_instance

    def validate(self):
        """Validate that the configuration is valid."""
        if not self.raw_csv_data_dir.exists():
                print(f"Warning: Raw CSV data directory not found at {self.raw_csv_data_dir}.")
        elif self.dataset_name == "meld" and not all((self.raw_csv_data_dir / f"{s}_sent_emo.csv").exists() for s in ["train", "dev", "test"]):
            print(f"Warning: Not all MELD CSV files (train_sent_emo.csv, etc.) found in {self.raw_csv_data_dir}")

        if not self.raw_media_dir.exists():
            print(f"Warning: Raw media base directory not found at {self.raw_media_dir}. Expecting subfolders like 'train_videos'.")

        if self.input_mode not in ["audio_text", "audio_only_asr", "video_audio_text"]: # Add other supported modes
            print(f"Warning: Unknown input_mode '{self.input_mode}'.")

        # Add more validation as needed
        return True

print(f"BaseConfig class defined. Default device can be determined via an instance: BaseConfig().device_name") # Changed to device_name for clarity
### common/config.py content ends ### 