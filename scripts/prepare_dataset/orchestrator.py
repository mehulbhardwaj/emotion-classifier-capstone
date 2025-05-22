from pathlib import Path
from functools import partial
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
import torch # For Whisper model device placement

# Assuming BaseConfig and ensure_dir are in accessible locations
# from configs.base_config import BaseConfig # Placeholder
from utils.utils import ensure_dir # Changed

# Import from other modules in this package
from .csv_loader import create_dataset_from_csv
from .audio_feature_extractor import get_hf_audio_feature_extractor, build_output_hf_features_schema
from .processor import DatasetItemProcessor

# import logging
# logger = logging.getLogger(__name__)

class MELDDatasetOrchestrator:
    """
    Orchestrates the creation, processing, and saving of MELD Hugging Face datasets.
    Initializes necessary models and tokenizers once.
    """
    def __init__(self, config):
        """
        Args:
            config (BaseConfig): The main configuration object.
        """
        self.config = config
        self.text_tokenizer_hf = None
        self.hf_audio_feature_extractor = None
        self.whisper_processor_hf = None
        self.whisper_model_hf = None
        self._initialize_components()

    def _initialize_components(self):
        """Loads and initializes tokenizers, feature extractors, and ASR models if configured."""
        # logger.info("Initializing components for MELDDatasetOrchestrator...")
        print("Initializing components for MELDDatasetOrchestrator...")

        # Text Tokenizer
        if self.config.text_encoder_model_name:
            try:
                self.text_tokenizer_hf = AutoTokenizer.from_pretrained(self.config.text_encoder_model_name)
                # logger.info(f"Text tokenizer loaded: {self.config.text_encoder_model_name}")
                print(f"Text tokenizer loaded: {self.config.text_encoder_model_name}")
            except Exception as e_tok:
                # logger.error(f"Could not load text tokenizer {self.config.text_encoder_model_name}: {e_tok}. Text processing might fail.", exc_info=True)
                print(f"ERROR: Could not load text tokenizer {self.config.text_encoder_model_name}: {e_tok}. Text processing might fail.")
        else:
            # logger.warning("config.text_encoder_model_name is not set. Text tokenization will be skipped.")
            print("Warning: config.text_encoder_model_name is not set. Text tokenization will be skipped.")

        # Hugging Face Audio Feature Extractor (if audio_input_type is hf_features)
        if self.config.audio_input_type == "hf_features":
            if self.config.audio_encoder_model_name:
                self.hf_audio_feature_extractor = get_hf_audio_feature_extractor(self.config)
                if not self.hf_audio_feature_extractor:
                    # logger.critical(f"Failed to load HF Audio Feature Extractor for {self.config.audio_encoder_model_name}. 'hf_features' mode will not work.")
                    print(f"CRITICAL: Failed to load HF Audio Feature Extractor for {self.config.audio_encoder_model_name}. 'hf_features' mode will not work.")
            else:
                # logger.critical("audio_input_type is 'hf_features' but config.audio_encoder_model_name is not set.")
                print("CRITICAL: audio_input_type is 'hf_features' but config.audio_encoder_model_name is not set.")

        # ASR (Whisper) Models
        self.actual_use_asr = getattr(self.config, 'use_asr_for_text_generation_in_hf_dataset', False)
        asr_model_name_to_load = getattr(self.config, 'asr_model_name_for_hf_dataset', getattr(self.config, 'asr_model_name', None))

        if self.actual_use_asr and asr_model_name_to_load:
            # logger.info(f"Loading ASR model ({asr_model_name_to_load}) for processing...")
            print(f"Loading ASR model ({asr_model_name_to_load}) for processing...")
            try:
                self.whisper_processor_hf = WhisperProcessor.from_pretrained(asr_model_name_to_load)
                self.whisper_model_hf = WhisperForConditionalGeneration.from_pretrained(asr_model_name_to_load).to(self.config.device)
                self.whisper_model_hf.eval()
                # logger.info(f"ASR model {asr_model_name_to_load} loaded to device: {self.config.device}")
                print(f"ASR model {asr_model_name_to_load} loaded to device: {self.config.device}")
            except Exception as e_asr_load:
                # logger.error(f"Error loading ASR model {asr_model_name_to_load}: {e_asr_load}. ASR will be disabled.", exc_info=True)
                print(f"ERROR loading ASR model {asr_model_name_to_load}: {e_asr_load}. ASR will be disabled.")
                self.actual_use_asr = False # Disable if loading failed
        elif self.actual_use_asr and not asr_model_name_to_load:
            # logger.warning("ASR is requested but no asr_model_name specified in config. ASR will be disabled.")
            print("Warning: ASR is requested but no asr_model_name specified in config. ASR will be disabled.")
            self.actual_use_asr = False
        # logger.info(f"ASR for text generation in HF dataset: {self.actual_use_asr}")
        print(f"ASR for text generation in HF dataset: {self.actual_use_asr}")

    def _build_and_process_single_split(self, split_name: str, limit_dialogues: int = None, num_proc: int = None):
        """
        Builds and processes a single dataset split (e.g., 'train', 'dev', 'test').
        """
        # logger.info(f"\nBuilding and processing dataset for {split_name} split...")
        print(f"\nBuilding and processing dataset for {split_name} split...")

        # 1. Create initial dataset from CSV
        # The create_dataset_from_csv function is imported from .csv_loader
        initial_dataset = create_dataset_from_csv(split_name, config=self.config, limit_dialogues=limit_dialogues)
        
        if initial_dataset and len(initial_dataset) > 0: # Add this check
            print(f"DEBUG Orchestrator: initial_dataset columns: {initial_dataset.column_names}")
            print(f"DEBUG Orchestrator: initial_dataset features: {initial_dataset.features}")
            print(f"DEBUG Orchestrator: First item from initial_dataset for {split_name}: {initial_dataset[0]}")
            if len(initial_dataset) > 1:
                 print(f"DEBUG Orchestrator: Second item from initial_dataset for {split_name}: {initial_dataset[1]}")
            print(f"DEBUG Orchestrator: Keys in first item: {initial_dataset[0].keys()}")
        elif initial_dataset is not None: # It could be an empty Dataset object
            print(f"DEBUG Orchestrator: initial_dataset for {split_name} is empty (length {len(initial_dataset)}). Columns: {initial_dataset.column_names if hasattr(initial_dataset, 'column_names') else 'N/A'}")
        else:
            print(f"DEBUG Orchestrator: initial_dataset for {split_name} is None.")

        if not initial_dataset or len(initial_dataset) == 0:
            # logger.warning(f"Initial dataset for {split_name} is empty or None. Skipping further processing.")
            print(f"Initial dataset for {split_name} is empty or None. Skipping further processing.")
            return None

        # RENAME original audio_path to avoid conflicts within .map()
        if 'audio_path' in initial_dataset.column_names:
            print(f"DEBUG Orchestrator: Renaming 'audio_path' to '_original_audio_path_abs' in initial_dataset for {split_name}")
            initial_dataset = initial_dataset.rename_column("audio_path", "_original_audio_path_abs")
            print(f"DEBUG Orchestrator: initial_dataset columns after rename: {initial_dataset.column_names}")

        # 2. Define the output schema for the processed Hugging Face dataset
        # This function is imported from .audio_feature_extractor
        output_schema = build_output_hf_features_schema(self.config)
        # logger.info(f"Output schema for {split_name}: {output_schema}")
        print(f"Output schema for {split_name}: {output_schema}")

        # 3. Instantiate the item processor with pre-loaded components
        item_processor_instance = DatasetItemProcessor(
            config=self.config,
            audio_feature_extractor_hf=self.hf_audio_feature_extractor,
            text_tokenizer_hf=self.text_tokenizer_hf,
            whisper_processor_hf=self.whisper_processor_hf,
            whisper_model_hf=self.whisper_model_hf
        )
        # The DatasetItemProcessor's __init__ will also set its internal self.use_asr based on config and presence of models.

        # Columns to remove after .map(). 
        # 'audio_path' (full path) is always removed as it's either replaced by a relative path 
        # or by extracted audio features. Other original columns like dialogue_id, utterance_id, etc.,
        # are kept if they are part of the output_schema and not explicitly overwritten by processor.
        columns_to_remove_after_map = ['audio_path'] 
        # If the processor returns a dict that completely replaces the input item, then all original
        # columns of `initial_dataset` would be removed unless they are also keys in the processor's output
        # AND in the `output_schema`. The DatasetItemProcessor now merges its results with the input item.

        # logger.info(f"Applying processing to {split_name} dataset ({len(initial_dataset)} items). Num_proc: {num_proc}")
        print(f"Applying processing to {split_name} dataset ({len(initial_dataset)} items). Num_proc: {num_proc}")
        try:
            processed_dataset = initial_dataset.map(
                item_processor_instance, # The __call__ method of the instance will be used
                num_proc=None, # Force single process for debugging the KeyError on 'audio_path'
                remove_columns=None, 
                features=output_schema, # Apply the explicit schema
                batched=False # DatasetItemProcessor is designed for single items
            )
        except Exception as e_map:
            # logger.error(f"Error during .map() for {split_name} split: {e_map}. Processing failed for this split.", exc_info=True)
            print(f"ERROR during .map() for {split_name} split: {e_map}. Processing failed for this split.")
            return None
        
        # TEMPORARY: Remove the _original_audio_path_abs column after mapping is done
        if processed_dataset and '_original_audio_path_abs' in processed_dataset.column_names:
            print(f"DEBUG Orchestrator: Removing temporary '_original_audio_path_abs' column from processed {split_name} dataset.")
            processed_dataset = processed_dataset.remove_columns(['_original_audio_path_abs'])
            print(f"DEBUG Orchestrator: Columns after removal: {processed_dataset.column_names}")

        num_filtered = len(initial_dataset) - len(processed_dataset)
        if num_filtered > 0:
            # logger.info(f"Filtered out {num_filtered} items from {split_name} due to processing errors (processor returned None). indigestion" )
            print(f"Filtered out {num_filtered} items from {split_name} due to processing errors (processor returned None).")

        # logger.info(f"Finished processing {split_name} dataset. Final size: {len(processed_dataset)} items.")
        print(f"Finished processing {split_name} dataset. Final size: {len(processed_dataset)} items.")
        if len(processed_dataset) > 0:
            # logger.debug(f"Columns in processed {split_name} dataset: {processed_dataset.column_names}")
            print(f"Columns in processed {split_name} dataset: {processed_dataset.column_names}")
        return processed_dataset

    def prepare_all_splits(self, 
                           splits_to_process=None, 
                           force_reprocess_items=False, 
                           num_workers_dataset_map=None,
                           limit_dialogues_train=None, 
                           limit_dialogues_dev=None,   
                           limit_dialogues_test=None):
        """
        Prepares all specified MELD Hugging Face Dataset splits.
        Orchestrates loading, processing, and saving with caching.
        """
        # logger.info(f"Starting MELD Hugging Face dataset preparation using config: {self.config.architecture_name if hasattr(self.config, 'architecture_name') else 'N/A'}")
        print(f"Starting MELD Hugging Face dataset preparation using config: {getattr(self.config, 'architecture_name', 'N/A')}")
        # logger.info(f"Audio input type set to: {self.config.audio_input_type}")
        print(f"Audio input type set to: {self.config.audio_input_type}")
        if self.config.audio_input_type == "hf_features":
            # logger.info(f"Audio encoder for 'hf_features' mode: {self.config.audio_encoder_model_name}")
            print(f"Audio encoder for 'hf_features' mode: {self.config.audio_encoder_model_name}")
        elif self.config.audio_input_type == "raw_wav":
            path_col = getattr(self.config, 'audio_path_column_name_in_hf_dataset', 'relative_audio_path')
            # logger.info(f"Mode 'raw_wav': Storing relative audio paths in column: '{path_col}'")
            print(f"Mode 'raw_wav': Storing relative audio paths in column: '{path_col}'")

        num_workers_map = num_workers_dataset_map if num_workers_dataset_map is not None else getattr(self.config, 'num_dataloader_workers', 0)
        # if num_workers_map == 0: num_workers_map = None # For .map(), None means main process, but 0 or 1 might be safer for some systems

        all_datasets_processed = {}
        splits_to_run = splits_to_process if splits_to_process else getattr(self.config, 'default_splits_to_process', ['train', 'dev', 'test'])
        hf_save_dir_base = Path(self.config.processed_hf_dataset_dir) # Ensure this is a Path
        ensure_dir(hf_save_dir_base) # Changed

        # Consolidate dialogue limits from args or config
        dialogue_limits_per_split = {
            'train': limit_dialogues_train if limit_dialogues_train is not None else getattr(self.config, 'limit_dialogues_train', None),
            'dev': limit_dialogues_dev if limit_dialogues_dev is not None else getattr(self.config, 'limit_dialogues_dev', None),
            'test': limit_dialogues_test if limit_dialogues_test is not None else getattr(self.config, 'limit_dialogues_test', None)
        }

        for split_name in splits_to_run:
            split_save_path = hf_save_dir_base / split_name
            if not force_reprocess_items and split_save_path.exists() and (split_save_path / "dataset_info.json").exists():
                # logger.info(f"Processed HF dataset for {split_name} found at {split_save_path}. Loading from disk.")
                print(f"Processed HF dataset for {split_name} found at {split_save_path}. Loading from disk.")
                try:
                    # TODO: Add sophisticated cache validation here later (e.g., based on config hash)
                    loaded_ds = Dataset.load_from_disk(str(split_save_path))
                    # logger.info(f"Successfully loaded {split_name} split from disk. Length: {len(loaded_ds)}")
                    print(f"Successfully loaded {split_name} split from disk. Length: {len(loaded_ds)}")
                    all_datasets_processed[split_name] = loaded_ds
                    continue
                except Exception as e_load_disk:
                    # logger.warning(f"Error loading {split_name} from disk ({split_save_path}): {e_load_disk}. Reprocessing.", exc_info=True)
                    print(f"Warning: Error loading {split_name} from disk ({split_save_path}): {e_load_disk}. Reprocessing.")

            split_dialogue_limit = dialogue_limits_per_split.get(split_name)
            # logger.info(f"Limit dialogues for {split_name}: {split_dialogue_limit if split_dialogue_limit is not None else 'None'}")
            print(f"Limit dialogues for {split_name}: {split_dialogue_limit if split_dialogue_limit is not None else 'None'}")
            
            processed_ds_split = self._build_and_process_single_split(
                split_name, 
                limit_dialogues=split_dialogue_limit,
                num_proc=num_workers_map
            )
            
            if processed_ds_split and len(processed_ds_split) > 0:
                all_datasets_processed[split_name] = processed_ds_split
                # logger.info(f"Saving processed {split_name} dataset ({len(processed_ds_split)} items) to: {split_save_path}")
                print(f"Saving processed {split_name} dataset ({len(processed_ds_split)} items) to: {split_save_path}")
                try:
                    # self._ensure_dir(split_save_path.parent) # Replaced
                    ensure_dir(split_save_path.parent) # Changed
                    processed_ds_split.save_to_disk(str(split_save_path))
                    # logger.info(f"Saved {split_name} dataset successfully.")
                    print(f"Saved {split_name} dataset successfully.")
                except Exception as e_save_disk:
                    # logger.error(f"Could not save processed {split_name} dataset to {split_save_path}: {e_save_disk}", exc_info=True)
                    print(f"ERROR: Could not save processed {split_name} dataset to {split_save_path}: {e_save_disk}")
            else:
                # logger.warning(f"Processing for {split_name} resulted in an empty or None dataset. Not saving.")
                print(f"Warning: Processing for {split_name} resulted in an empty or None dataset. Not saving.")
                all_datasets_processed[split_name] = None # Explicitly store None if processing failed

        if not all_datasets_processed or all(ds is None for ds in all_datasets_processed.values()):
            # logger.warning("No datasets were successfully processed or loaded. Returning empty dictionary.")
            print("Warning: No datasets were successfully processed or loaded. Returning empty dictionary.")
            return {}
            
        # logger.info("MELD Hugging Face dataset preparation finished.")
        print("MELD Hugging Face dataset preparation finished.")
        return all_datasets_processed

# Example Usage (Conceptual - requires BaseConfig and other modules/files)
if __name__ == '__main__':
    print("Testing orchestrator.py...")

    # This test requires a more complete setup: 
    # - A valid BaseConfig instance (or a comprehensive DummyConfig)
    # - Dummy CSV data files (train_sent_emo.csv, dev_sent_emo.csv)
    # - Dummy audio WAV files in the expected directory structure
    # - The other modules (csv_loader, audio_feature_extractor, processor) in the same package directory.

    class DummyFullConfig: # More comprehensive dummy config for orchestrator
        def __init__(self, dataset_name="meld_orchestrator_test", audio_type="raw_wav"):
            self.dataset_name = dataset_name
            self.data_root = Path(f"temp_test_data_orch/{dataset_name}")
            # self.logger = logging.getLogger(f"test_{dataset_name}")
            # logging.basicConfig(level=logging.INFO)

            # Paths (Derived from BaseConfig logic)
            self.raw_data_dir = self.data_root / "raw"
            self.processed_audio_output_dir = self.data_root / "processed_audio_16kHz_mono"
            self.processed_hf_dataset_dir = self.data_root / "processed_hf_datasets" / audio_type
            # self.ensure_dir(self.raw_data_dir) # Replaced
            # self.ensure_dir(self.processed_audio_output_dir / "train") # Replaced
            # self.ensure_dir(self.processed_audio_output_dir / "dev") # Replaced
            # self.ensure_dir(self.processed_hf_dataset_dir) # Replaced
            ensure_dir(self.raw_data_dir) # Changed
            ensure_dir(self.processed_audio_output_dir / "train") # Changed
            ensure_dir(self.processed_audio_output_dir / "dev") # Changed
            ensure_dir(self.processed_hf_dataset_dir) # Changed
            
            # Core attributes
            self.sample_rate = 16000
            self.n_mels = 80 # For mel spec, though not directly used if type is raw/hf
            self.hop_length = 160
            self.n_fft = 400
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.num_dataloader_workers = 0 # For .map(), 0 or None for main process

            # Dataset specific (MELD example)
            self.label_encoder = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
            self.id_to_label_map = {v: k for k, v in self.label_encoder.items()}
            self.num_classes = len(self.label_encoder)

            # HF Model Names
            self.text_encoder_model_name = "distilroberta-base"
            self.audio_input_type = audio_type # "raw_wav" or "hf_features"
            self.audio_path_column_name_in_hf_dataset = "relative_audio_path" # For raw_wav
            self.audio_encoder_model_name = "facebook/wav2vec2-base" # For hf_features

            # ASR - set use_asr_for_text_generation_in_hf_dataset to True to test
            self.use_asr_for_text_generation_in_hf_dataset = False 
            self.asr_model_name_for_hf_dataset = "openai/whisper-tiny"
            # self.asr_model_name = "openai/whisper-tiny" # Fallback if above not set
            
            # Text processing for HF dataset
            self.text_max_length_for_hf_dataset = 64 

            # Other potential config items from original BaseConfig might be needed depending on usage
            self.default_splits_to_process = ['train', 'dev'] # For testing convenience
            self.limit_dialogues_train = 1 # Limit data for faster testing
            self.limit_dialogues_dev = 1

            self._create_dummy_data()

        def _create_dummy_data(self):
            # Dummy CSVs
            import pandas as pd
            train_csv_path = self.raw_data_dir / "train_sent_emo.csv"
            dev_csv_path = self.raw_data_dir / "dev_sent_emo.csv"
            if not train_csv_path.exists():
                pd.DataFrame({
                    'Dialogue_ID': [0, 0], 'Utterance_ID': [0, 1], 'Speaker': ['S1', 'S2'], 
                    'Utterance': ['Train utterance one.', 'Train utterance two.'], 
                    'Emotion': ['neutral', 'joy'], 'Sentiment': ['neutral', 'positive']
                }).to_csv(train_csv_path, index=False)
            if not dev_csv_path.exists():
                 pd.DataFrame({
                    'Dialogue_ID': [0], 'Utterance_ID': [0], 'Speaker': ['S3'], 
                    'Utterance': ['Dev utterance one.'], 
                    'Emotion': ['sadness'], 'Sentiment': ['negative']
                }).to_csv(dev_csv_path, index=False)

            # Dummy WAV files (16kHz mono)
            if not (self.processed_audio_output_dir / "train" / "dia0_utt0.wav").exists():
                torchaudio.save(str(self.processed_audio_output_dir / "train" / "dia0_utt0.wav"), torch.randn(1, 16000), 16000)
            if not (self.processed_audio_output_dir / "train" / "dia0_utt1.wav").exists():
                torchaudio.save(str(self.processed_audio_output_dir / "train" / "dia0_utt1.wav"), torch.randn(1, 16000), 16000)
            if not (self.processed_audio_output_dir / "dev" / "dia0_utt0.wav").exists():
                torchaudio.save(str(self.processed_audio_output_dir / "dev" / "dia0_utt0.wav"), torch.randn(1, 16000), 16000)

    # --- Test with audio_input_type = "raw_wav" ---
    print("\n--- Testing Orchestrator with audio_input_type: raw_wav ---")
    cfg_raw_orch = DummyFullConfig(audio_type="raw_wav")
    orchestrator_raw = MELDDatasetOrchestrator(cfg_raw_orch)
    datasets_raw = orchestrator_raw.prepare_all_splits(force_reprocess_items=True)
    
    if datasets_raw.get('train') and len(datasets_raw['train']) > 0:
        print("RAW_WAV Orchestrator - Train Dataset Sample (first item):")
        print(datasets_raw['train'][0])
        print(f"Train dataset columns: {datasets_raw['train'].column_names}")
        assert getattr(cfg_raw_orch, 'audio_path_column_name_in_hf_dataset', 'relative_audio_path') in datasets_raw['train'].column_names
    else:
        print("RAW_WAV Orchestrator - Train dataset is empty or None after processing.")

    # --- Test with audio_input_type = "hf_features" ---
    print("\n--- Testing Orchestrator with audio_input_type: hf_features ---")
    cfg_hf_orch = DummyFullConfig(audio_type="hf_features")
    # cfg_hf_orch.use_asr_for_text_generation_in_hf_dataset = True # Enable to test ASR path
    orchestrator_hf = MELDDatasetOrchestrator(cfg_hf_orch)
    datasets_hf = orchestrator_hf.prepare_all_splits(force_reprocess_items=True)

    if datasets_hf.get('train') and len(datasets_hf['train']) > 0:
        print("HF_FEATURES Orchestrator - Train Dataset Sample (first item):")
        print(datasets_hf['train'][0])
        print(f"Train dataset columns: {datasets_hf['train'].column_names}")
        assert 'audio_input_values' in datasets_hf['train'].column_names
    else:
        print("HF_FEATURES Orchestrator - Train dataset is empty or None after processing.")

    # Clean up dummy data directories (optional)
    # import shutil
    # shutil.rmtree("temp_test_data_orch", ignore_errors=True)

    print("\norchestrator.py test run finished.") 