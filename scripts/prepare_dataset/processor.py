import torch
import torchaudio
from pathlib import Path
from transformers import AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration

# Assuming BaseConfig is accessible, e.g., from configs.base_config
# from configs.base_config import BaseConfig # Placeholder

# Assuming audio feature extraction functions are in audio_feature_extractor
# from .audio_feature_extractor import get_hf_audio_feature_extractor, extract_hf_audio_features, extract_mel_spectrogram_custom # Placeholder for relative import

# It's good practice to use a logger instead of print for production code.
# import logging
# logger = logging.getLogger(__name__)

class DatasetItemProcessor:
    """
    Processes a single item from the initial dataset (created by csv_loader).
    Handles audio feature extraction (via audio_feature_extractor module),
    optional ASR, and text tokenization.
    Returns a dictionary of features for the final Hugging Face dataset.
    """
    def __init__(self, config, audio_feature_extractor_hf=None, text_tokenizer_hf=None, 
                 whisper_processor_hf=None, whisper_model_hf=None):
        """
        Initializes the processor with necessary components based on the config.
        Components like tokenizers and models can be pre-loaded and passed in 
        to avoid re-loading for each item or each processor instantiation if used in .map().
        """
        self.config = config
        self.current_device = config.device

        # Audio processing components
        self.audio_feature_extractor_hf = audio_feature_extractor_hf
        # If using a custom mel spectrogram directly as a feature:
        # self.use_custom_mel = (config.audio_input_type == "custom_mel_spectrogram") # Example config flag

        # Text processing components
        self.text_tokenizer = text_tokenizer_hf
        
        # ASR components (Whisper)
        self.use_asr = getattr(config, 'use_asr_for_text_generation_in_hf_dataset', False)
        self.whisper_processor = whisper_processor_hf
        self.whisper_model = whisper_model_hf
        if self.use_asr and (not self.whisper_processor or not self.whisper_model):
            # logger.warning("ASR is enabled in config, but Whisper processor/model not provided to DatasetItemProcessor. ASR will be skipped.")
            print("Warning: ASR is enabled in config, but Whisper processor/model not provided to DatasetItemProcessor. ASR will be skipped.")
            self.use_asr = False # Disable ASR if components are missing

    def _get_relative_audio_path(self, full_audio_path_str):
        try:
            # Ensure processed_audio_output_dir is a Path object if it comes from config
            base_path = Path(self.config.processed_audio_output_dir)
            relative_path = Path(full_audio_path_str).relative_to(base_path)
            return str(relative_path)
        except ValueError as e_rel_path:
            # logger.warning(f"Error creating relative path for {full_audio_path_str} relative to {self.config.processed_audio_output_dir}: {e_rel_path}. Storing absolute path as fallback.")
            print(f"Warning: Error creating relative path for {full_audio_path_str} relative to {self.config.processed_audio_output_dir}: {e_rel_path}. Storing absolute path as fallback.")
            return full_audio_path_str
        except Exception as e_other_path:
            # logger.error(f"Unexpected error getting relative path for {full_audio_path_str}: {e_other_path}")
            print(f"Error: Unexpected error getting relative path for {full_audio_path_str}: {e_other_path}")
            return full_audio_path_str # Fallback to absolute path

    def _perform_asr(self, audio_path_str):
        """Helper to perform ASR using Whisper."""
        try:
            # Waveform loading for ASR, resample if needed for Whisper processor
            waveform_asr, sr_asr = torchaudio.load(audio_path_str)
            expected_asr_sr = self.whisper_processor.feature_extractor.sampling_rate
            if sr_asr != expected_asr_sr:
                resampler_asr = torchaudio.transforms.Resample(orig_freq=sr_asr, new_freq=expected_asr_sr)
                waveform_asr = resampler_asr(waveform_asr)
            
            if waveform_asr.shape[0] > 1: # Ensure mono
                waveform_asr = torch.mean(waveform_asr, dim=0)
            else: # Ensure 1D for processor if it was [1, N]
                waveform_asr = waveform_asr.squeeze(0)
            
            # Process and generate transcription
            inputs_asr = self.whisper_processor(waveform_asr.to(self.current_device), 
                                                sampling_rate=expected_asr_sr, 
                                                return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(inputs_asr.to(self.current_device))
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        except Exception as e_asr:
            # logger.error(f"Error during ASR for {audio_path_str}: {e_asr}. Returning empty string.", exc_info=True)
            print(f"Error during ASR for {audio_path_str}: {e_asr}. Returning empty string.")
            return "" # Fallback to empty string or original text outside this func

    def __call__(self, item):
        """
        Processes a single data item.
        `item` is expected to be a dictionary from `create_dataset_from_csv`.
        """
        print(f"DEBUG Processor: Received item keys: {list(item.keys())}")
        if 'audio_path' in item:
            print(f"DEBUG Processor: Received item['audio_path']: {item['audio_path']}")
        else:
            print(f"DEBUG Processor: 'audio_path' NOT in received item for item: {item.get('utterance_id', 'N/A')}")

        processed_features = {} # This will hold features to be added/updated
        full_audio_path_str = item['_original_audio_path_abs'] # Read from the renamed column
        original_text = item['text']

        # --- Audio Processing ---
        if self.config.audio_input_type == "raw_wav":
            # Store relative path to the 16kHz WAV file
            # The column name should come from config, e.g., config.audio_path_column_name_in_hf_dataset
            path_col_name = getattr(self.config, 'audio_path_column_name_in_hf_dataset', 'relative_audio_path')
            processed_features[path_col_name] = self._get_relative_audio_path(full_audio_path_str)
            # No actual audio waveform/features are stored in the HF dataset for this mode.

        elif self.config.audio_input_type == "hf_features":
            if not self.audio_feature_extractor_hf:
                # logger.critical(f"audio_input_type is 'hf_features' but no audio_feature_extractor_hf provided for {full_audio_path_str}. Skipping item.")
                print(f"CRITICAL: audio_input_type is 'hf_features' but no audio_feature_extractor_hf provided for {full_audio_path_str}. Skipping item.")
                return None # Signal to .map() to filter out this item
            
            # We need to import extract_hf_audio_features here or pass it in.
            # For now, assuming it's available globally or via self if we structured it differently.
            # This is a temporary solution; proper dependency injection or modular structure is better.
            try:
                # This function is defined in audio_feature_extractor.py
                # You need to ensure it's correctly imported/accessible in the context where processor.py runs
                from .audio_feature_extractor import extract_hf_audio_features # Relative import
                
                hf_audio_data = extract_hf_audio_features(full_audio_path_str, self.audio_feature_extractor_hf, self.config)
                if hf_audio_data and 'audio_input_values' in hf_audio_data:
                    processed_features['audio_input_values'] = hf_audio_data['audio_input_values']
                    if 'audio_attention_mask' in hf_audio_data:
                        processed_features['audio_attention_mask'] = hf_audio_data['audio_attention_mask']
                    else: # Ensure mask is present, e.g. all ones if extractor doesn't provide and model needs it
                        # This depends on model requirements; for now, let's assume it might be optional for some models
                        # or the collator handles it. If schema demands it, must be there.
                        # logger.debug(f"No audio_attention_mask from HF extractor for {full_audio_path_str}")
                        pass 
                else:
                    # logger.warning(f"HF audio feature extraction failed or returned no 'audio_input_values' for {full_audio_path_str}. Skipping item.")
                    print(f"Warning: HF audio feature extraction failed or returned no 'audio_input_values' for {full_audio_path_str}. Skipping item.")
                    return None # Filter item
            except ImportError:
                # logger.critical("Could not import extract_hf_audio_features. Ensure audio_feature_extractor.py is in the same directory for relative import.")
                print("CRITICAL: Could not import extract_hf_audio_features from .audio_feature_extractor. Check imports.")
                return None
            except Exception as e_hf_call:
                # logger.error(f"Unexpected error calling extract_hf_audio_features for {full_audio_path_str}: {e_hf_call}", exc_info=True)
                print(f"Error: Unexpected error calling extract_hf_audio_features for {full_audio_path_str}: {e_hf_call}")
                return None
        
        # Example for custom mel spectrogram if you add it as an audio_input_type:
        # elif self.config.audio_input_type == "custom_mel_spectrogram":
        #     from .audio_feature_extractor import extract_mel_spectrogram_custom # Relative import
        #     mel_spec = extract_mel_spectrogram_custom(full_audio_path_str, self.config)
        #     if mel_spec is not None:
        #         processed_features['mel_spectrogram'] = mel_spec.numpy() # Convert to numpy for HF Dataset
        #     else:
        #         logger.warning(f"Custom mel spectrogram extraction failed for {full_audio_path_str}. Skipping item.")
        #         return None

        else:
            # logger.critical(f"Unknown config.audio_input_type: '{self.config.audio_input_type}' for {full_audio_path_str}. Skipping item.")
            print(f"CRITICAL Error: Unknown config.audio_input_type: '{self.config.audio_input_type}' for {full_audio_path_str}. Skipping item.")
            return None

        # --- Text Processing (ASR and Tokenization) ---
        text_for_tokenization = original_text
        asr_transcription = "" # Default empty string for ASR text

        if self.use_asr:
            # logger.debug(f"Performing ASR for {full_audio_path_str}")
            asr_transcription = self._perform_asr(full_audio_path_str)
            if asr_transcription: # If ASR succeeded and produced text
                text_for_tokenization = asr_transcription
            else:
                # logger.warning(f"ASR failed or returned empty for {full_audio_path_str}. Falling back to original text: '{original_text}'.")
                print(f"Warning: ASR failed or returned empty for {full_audio_path_str}. Falling back to original text.")
        
        processed_features['asr_text'] = asr_transcription # Store ASR text regardless of fallback for tokenization

        if self.text_tokenizer:
            # Max length for tokenization from config, e.g., config.text_max_length_for_hf_dataset
            max_len = getattr(self.config, 'text_max_length_for_hf_dataset', 128)
            try:
                tokenized_output = self.text_tokenizer(text_for_tokenization, padding="max_length", 
                                                     truncation=True, max_length=max_len,
                                                     return_attention_mask=True, 
                                                     # return_tensors="np" # If HF dataset expects numpy arrays directly
                                                     )
                processed_features['input_ids'] = tokenized_output['input_ids']
                processed_features['attention_mask'] = tokenized_output['attention_mask']
            except Exception as e_tok:
                # logger.error(f"Error during text tokenization for text '{text_for_tokenization[:50]}...': {e_tok}. Setting empty lists for token IDs/mask.", exc_info=True)
                print(f"Error: Text tokenization failed for '{text_for_tokenization[:50]}...': {e_tok}. Skipping item.")
                # Depending on schema, returning None might be better if tokenization is critical
                return None 
        else:
            # logger.warning(f"text_tokenizer is None. Skipping text tokenization for {full_audio_path_str}. input_ids/attention_mask will be empty.")
            print(f"Warning: text_tokenizer is None. Skipping text tokenization for {full_audio_path_str}.")
            processed_features['input_ids'] = [] # Schema expects Sequence, so empty list is okay
            processed_features['attention_mask'] = []
        
        # --- Include original item fields required by the schema that are not modified ---
        # These should match the `build_output_hf_features_schema`
        # The label should already be in `item` from `create_dataset_from_csv`.
        # Other fields like dialogue_id, utterance_id, speaker, emotion, sentiment, original_text
        # are typically kept if they are part of the output schema.
        # The .map() function in Hugging Face Datasets will carry over columns that are:
        #   1. Present in the input item.
        #   2. Not in `remove_columns` argument of .map().
        #   3. Present in the `features` (schema) argument of .map().
        # So, we only need to return the *new* or *modified* fields from this processor function.
        # The `item` itself passed to this callable will have the original fields.
        # However, it's safer to explicitly return all fields expected by the schema if there's ambiguity.

        # For clarity, let's merge `item` (original data) with `processed_features` (new/modified data).
        # The `processed_features` will overwrite any common keys in `item`.
        # final_output_item = {**item, **processed_features}

        final_output_item = {}
        # Copy essential fields from original item that are expected in the schema
        for key in ['dialogue_id', 'utterance_id', 'speaker', 'text', 'emotion', 'sentiment', 'label']:
            if key in item:
                final_output_item[key] = item[key]
        
        # TEMPORARY: Also pass through the _original_audio_path_abs to match temporary schema
        if '_original_audio_path_abs' in item:
            final_output_item['_original_audio_path_abs'] = item['_original_audio_path_abs']

        # Add new/modified features (which might include the relative audio path, tokenized text etc.)
        final_output_item.update(processed_features)
        
        # Ensure label is present (it should be from item, but double-check)
        if 'label' not in final_output_item and 'label' in item: # Defensive check
            final_output_item['label'] = item['label']
        elif 'label' not in final_output_item:
            # logger.critical(f"'label' not found in final output item for {full_audio_path_str}. This should not happen. Setting to -1.")
            print(f"CRITICAL Warning: 'label' not found in final output item for {full_audio_path_str}. Setting -1.")
            final_output_item['label'] = -1 # Default/error label
            
        print(f"DEBUG Processor: Returning final_output_item keys: {list(final_output_item.keys())}")
        return final_output_item

    # Note: The original `process_dataset_item` had a try-except around the whole thing.
    # It's generally better to handle specific errors in sub-operations as done above.
    # If a global catch is needed, it can be added here or in the orchestrator's .map() call.

# Example Usage (Conceptual - requires BaseConfig and other modules/files)
if __name__ == '__main__':
    print("Testing processor.py...")

    # This test requires a more complete setup: BaseConfig, dummy audio files,
    # pre-loaded models/tokenizers, and the audio_feature_extractor module.

    # --- Mocking/Dummy Setup (Simplified) ---
    class DummyBaseConfig:
        def __init__(self, audio_type="hf_features"):
            self.device = 'cpu'
            self.processed_audio_output_dir = Path("temp_test_data_proc/processed_wavs") # Used by _get_relative_audio_path
            self.audio_input_type = audio_type # "raw_wav" or "hf_features"
            self.audio_path_column_name_in_hf_dataset = "relative_audio_path"
            self.audio_encoder_model_name = "facebook/wav2vec2-base" # For HF audio extractor
            self.sample_rate = 16000 # Expected by wav2vec2-base

            self.use_asr_for_text_generation_in_hf_dataset = False # Set to True to test ASR
            self.asr_model_name_for_hf_dataset = "openai/whisper-tiny"
            # self.asr_model_name = "openai/whisper-tiny" # from original build_hf_dataset

            self.text_encoder_model_name = "distilroberta-base"
            self.text_max_length_for_hf_dataset = 64
            # self.logger = logging.getLogger("test_processor")
            # logging.basicConfig(level=logging.INFO)

            # Dummy data paths for testing
            self.test_audio_dir = Path("temp_test_data_proc/dummy_audio_files")
            self.test_audio_dir.mkdir(parents=True, exist_ok=True)
            self.dummy_audio_path = self.test_audio_dir / "sample1.wav"
            if not self.dummy_audio_path.exists():
                torchaudio.save(str(self.dummy_audio_path), torch.randn(1, 16000), 16000)
            
            (self.processed_audio_output_dir / "train").mkdir(parents=True, exist_ok=True)
            # Path for relative path testing
            self.dummy_processed_audio_path = self.processed_audio_output_dir / "train" / "dia0_utt0.wav"
            if not self.dummy_processed_audio_path.exists():
                 torchaudio.save(str(self.dummy_processed_audio_path), torch.randn(1,16000), 16000)

    # Need to mock the audio_feature_extractor imports for this standalone test
    # In a real scenario, these would be actual imports.
    mock_audio_feature_extractor_module = True
    if mock_audio_feature_extractor_module:
        class MockAudioFeatureExtractorModule:
            def extract_hf_audio_features(self, path_str, extractor, config_obj):
                # logger.info(f"[MOCK] extract_hf_audio_features called for {path_str}")
                print(f"[MOCK] extract_hf_audio_features called for {path_str}")
                # Return dummy data that matches expected structure
                return {
                    'audio_input_values': np.random.rand(config_obj.sample_rate).astype(np.float32), # 1 sec audio
                    'audio_attention_mask': np.ones(config_obj.sample_rate).astype(np.int8)
                }
        # This is a hack for testing; normally, you'd use Python's import system.
        import sys
        sys.modules['.audio_feature_extractor'] = MockAudioFeatureExtractorModule()
        # A more robust way for tests is using unittest.mock

    # -- Instantiate components --
    cfg_hf_test = DummyBaseConfig(audio_type="hf_features")
    cfg_raw_test = DummyBaseConfig(audio_type="raw_wav")
    
    # Text Tokenizer
    try:
        text_tok = AutoTokenizer.from_pretrained(cfg_hf_test.text_encoder_model_name)
    except Exception as e:
        print(f"Failed to load text tokenizer for test: {e}")
        text_tok = None

    # Audio Feature Extractor (HF)
    # In real orchestrator, this is loaded once and passed.
    # Here, we'd use get_hf_audio_feature_extractor from the actual module if not mocking.
    try:
        from .audio_feature_extractor import get_hf_audio_feature_extractor # Test actual import if possible
        audio_feat_ext = get_hf_audio_feature_extractor(cfg_hf_test)
    except (ImportError, Exception) as e:
        print(f"Failed to load actual HF audio feature extractor for test: {e}. Using None.")
        audio_feat_ext = None # Fallback if actual can't be loaded

    # ASR Models (Whisper) - Optional, only if testing ASR path
    w_processor = None
    w_model = None
    if cfg_hf_test.use_asr_for_text_generation_in_hf_dataset:
        try:
            asr_model_name = cfg_hf_test.asr_model_name_for_hf_dataset
            w_processor = WhisperProcessor.from_pretrained(asr_model_name)
            w_model = WhisperForConditionalGeneration.from_pretrained(asr_model_name).to(cfg_hf_test.device)
            w_model.eval()
            print(f"Whisper models ({asr_model_name}) loaded for ASR testing.")
        except Exception as e:
            print(f"Failed to load Whisper models for test: {e}. ASR tests might be affected.")

    # --- Create Processors ---
    processor_hf = DatasetItemProcessor(
        cfg_hf_test, 
        audio_feature_extractor_hf=audio_feat_ext, 
        text_tokenizer_hf=text_tok,
        whisper_processor_hf=w_processor,
        whisper_model_hf=w_model
    )
    processor_raw = DatasetItemProcessor(
        cfg_raw_test, 
        text_tokenizer_hf=text_tok # Raw wav still needs text tokenization
    )

    # --- Dummy input item (from csv_loader) ---
    dummy_item = {
        'dialogue_id': 0, 'utterance_id': 0, 'speaker': 'Test Dummy', 
        'text': 'This is a test sentence for processing.',
        'emotion': 'neutral', 'sentiment': 'neutral',
        'audio_path': str(cfg_hf_test.dummy_processed_audio_path), # Use a path within processed_audio_output_dir for rel path test
        'label': 0 
    }

    # --- Test HF Features Processing ---
    print("\n--- Testing processor with audio_input_type: hf_features ---")
    if audio_feat_ext and text_tok:
        processed_item_hf = processor_hf(dummy_item.copy()) # Pass a copy
        if processed_item_hf:
            print("HF Processed item keys:", processed_item_hf.keys())
            assert 'audio_input_values' in processed_item_hf
            assert 'input_ids' in processed_item_hf
            assert processed_item_hf['asr_text'] is not None # Should exist even if empty
            print("Sample processed (hf_features):")
            for k, v in processed_item_hf.items():
                if isinstance(v, (list, np.ndarray)) and len(v) > 10:
                    print(f"  {k}: shape {getattr(v, 'shape', len(v))}, first few: {v[:3]}...")
                else:
                    print(f"  {k}: {v}")
        else:
            print("HF processing returned None (item skipped).")
    else:
        print("Skipping HF processing test due to missing audio_feat_ext or text_tok.")

    # --- Test Raw WAV Path Processing ---
    print("\n--- Testing processor with audio_input_type: raw_wav ---")
    if text_tok:
        processed_item_raw = processor_raw(dummy_item.copy()) # Pass a copy
        if processed_item_raw:
            print("Raw WAV Processed item keys:", processed_item_raw.keys())
            path_col_name = cfg_raw_test.audio_path_column_name_in_hf_dataset
            assert path_col_name in processed_item_raw
            assert processed_item_raw[path_col_name] == "train/dia0_utt0.wav" # Expected relative path
            assert 'audio_input_values' not in processed_item_raw
            assert 'input_ids' in processed_item_raw
            print("Sample processed (raw_wav):")
            for k,v in processed_item_raw.items(): print(f"  {k}: {v[:50] if isinstance(v,str) and len(v)>50 else v}")
        else:
            print("Raw WAV processing returned None (item skipped).")
    else:
        print("Skipping Raw WAV processing test due to missing text_tok.")
    
    # --- Test ASR Path (if enabled and models loaded) ---
    if cfg_hf_test.use_asr_for_text_generation_in_hf_dataset and w_processor and w_model and audio_feat_ext and text_tok:
        print("\n--- Testing processor with ASR enabled ---")
        # Ensure use_asr is True on the processor's config if it was changed for the instance
        processor_hf.use_asr = True # Or re-init processor_hf with a config where use_asr is true
        
        dummy_item_asr = dummy_item.copy()
        dummy_item_asr['audio_path'] = str(cfg_hf_test.dummy_audio_path) # Use a generic path for ASR test

        processed_item_asr = processor_hf(dummy_item_asr)
        if processed_item_asr:
            print(f"ASR text: '{processed_item_asr['asr_text']}'")
            print(f"Original text: '{processed_item_asr['text']}'") # Should be original from item
            print(f"Tokenized input_ids (first 10): {processed_item_asr['input_ids'][:10]}")
            assert processed_item_asr['asr_text'] != "" # Expect some ASR output from dummy audio
            # Further assertions can check if input_ids are based on asr_text
        else:
            print("ASR processing returned None.")

    # Clean up dummy files (optional)
    # import shutil
    # shutil.rmtree("temp_test_data_proc", ignore_errors=True)
    # if mock_audio_feature_extractor_module:
    #     del sys.modules['.audio_feature_extractor'] # Clean up mock

    print("\nprocessor.py test run finished.") 