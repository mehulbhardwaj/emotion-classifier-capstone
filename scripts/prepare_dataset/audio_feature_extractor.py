import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor
from datasets import Features, Sequence, Value, ClassLabel # Added ClassLabel

# from configs.base_config import BaseConfig # Placeholder
# import logging
# logger = logging.getLogger(__name__)

def extract_mel_spectrogram_custom(audio_path_or_waveform, config, input_sample_rate=None):
    """
    Extract log-mel spectrogram from audio file or waveform using custom logic.
    This is based on the original extract_mel_spectrogram function.
    Uses cfg.sample_rate, cfg.hop_length, cfg.n_mels (num_mel_bins), cfg.n_fft.
    
    Args:
        audio_path_or_waveform (str or Path or torch.Tensor): Path to audio file or loaded waveform tensor.
        config (BaseConfig): Configuration object containing audio processing parameters.
        input_sample_rate (int, optional): Sample rate of the input waveform (if waveform is passed).
        
    Returns:
        torch.Tensor or None: Log-mel spectrogram tensor [time, n_mels] or None if error.
    """
    try:
        if isinstance(audio_path_or_waveform, (str, Path)):
            try:
                import librosa # Keep librosa for robust loading as in original
                waveform_lib, sample_rate_lib = librosa.load(str(audio_path_or_waveform), sr=config.sample_rate, mono=True)
                waveform = torch.from_numpy(waveform_lib)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0) # Add channel dim: [1, num_samples]
                current_sample_rate = sample_rate_lib
            except Exception as e_librosa:
                # logger.error(f"Error loading {audio_path_or_waveform} with librosa: {e_librosa}")
                print(f"Error loading {audio_path_or_waveform} with librosa: {e_librosa}")
                return None
        elif torch.is_tensor(audio_path_or_waveform):
            waveform = audio_path_or_waveform
            current_sample_rate = input_sample_rate if input_sample_rate is not None else config.sample_rate
            
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0) # Ensure [channel, time]
            
            if current_sample_rate != config.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=current_sample_rate, new_freq=config.sample_rate)
                waveform = resampler(waveform)
            
            if waveform.shape[0] > 1: # Ensure mono
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        else:
            raise TypeError("audio_path_or_waveform must be a path or a tensor.")

        if waveform.ndim == 1: # Should be [1, time] by now
            waveform = waveform.unsqueeze(0)

        # Padding if waveform is shorter than n_fft
        n_fft = getattr(config, 'n_fft', 400) # Default from original
        if waveform.shape[-1] < n_fft:
            # logger.warning(f"Waveform is shorter ({waveform.shape[-1]} samples) than n_fft ({n_fft}). Padding.")
            print(f"Warning: Waveform is shorter ({waveform.shape[-1]} samples) than n_fft ({n_fft}). Padding.")
            padding_needed = n_fft - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels, # renamed from num_mel_bins in original BaseConfig use
            power=2.0
        )
        
        mel_spec = mel_transform(waveform)
        log_mel_spec = torch.log(mel_spec + 1e-9)
        
        return log_mel_spec.squeeze(0).T # [time, n_mels]
    except Exception as e:
        # logger.error(f"Error extracting mel spectrogram for {audio_path_or_waveform}: {e}", exc_info=True)
        print(f"Error extracting mel spectrogram for {audio_path_or_waveform}: {e}")
        return None

def get_hf_audio_feature_extractor(config):
    """
    Initializes and returns a Hugging Face audio feature extractor based on the config.
    """
    if not config.audio_encoder_model_name:
        # logger.error("audio_encoder_model_name not set in config, cannot get HF feature extractor.")
        print("ERROR: audio_encoder_model_name not set in config, cannot get HF feature extractor.")
        return None
    try:
        extractor = AutoFeatureExtractor.from_pretrained(config.audio_encoder_model_name)
        # logger.info(f"Hugging Face audio feature extractor loaded for: {config.audio_encoder_model_name}")
        print(f"Hugging Face audio feature extractor loaded for: {config.audio_encoder_model_name}")
        return extractor
    except Exception as e:
        # logger.error(f"Could not load AutoFeatureExtractor for {config.audio_encoder_model_name}: {e}", exc_info=True)
        print(f"CRITICAL ERROR: Could not load AutoFeatureExtractor for {config.audio_encoder_model_name}: {e}")
        return None # Critical error

def extract_hf_audio_features(full_audio_path_str: str, feature_extractor_hf, config):
    """
    Loads an audio file, preprocesses it, and extracts features using a Hugging Face audio feature extractor.
    Returns a dictionary containing 'audio_input_values' and optionally 'audio_attention_mask'.
    Returns None if feature extraction fails.
    """
    try:
        waveform, sr = torchaudio.load(full_audio_path_str)
        
        # Ensure correct sample rate for the HF feature extractor (typically config.sample_rate, e.g., 16kHz)
        if sr != feature_extractor_hf.sampling_rate: # Check against extractor's expected SR
            # logger.info(f"Resampling {full_audio_path_str} from {sr}Hz to {feature_extractor_hf.sampling_rate}Hz for HF extractor.")
            print(f"Resampling {full_audio_path_str} from {sr}Hz to {feature_extractor_hf.sampling_rate}Hz for HF extractor.")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=feature_extractor_hf.sampling_rate)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1: # Ensure mono
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # HF extractors often expect a 1D numpy array
        waveform_np = waveform.squeeze(0).numpy()

        hf_processed_audio = feature_extractor_hf(
            waveform_np,
            sampling_rate=feature_extractor_hf.sampling_rate, 
            return_tensors="np", # Store as numpy arrays in HF dataset
            padding=False # Padding typically handled by collate_fn or later in processor
        )
        
        # 'input_values' or 'input_features' based on the model type
        feature_key = 'input_values' if 'input_values' in hf_processed_audio else 'input_features'
        if feature_key not in hf_processed_audio:
            # logger.error(f"Neither 'input_values' nor 'input_features' in HF audio output. Keys: {hf_processed_audio.keys()}")
            print(f"ERROR: Neither 'input_values' nor 'input_features' in HF audio output for {full_audio_path_str}. Keys: {hf_processed_audio.keys()}")
            return None
            
        audio_data = hf_processed_audio[feature_key]
        if audio_data.ndim > 1: # Squeeze if batch dim [1, N] or [1, N, M]
            audio_data = audio_data.squeeze(0)
        
        extracted_features = {'audio_input_values': audio_data}

        if 'attention_mask' in hf_processed_audio and hf_processed_audio.attention_mask is not None:
            mask_data = hf_processed_audio.attention_mask
            if mask_data.ndim > 1: # Squeeze if batch dim [1, N]
                 mask_data = mask_data.squeeze(0)
            extracted_features['audio_attention_mask'] = mask_data
        
        return extracted_features

    except Exception as e_feat_ext:
        # logger.error(f"Error during Hugging Face audio feature extraction for {full_audio_path_str}: {e_feat_ext}", exc_info=True)
        print(f"Error during Hugging Face audio feature extraction for {full_audio_path_str}: {e_feat_ext}")
        return None

def build_output_hf_features_schema(config):
    """
    Defines the Hugging Face Dataset `Features` schema based on the configuration.
    This was part of `build_and_process_split_dataset` in the original script.
    """
    schema_dict = {
        'dialogue_id': Value("int32"), 
        'utterance_id': Value("int32"),
        'speaker': Value("string"), 
        'text': Value("string"), # Original text from CSV
        'emotion': Value("string"), 
        'sentiment': Value("string"),
        # 'label': Value("int64"), # Handled by ClassLabel now if num_classes is known
        'input_ids': Sequence(Value("int32")), # For tokenized text
        'attention_mask': Sequence(Value("int8")), # For tokenized text attention
        'asr_text': Value("string"), # Text from ASR, if used
        '_original_audio_path_abs': Value("string")
    }

    # Add label with ClassLabel if possible
    if hasattr(config, 'num_classes') and config.num_classes > 0 and hasattr(config, 'id_to_label_map'):
        # Ensure id_to_label_map keys are integers if they represent class indices
        # And that names are strings.
        # For ClassLabel, names should be a list of strings in order of their IDs.
        # Example: if id_to_label_map = {0: 'neutral', 1: 'joy'}, names = ['neutral', 'joy']
        try:
            # Create names list assuming id_to_label_map keys are 0-indexed and contiguous
            label_names = [config.id_to_label_map[i] for i in range(config.num_classes)]
            schema_dict['label'] = ClassLabel(num_classes=config.num_classes, names=label_names)
            # logger.info(f"Using ClassLabel for 'label' with {config.num_classes} classes: {label_names}")
            print(f"Using ClassLabel for 'label' with {config.num_classes} classes: {label_names}")
        except Exception as e_cl:
            # logger.warning(f"Could not create ClassLabel, falling back to Value('int64') for 'label'. Error: {e_cl}")
            print(f"Warning: Could not create ClassLabel (num_classes={config.num_classes}, id_to_label_map might be an issue), falling back to Value('int64') for 'label'. Error: {e_cl}")
            schema_dict['label'] = Value("int64")
    else:
        # logger.info("num_classes or id_to_label_map not available in config, using Value('int64') for 'label'.")
        print("Warning: num_classes or id_to_label_map not available in config, using Value('int64') for 'label'.")
        schema_dict['label'] = Value("int64")

    if config.audio_input_type == "raw_wav":
        # Use the specific column name from config for the relative audio path
        path_col_name = getattr(config, 'audio_path_column_name_in_hf_dataset', 'relative_audio_path')
        schema_dict[path_col_name] = Value("string")
    elif config.audio_input_type == "hf_features":
        # For HF features like Wav2Vec2, typically 'input_values' or 'input_features'
        # These are usually 1D float arrays after processing
        schema_dict['audio_input_values'] = Sequence(Value("float32"))
        # audio_attention_mask is also often a 1D int array (0s and 1s)
        # This schema assumes it will always be present if audio_input_type is hf_features
        # and the extractor provides it. If it can be optional, HF Datasets might need more care
        # or the processor should ensure a default (e.g., all ones if no mask).
        schema_dict['audio_attention_mask'] = Sequence(Value("int8"))
    # elif config.audio_input_type == "mel_spectrogram": # If you add this option
        # schema_dict['mel_spectrogram'] = Sequence(Sequence(Value("float32"))) # [time, n_mels]
    else:
        # logger.error(f"Unsupported audio_input_type for schema: {config.audio_input_type}")
        print(f"ERROR: Unsupported audio_input_type for schema: {config.audio_input_type}")

    return Features(schema_dict)

# Example of how you might set up a dummy config for testing this module
if __name__ == '__main__':
    print("Testing audio_feature_extractor.py...")

    class DummyConfigForAudio:
        def __init__(self):
            self.sample_rate = 16000
            self.hop_length = 160 # 10ms hop for 16kHz SR
            self.n_mels = 80
            self.n_fft = 400    # 25ms window for 16kHz SR
            self.audio_encoder_model_name = "facebook/wav2vec2-base" # For HF extractor testing
            self.audio_input_type = "hf_features" # or "raw_wav"
            self.audio_path_column_name_in_hf_dataset = "relative_audio_path"
            # For schema building test
            self.num_classes = 7 # Example for MELD
            self.id_to_label_map = {0:"neutral", 1:"joy", 2:"sadness", 3:"anger", 4:"surprise", 5:"fear", 6:"disgust"}
            # self.logger = logging.getLogger("test_audio_feature_extractor")
            # logging.basicConfig(level=logging.INFO)

            # Create a dummy audio file for testing
            self.test_data_dir = Path("temp_audio_test_data")
            self.test_data_dir.mkdir(parents=True, exist_ok=True)
            self.dummy_audio_path_str = str(self.test_data_dir / "dummy_audio.wav")
            if not Path(self.dummy_audio_path_str).exists():
                 # Create a 1-second mono dummy audio at 16kHz
                dummy_waveform = torch.randn(1, self.sample_rate)
                torchaudio.save(self.dummy_audio_path_str, dummy_waveform, self.sample_rate)

    cfg_audio = DummyConfigForAudio()

    # Test 1: Mel Spectrogram Extraction (from file)
    print("\n--- Testing Mel Spectrogram (from file) ---")
    mel_spec = extract_mel_spectrogram_custom(cfg_audio.dummy_audio_path_str, cfg_audio)
    if mel_spec is not None:
        print(f"Mel spectrogram (custom) extracted. Shape: {mel_spec.shape}") # Expected: [time_frames, n_mels]
        assert mel_spec.shape[1] == cfg_audio.n_mels
    else:
        print("Mel spectrogram extraction failed.")

    # Test 2: Get HF Audio Feature Extractor
    print("\n--- Testing Get HF Audio Feature Extractor ---")
    hf_extractor = get_hf_audio_feature_extractor(cfg_audio)
    if hf_extractor:
        print(f"HF Extractor sampling rate: {hf_extractor.sampling_rate}")
        assert hf_extractor.sampling_rate == cfg_audio.sample_rate # Wav2Vec2-base is 16k
    else:
        print("Failed to get HF feature extractor.")

    # Test 3: Extract HF Audio Features (if extractor was loaded)
    if hf_extractor:
        print("\n--- Testing Extract HF Audio Features ---")
        hf_features = extract_hf_audio_features(cfg_audio.dummy_audio_path_str, hf_extractor, cfg_audio)
        if hf_features and 'audio_input_values' in hf_features:
            print(f"HF audio_input_values extracted. Shape: {hf_features['audio_input_values'].shape}") # Typically 1D array
            assert hf_features['audio_input_values'].ndim == 1
            if 'audio_attention_mask' in hf_features:
                 print(f"HF audio_attention_mask extracted. Shape: {hf_features['audio_attention_mask'].shape}")
                 assert hf_features['audio_attention_mask'].shape == hf_features['audio_input_values'].shape
        else:
            print("HF audio feature extraction failed or returned unexpected structure.")

    # Test 4: Build Features Schema (for hf_features)
    print("\n--- Testing Build Output HF Features Schema (hf_features) ---")
    cfg_audio.audio_input_type = "hf_features"
    schema_hf = build_output_hf_features_schema(cfg_audio)
    print(f"Schema for hf_features: {schema_hf}")
    assert 'audio_input_values' in schema_hf
    assert 'audio_attention_mask' in schema_hf
    assert isinstance(schema_hf['label'], ClassLabel)

    # Test 5: Build Features Schema (for raw_wav)
    print("\n--- Testing Build Output HF Features Schema (raw_wav) ---")
    cfg_audio.audio_input_type = "raw_wav"
    schema_raw = build_output_hf_features_schema(cfg_audio)
    print(f"Schema for raw_wav: {schema_raw}")
    assert cfg_audio.audio_path_column_name_in_hf_dataset in schema_raw
    assert 'audio_input_values' not in schema_raw

    # Clean up dummy audio data
    # import shutil
    # shutil.rmtree(cfg_audio.test_data_dir, ignore_errors=True)
    print("\naudio_feature_extractor.py test run finished.") 