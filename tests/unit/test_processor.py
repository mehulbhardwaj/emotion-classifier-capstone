"""Unit tests for the dataset processor."""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# Import the processor to test
from scripts.prepare_dataset.processor import DatasetItemProcessor
from transformers import AutoTokenizer, AutoFeatureExtractor

# Import torchaudio for audio processing
import torchaudio

# Mock torchaudio.load to avoid file I/O during tests
@patch('torchaudio.load')
def mock_audio_load(mock_load):
    mock_load.return_value = (torch.randn(1, 16000), 16000)
    return mock_load

# Fixture for test configuration
class MockConfig:
    def __init__(self, audio_type="hf_features"):
        self.device = 'cpu'
        self.processed_audio_output_dir = Path("temp_test_data_proc/processed_wavs")
        self.audio_input_type = audio_type  # "raw_wav" or "hf_features"
        self.audio_path_column_name_in_hf_dataset = "relative_audio_path"
        self.audio_encoder_model_name = "facebook/wav2vec2-base"
        self.sample_rate = 16000
        self.use_asr_for_text_generation_in_hf_dataset = False
        self.asr_model_name_for_hf_dataset = "openai/whisper-tiny"
        self.text_encoder_model_name = "distilroberta-base"
        self.text_max_length_for_hf_dataset = 64
        self.label_encoder = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
        self.num_classes = len(self.label_encoder)
        
        # Set up test file paths without creating actual files
        self.test_audio_dir = Path("temp_test_data_proc/dummy_audio_files")
        self.dummy_audio_path = self.test_audio_dir / "sample1.wav"
        self.dummy_processed_audio_path = self.processed_audio_output_dir / "train" / "dia0_utt0.wav"

# Fixture for test data
@pytest.fixture
def test_item():
    """Fixture that returns a sample dataset item for testing."""
    return {
        'dialogue_id': 'd001',
        'utterance_id': 'd001_utt00',
        'speaker': 'Speaker1',
        'text': 'This is a test utterance.',
        'emotion': 'neutral',
        'sentiment': 'neutral',
        'relative_audio_path': 'path/to/audio.wav',
        'label': 0,
        '_original_audio_path_abs': str(Path('temp_test_data_proc/processed_wavs/train/dia0_utt0.wav').absolute())
    }

# Fixture for the processor
@pytest.fixture
def processor():
    # Create a mock config
    config = MockConfig()
    
    # Create mock tokenizer and feature extractor
    mock_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    mock_feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    
    # Mock the extract_hf_audio_features function and torchaudio.load
    with patch('scripts.prepare_dataset.audio_feature_extractor.extract_hf_audio_features') as mock_extract, \
         patch('torchaudio.load') as mock_audio_load_func:
        
        # Setup mock return values
        mock_extract.return_value = {
            'audio_input_values': torch.randn(16000),
            'audio_attention_mask': torch.ones(16000, dtype=torch.long)
        }
        mock_audio_load_func.return_value = (torch.randn(1, 16000), 16000)
        
        # Create the processor with mocks
        processor = DatasetItemProcessor(
            config=config,
            audio_feature_extractor_hf=mock_feature_extractor,
            text_tokenizer_hf=mock_tokenizer,
            whisper_processor_hf=None,
            whisper_model_hf=None
        )
    
    return processor

def test_processor_initialization(processor):
    """Test that the processor initializes correctly."""
    assert processor is not None
    assert hasattr(processor, 'text_tokenizer') and processor.text_tokenizer is not None
    assert hasattr(processor, 'audio_feature_extractor_hf') and processor.audio_feature_extractor_hf is not None
    assert hasattr(processor, 'use_asr')  # Check if use_asr is an attribute
    assert processor.config is not None
    assert hasattr(processor.config, 'audio_input_type')

def test_process_text(processor, test_item):
    """Test text processing."""
    # Mock the tokenizer call
    with patch.object(processor.text_tokenizer, '__call__') as mock_tokenize:
        mock_tokenize.return_value = {
            "input_ids": torch.tensor([[101, 2023, 2003, 1037, 3231, 1010, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        }
        
        # Call the processor's __call__ method which will use _process_text internally
        processed = processor(test_item)
    
    # Verify output structure
    assert isinstance(processed, dict)
    assert "input_ids" in processed
    assert "attention_mask" in processed
    assert isinstance(processed["input_ids"], list) or isinstance(processed["input_ids"], torch.Tensor)
    assert isinstance(processed["attention_mask"], list) or isinstance(processed["attention_mask"], torch.Tensor)

def test_process_audio_hf_features(processor, test_item):
    """Test audio processing with HF features."""
    processor.config.audio_input_type = "hf_features"
    
    # Call the processor's __call__ method which will process audio internally
    # The mocks are already set up in the processor fixture
    processed = processor(test_item)
    
    # Verify the output structure
    assert isinstance(processed, dict)
    assert "audio_input_values" in processed
    assert isinstance(processed["audio_input_values"], (list, torch.Tensor, np.ndarray))

def test_process_audio_raw_wav(processor, test_item):
    """Test audio processing with raw WAV input."""
    processor.config.audio_input_type = "raw_wav"
    
    # Call the processor's __call__ method which will process audio internally
    # The mocks are already set up in the processor fixture
    processed = processor(test_item)
    
    # Verify the output structure
    assert isinstance(processed, dict)
    assert "relative_audio_path" in processed
    assert isinstance(processed["relative_audio_path"], str)

def test_process_label(processor, test_item):
    """Test label processing."""
    # Call the processor's __call__ method which will process the label internally
    processed = processor(test_item)
    
    # The processor should have processed the label from the test_item
    assert isinstance(processed, dict)
    assert "label" in processed
    assert isinstance(processed["label"], (int, np.integer))

def test_call_method(processor, test_item):
    """Test the __call__ method that processes a complete item."""
    # Call the processor - mocks are set up in the fixture
    processed = processor(test_item)
    
    # Check that all expected keys are present
    expected_keys = ["input_ids", "attention_mask"]
    if processor.config.audio_input_type == "hf_features":
        expected_keys.append("audio_input_values")
    else:
        expected_keys.append("relative_audio_path")
    expected_keys.append("label")
    
    for key in expected_keys:
        assert key in processed, f"Expected key '{key}' not found in processed item"

# Add more test cases for error handling and edge cases
