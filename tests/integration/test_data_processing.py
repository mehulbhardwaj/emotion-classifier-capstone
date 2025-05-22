"""Integration tests for the data processing pipeline."""
import os
import tempfile
import pytest
import torch
from pathlib import Path
from datasets import Dataset

# Import the orchestrator and related classes
from scripts.prepare_dataset.orchestrator import MELDDatasetOrchestrator
from scripts.prepare_dataset.processor import DatasetItemProcessor
from scripts.prepare_dataset.audio_feature_extractor import get_hf_audio_feature_extractor
from transformers import AutoTokenizer, AutoFeatureExtractor

# Fixture for test configuration
class TestConfig:
    def __init__(self):
        # Create a temporary directory for test data
        self.data_root = Path(tempfile.mkdtemp())
        self.raw_data_dir = self.data_root / "raw"
        self.processed_audio_output_dir = self.data_root / "processed_audio_16kHz_mono"
        self.processed_hf_dataset_dir = self.data_root / "processed_hf_datasets"
        
        # Create necessary directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_audio_output_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.processed_audio_output_dir / "dev").mkdir(parents=True, exist_ok=True)
        self.processed_hf_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.text_encoder_model_name = "distilroberta-base"
        self.audio_encoder_model_name = "facebook/wav2vec2-base"
        self.audio_input_type = "hf_features"  # or "raw_wav"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_path_column_name_in_hf_dataset = "relative_audio_path"
        
        # Dataset specific
        self.label_encoder = {
            "neutral": 0, "joy": 1, "sadness": 2, 
            "anger": 3, "surprise": 4, "fear": 5, "disgust": 6
        }
        self.num_classes = len(self.label_encoder)
        self.text_max_length_for_hf_dataset = 64
        
        # Test data parameters
        self.sample_rate = 16000
        self.n_mels = 80
        self.hop_length = 160
        self.n_fft = 400
        
        # ASR configuration
        self.use_asr_for_text_generation_in_hf_dataset = False
        self.asr_model_name_for_hf_dataset = "openai/whisper-tiny"

# Fixture for test data
def create_test_audio_file(file_path: Path, duration: float = 1.0, sample_rate: int = 16000):
    """Create a test audio file with random noise."""
    import numpy as np
    import soundfile as sf
    
    # Generate random noise
    samples = np.random.randn(int(duration * sample_rate))
    
    # Save as WAV file
    sf.write(str(file_path), samples, sample_rate)
    return file_path

# Fixture for test dataset
def create_test_dataset(config, num_samples: int = 5):
    """Create a test dataset with dummy data."""
    data = {
        "dialogue_id": [f"d{i:03d}" for i in range(num_samples)],
        "utterance_id": [f"d{i:03d}_utt00" for i in range(num_samples)],
        "speaker": ["Speaker1" if i % 2 == 0 else "Speaker2" for i in range(num_samples)],
        "utterance": [f"Test utterance {i}" for i in range(num_samples)],
        "emotion": ["neutral", "joy", "sadness", "anger", "surprise"][:num_samples],
        "sentiment": ["neutral", "positive", "negative", "negative", "positive"][:num_samples],
        "dialogue_emotion": ["neutral"] * num_samples,
        "utterance_emotion": ["neutral"] * num_samples,
        "dialogue_sentiment": ["neutral"] * num_samples,
        "utterance_sentiment": ["neutral"] * num_samples,
        "season": ["S01"] * num_samples,
        "episode": ["E01"] * num_samples,
        "start_time": ["00:00:00"] * num_samples,
        "end_time": ["00:00:03"] * num_samples,
        "video_id": [f"video_{i:03d}" for i in range(num_samples)],
        "relative_audio_path": [str(config.processed_audio_output_dir / f"train/audio_{i:03d}.wav") 
                               for i in range(num_samples)],
    }
    
    # Create audio files
    for audio_path in data["relative_audio_path"]:
        create_test_audio_file(Path(audio_path))
    
    return Dataset.from_dict(data)

# Fixture for the orchestrator
@pytest.fixture(scope="module")
def test_config():
    return TestConfig()

@pytest.fixture(scope="module")
def test_dataset(test_config):
    return create_test_dataset(test_config)

# Test cases
def test_orchestrator_initialization(test_config):
    """Test that the orchestrator initializes correctly."""
    # Test with default config (ASR disabled)
    orchestrator = MELDDatasetOrchestrator(test_config)
    
    # Check that components are initialized
    assert orchestrator.text_tokenizer_hf is not None
    assert orchestrator.hf_audio_feature_extractor is not None
    assert orchestrator.whisper_processor_hf is None  # ASR is disabled by default
    assert orchestrator.whisper_model_hf is None     # ASR is disabled by default
    assert not orchestrator.actual_use_asr  # ASR should be disabled
    
    # Test with ASR enabled
    test_config.use_asr_for_text_generation_in_hf_dataset = True
    orchestrator_with_asr = MELDDatasetOrchestrator(test_config)
    
    # Check that ASR components are initialized
    assert orchestrator_with_asr.whisper_processor_hf is not None
    assert orchestrator_with_asr.whisper_model_hf is not None
    assert orchestrator_with_asr.actual_use_asr  # ASR should be enabled

def test_audio_feature_extraction(test_config, test_dataset):
    """Test audio feature extraction."""
    # Load the tokenizer and feature extractor
    text_tokenizer = AutoTokenizer.from_pretrained(test_config.text_encoder_model_name)
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(test_config.audio_encoder_model_name)
    
    # Create a processor instance
    processor = DatasetItemProcessor(
        config=test_config,
        text_tokenizer_hf=text_tokenizer,
        audio_feature_extractor_hf=audio_feature_extractor,
        whisper_processor_hf=None,
        whisper_model_hf=None
    )
    
    # Process a single item
    item = test_dataset[0]
    processed_item = processor(item)
    
    # Check that the output has the expected keys
    expected_keys = ["input_ids", "attention_mask"]
    if test_config.audio_input_type == "hf_features":
        expected_keys.extend(["audio_features"])
    else:
        expected_keys.extend(["input_values"])
    
    for key in expected_keys:
        assert key in processed_item, f"Expected key '{key}' not found in processed item"
    
    # Check that the text was tokenized correctly
    assert isinstance(processed_item["input_ids"], torch.Tensor)
    assert isinstance(processed_item["attention_mask"], torch.Tensor)
    assert len(processed_item["input_ids"].shape) == 1  # 1D tensor
    
    # Check that the audio features have the expected shape
    if test_config.audio_input_type == "hf_features":
        assert "audio_features" in processed_item
        assert isinstance(processed_item["audio_features"], torch.Tensor)
        assert len(processed_item["audio_features"].shape) == 1  # 1D feature vector
    else:
        assert "input_values" in processed_item
        assert isinstance(processed_item["input_values"], torch.Tensor)
        assert len(processed_item["input_values"].shape) == 1  # 1D audio waveform

def test_asr_processing(test_config, test_dataset):
    """Test ASR text generation."""
    # Skip if ASR dependencies are not available
    pytest.importorskip("transformers")
    
    # Enable ASR
    test_config.use_asr_for_text_generation_in_hf_dataset = True
    
    # Create a processor with ASR
    processor = DatasetItemProcessor(
        config=test_config,
        text_tokenizer_hf=AutoTokenizer.from_pretrained(test_config.text_encoder_model_name),
        audio_feature_extractor_hf=AutoFeatureExtractor.from_pretrained(test_config.audio_encoder_model_name),
        whisper_processor_hf=WhisperProcessor.from_pretrained(test_config.asr_model_name_for_hf_dataset),
        whisper_model_hf=WhisperForConditionalGeneration.from_pretrained(test_config.asr_model_name_for_hf_dataset).to(test_config.device)
    )
    
    # Process a single item
    item = test_dataset[0]
    with patch("torchaudio.load") as mock_load:
        # Mock audio loading to return a valid audio tensor
        mock_load.return_value = (torch.randn(1, 16000), 16000)
        processed_item = processor(item)
    
    # Check that ASR text was generated
    assert "asr_text" in processed_item
    assert isinstance(processed_item["asr_text"], str)

def test_missing_audio_file_handling(test_config, test_dataset):
    """Test handling of missing audio files."""
    # Create a processor
    processor = DatasetItemProcessor(
        config=test_config,
        text_tokenizer_hf=AutoTokenizer.from_pretrained(test_config.text_encoder_model_name),
        audio_feature_extractor_hf=AutoFeatureExtractor.from_pretrained(test_config.audio_encoder_model_name)
    )
    
    # Create an item with a non-existent audio file
    item = dict(test_dataset[0])
    item["relative_audio_path"] = "nonexistent/file.wav"
    
    # Process the item and check that it handles the missing file gracefully
    processed_item = processor(item)
    
    # The processor should still return the text features even if audio is missing
    assert "input_ids" in processed_item
    assert "attention_mask" in processed_item
    
    # Audio features should be None or have a special value
    if test_config.audio_input_type == "hf_features":
        assert "audio_features" in processed_item
        assert processed_item["audio_features"] is None or torch.all(processed_item["audio_features"] == 0)
    else:
        assert "input_values" in processed_item
        assert processed_item["input_values"] is None or torch.all(processed_item["input_values"] == 0)

def test_batch_processing(test_config, test_dataset):
    """Test processing a batch of items."""
    # Create a processor
    processor = DatasetItemProcessor(
        config=test_config,
        text_tokenizer_hf=AutoTokenizer.from_pretrained(test_config.text_encoder_model_name),
        audio_feature_extractor_hf=AutoFeatureExtractor.from_pretrained(test_config.audio_encoder_model_name)
    )
    
    # Process multiple items
    batch = [dict(item) for item in test_dataset]
    processed_batch = [processor(item) for item in batch]
    
    # Check that all items were processed
    assert len(processed_batch) == len(batch)
    
    # Check that all items have the expected keys
    for item in processed_batch:
        assert "input_ids" in item
        assert "attention_mask" in item
        if test_config.audio_input_type == "hf_features":
            assert "audio_features" in item
        else:
            assert "input_values" in item

# Add more test cases for different scenarios
# 1. Test with different audio input types
# 2. Test dataset statistics and distributions
# 3. Test edge cases (very short/long audio, empty text, etc.)
