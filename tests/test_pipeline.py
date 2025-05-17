import pytest
from pathlib import Path
from datasets import Dataset
import sys
import os
import subprocess
import shutil

# Add project root to sys.path to allow importing project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_hf_dataset import create_dataset_from_csv
# from configs.base_config import BaseConfig # We will mock this for simplicity in the test

# A simple mock for BaseConfig for testing purposes
class MockConfig:
    def __init__(self, test_data_path: Path):
        self.dataset_name = "meld"  # MELD is used for label_encoder logic in create_dataset_from_csv
        # Point raw_data_dir to where the dummy CSV is (e.g., tests/test_data/)
        self.raw_data_dir = test_data_path
        # Point processed_audio_output_dir to where the dummy WAVs are (e.g., tests/test_data/ where 'dev/diaX_uttY.wav' will be sought)
        self.processed_audio_output_dir = test_data_path
        # Ensure label_encoder is available as it's used in create_dataset_from_csv
        # Copied from BaseConfig for MELD
        self.label_encoder = {"anger": 0, "disgust": 1, "fear": 2, "joy": 3, "neutral": 4, "sadness": 5, "surprise": 6}
        # Add any other attributes required by create_dataset_from_csv if errors arise
        self.sample_rate = 16000 # Default, might be needed by downstream processing if we extend tests
        self.num_mel_bins = 128 # Default
        self.hop_length = 512 # Default


@pytest.fixture
def test_data_path() -> Path:
    return PROJECT_ROOT / "tests" / "test_data"

@pytest.fixture
def mock_config(test_data_path: Path) -> MockConfig:
    return MockConfig(test_data_path)

def test_create_dataset_from_dummy_csv_and_wav(mock_config: MockConfig):
    """
    Tests the create_dataset_from_csv function with dummy data.
    It expects a dummy CSV and corresponding (empty) WAV files in:
    tests/test_data/dummy_dev_sent_emo.csv
    tests/test_data/dev/dia0_utt0.wav
    tests/test_data/dev/dia0_utt1.wav
    """
    split = "dev" # The dummy CSV is named dummy_dev_sent_emo.csv, create_dataset_from_csv will look for 'dev_sent_emo.csv'
    
    # We need to temporarily rename our dummy CSV to what create_dataset_from_csv expects
    dummy_csv_path = mock_config.raw_data_dir / "dummy_dev_sent_emo.csv"
    expected_csv_path = mock_config.raw_data_dir / f"{split}_sent_emo.csv"
    
    assert dummy_csv_path.exists(), "Dummy CSV file for test is missing."
    
    # Ensure no leftover from previous runs
    if expected_csv_path.exists():
        expected_csv_path.unlink()

    try:
        # Rename dummy CSV to the name the function expects
        dummy_csv_path.rename(expected_csv_path)
        
        # Ensure dummy WAVs exist (create_dataset_from_csv checks for their existence)
        # The dummy CSV refers to Dialogue_ID 0, Utterance_ID 0 and 1.
        # So, audio paths will be like: test_data_path/dev/dia0_utt0.wav
        wav_dir = mock_config.processed_audio_output_dir / split
        wav_dir.mkdir(parents=True, exist_ok=True) # Ensure the 'dev' subfolder exists
        
        # The dummy CSV creates dia0_utt0 and dia0_utt1
        dummy_wav_1 = wav_dir / "dia0_utt0.wav"
        dummy_wav_2 = wav_dir / "dia0_utt1.wav"
        
        assert dummy_wav_1.exists(), f"Dummy WAV file {dummy_wav_1} for test is missing."
        assert dummy_wav_2.exists(), f"Dummy WAV file {dummy_wav_2} for test is missing."

        dataset = create_dataset_from_csv(split=split, config=mock_config, limit_dialogues=1) # Limit to 1 dialogue (which has 2 utterances in our dummy)
        
        assert dataset is not None, "create_dataset_from_csv returned None"
        assert isinstance(dataset, Dataset), "Returned object is not a Hugging Face Dataset"
        assert len(dataset) == 2, f"Expected 2 items in the dataset, got {len(dataset)}"
        
        # Check content of the first item (optional, but good for sanity)
        if len(dataset) > 0:
            item = dataset[0]
            assert item['dialogue_id'] == 0
            assert item['utterance_id'] == 0
            assert item['text'] == "Okay?"
            assert item['emotion'] == "neutral"
            assert item['label'] == mock_config.label_encoder["neutral"]
            assert Path(item['audio_path']).name == "dia0_utt0.wav"

    finally:
        # Clean up: rename the CSV back
        if expected_csv_path.exists():
            expected_csv_path.rename(dummy_csv_path)
        # No need to delete dummy WAVs, they are part of test setup


# --- Test for running main.py --- 

@pytest.fixture
def test_run_env_setup(test_data_path: Path):
    """Sets up the directory structure and files for testing main.py run."""
    project_root = Path(__file__).resolve().parent.parent
    test_run_base_dir_name = "test_main_run_data" # Must match dataset_name in test_config.yaml for path derivation
    test_run_data_root = project_root / test_run_base_dir_name

    # Define source dummy files (already created by other fixtures/setup)
    dummy_csv_src = test_data_path / "dummy_dev_sent_emo.csv"
    dummy_wav_dir_src = test_data_path / "dev" # Contains dia0_utt0.wav, dia0_utt1.wav

    # Define target paths within the temporary test_run_data_root structure
    # main.py with --train_model will use 'train' and 'dev' splits by default for MELDDataModule
    target_raw_dir = test_run_data_root / "raw"
    target_processed_audio_dir = test_run_data_root / "processed" / "audio"
    
    target_train_csv = target_raw_dir / "train_sent_emo.csv"
    target_dev_csv = target_raw_dir / "dev_sent_emo.csv"
    # Test with a single CSV for both train and dev to simplify, as it's just for pipeline execution

    target_train_audio_dir = target_processed_audio_dir / "train"
    target_dev_audio_dir = target_processed_audio_dir / "dev"

    # Clean up before test, if exists from a previous failed run
    if test_run_data_root.exists():
        shutil.rmtree(test_run_data_root)

    # Create directory structure
    target_raw_dir.mkdir(parents=True, exist_ok=True)
    target_train_audio_dir.mkdir(parents=True, exist_ok=True)
    target_dev_audio_dir.mkdir(parents=True, exist_ok=True)

    # Copy dummy CSV to train and dev locations expected by main.py (via config)
    assert dummy_csv_src.exists(), f"Source dummy CSV not found: {dummy_csv_src}"
    shutil.copy(dummy_csv_src, target_train_csv)
    shutil.copy(dummy_csv_src, target_dev_csv)
    print(f"Copied dummy CSV to {target_train_csv} and {target_dev_csv}")

    # Copy dummy WAVs to train and dev audio locations
    # Our dummy CSV has dia0_utt0.wav and dia0_utt1.wav
    assert (dummy_wav_dir_src / "dia0_utt0.wav").exists(), "Source dummy WAV dia0_utt0.wav missing"
    assert (dummy_wav_dir_src / "dia0_utt1.wav").exists(), "Source dummy WAV dia0_utt1.wav missing"
    
    shutil.copy(dummy_wav_dir_src / "dia0_utt0.wav", target_train_audio_dir / "dia0_utt0.wav")
    shutil.copy(dummy_wav_dir_src / "dia0_utt1.wav", target_train_audio_dir / "dia0_utt1.wav")
    shutil.copy(dummy_wav_dir_src / "dia0_utt0.wav", target_dev_audio_dir / "dia0_utt0.wav")
    shutil.copy(dummy_wav_dir_src / "dia0_utt1.wav", target_dev_audio_dir / "dia0_utt1.wav")
    print(f"Copied dummy WAVs to {target_train_audio_dir} and {target_dev_audio_dir}")

    yield test_run_data_root # Provide the path for assertions if needed

    # Teardown: remove the temporary directory structure
    print(f"Tearing down test run environment at {test_run_data_root}")
    shutil.rmtree(test_run_data_root)

def test_run_main_script_minimal(test_run_env_setup):
    """
    Tests running main.py with --train_model --num_epochs 1 --limit_dialogues_train 1
    using a test-specific YAML config that points to dummy data.
    """
    project_root = Path(__file__).resolve().parent.parent
    main_script_path = project_root / "main.py"
    test_config_path = project_root / "tests" / "test_config.yaml"

    assert main_script_path.exists(), "main.py script not found"
    assert test_config_path.exists(), "Test YAML config (tests/test_config.yaml) not found"

    # Command to execute
    # The test_config.yaml already specifies num_epochs, limits, and architecture.
    # We also set device to cpu in the yaml to avoid CI issues.
    # The --architecture mlp_fusion is technically redundant if set in yaml but good for clarity.
    command = [
        sys.executable, # Use the same python interpreter that runs pytest
        str(main_script_path),
        "--config_file", str(test_config_path),
        "--train_model",
        # CLI args below can override YAML, but we've set them in YAML for this test
        # "--num_epochs", "1", 
        # "--limit_dialogues_train", "1",
        # "--architecture", "mlp_fusion", # Specified in YAML
        # "--device_name", "cpu" # Specified in YAML
    ]

    print(f"Executing command: {' '.join(command)}")

    # The test_run_env_setup fixture handles creating the necessary dummy data 
    # in PROJECT_ROOT/test_main_run_data/ which the test_config.yaml points to via dataset_name.

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(timeout=300) # 5 min timeout, should be very fast

    print("--- main.py STDOUT ---")
    print(stdout)
    print("--- main.py STDERR ---")
    print(stderr)

    assert process.returncode == 0, f"main.py script failed with exit code {process.returncode}.\nSTDERR:\n{stderr}"

    # Add checks for output files if necessary, e.g., a model file or logs
    # For example, check if the Hugging Face dataset was created in the temp dir:
    test_run_data_root = test_run_env_setup # Get the path from the fixture
    # Path where prepare_meld_hf_dataset (called by _run_data_preparation in main.py)
    # is expected to save the processed HF dataset for the 'train' split.
    # According to build_hf_dataset.py save logic (main.py line 66-70): cfg.processed_hf_dataset_dir / split_name
    # cfg.processed_hf_dataset_dir = cfg.processed_features_dir / "hf_datasets"
    # cfg.processed_features_dir = cfg.dataset_data_root / "processed" / "features"
    # cfg.dataset_data_root = project_root / "test_main_run_data"
    # So, expected_hf_train_path = project_root / "test_main_run_data" / "processed" / "features" / "hf_datasets" / "train"
    
    # Let's check the slightly different path used by MELDDataModule to load it, 
    # as this is what training uses.
    # From MELDDataModule _get_hf_dataset_path:
    # hf_dataset_dir_for_split = self.cfg.processed_features_dir / "hf_datasets" / split / "processed_data" / cache_params_str
    # Our test_config.yaml sets limit_dialogues_train=1, input_mode defaults to audio_text (so use_asr=False)
    cache_params_str = "asr_False_limit_1" # Based on default input_mode (audio_text -> asr=False) and limit_dialogues_train=1
    expected_hf_train_dataset_path = (
        test_run_data_root / "processed" / "features" / "hf_datasets" / "train" / "processed_data" / cache_params_str
    )

    assert expected_hf_train_dataset_path.exists(), \
        f"Processed Hugging Face dataset for train split not found at expected path: {expected_hf_train_dataset_path}\nCheck main.py data preparation save paths and MELDDataModule load paths."
    assert (expected_hf_train_dataset_path / "dataset_info.json").exists(), \
        f"dataset_info.json missing in {expected_hf_train_dataset_path}, dataset not saved correctly."

    print(f"Successfully verified existence of processed HF dataset at: {expected_hf_train_dataset_path}")

    # Check for a model file if training is expected to save one. Config controls this.
    # cfg.model_save_path = cfg.model_save_dir / f"{cfg.architecture_name}_model.pt"
    # cfg.model_save_dir = cfg.get_specific_model_dir(cfg.architecture_name)
    # get_specific_model_dir = self.models_dir / self.dataset_name / architecture_name
    # self.models_dir = PROJECT_ROOT / "models"
    # So, model_path = PROJECT_ROOT / "models" / "test_main_run" / "mlp_fusion" / "mlp_fusion_model.pt"
    expected_model_dir = project_root / "models" / "test_main_run" / "mlp_fusion"
    expected_model_file = expected_model_dir / "mlp_fusion_model.pt"
    assert expected_model_file.exists(), f"Model file not found at {expected_model_file}"
    print(f"Successfully verified existence of saved model at: {expected_model_file}") 