import unittest
import os
import sys
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch

# Ensure the code can find modules in the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base_config import BaseConfig
from configs.config_structures import PathsConfig, TrainerConfig, AudioConfig, TextConfig, _DEFAULT_CSV_COLUMN_MAPS


class TestBaseConfig(unittest.TestCase):
    """Test cases for the BaseConfig class."""

    def setUp(self):
        """Setup common test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create patches for methods that might be missing or need mocking
        self.update_dataset_specifics_patcher = patch.object(BaseConfig, '_update_dataset_specifics', autospec=True)
        self.set_device_patcher = patch.object(BaseConfig, '_set_device', autospec=True)
        self.validate_patcher = patch.object(BaseConfig, 'validate', autospec=True)
        
        # Start the patchers
        self.mock_update_dataset = self.update_dataset_specifics_patcher.start()
        self.mock_set_device = self.set_device_patcher.start()
        self.mock_validate = self.validate_patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop the patchers
        self.update_dataset_specifics_patcher.stop()
        self.set_device_patcher.stop()
        self.validate_patcher.stop()
        self.temp_dir.cleanup()

    def test_init_with_objects(self):
        """Test initialization with pre-instantiated config objects."""
        # Create test config objects
        paths_config = PathsConfig(data_root="test_data")
        trainer_config = TrainerConfig(batch_size=64, learning_rate=0.001)
        audio_config = AudioConfig(audio_model_name="test_audio_model")
        text_config = TextConfig(text_model_name="test_text_model")
        
        # Initialize BaseConfig with objects
        config = BaseConfig(
            dataset_name="test_dataset",
            paths_config_obj=paths_config,
            trainer_config_obj=trainer_config,
            audio_config_obj=audio_config,
            text_config_obj=text_config
        )
        
        # Verify objects were correctly assigned
        self.assertEqual(config.paths, paths_config)
        self.assertEqual(config.trainer_params, trainer_config)
        self.assertEqual(config.audio_config, audio_config)
        self.assertEqual(config.text_config, text_config)
        self.assertEqual(config.dataset_name, "test_dataset")
        
        # Test property delegation
        self.assertEqual(config.data_root, paths_config.abs_data_root)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.audio_model_name, "test_audio_model")
        self.assertEqual(config.text_model_name, "test_text_model")

    def test_init_with_dicts(self):
        """Test initialization with configuration dictionaries."""
        # Create test config dictionaries
        paths_dict = {"data_root": "dict_data"}
        trainer_dict = {"batch_size": 128, "learning_rate": 0.01}
        audio_dict = {"audio_model_name": "dict_audio_model"}
        text_dict = {"text_model_name": "dict_text_model"}
        
        # Initialize BaseConfig with dictionaries
        config = BaseConfig(
            dataset_name="dict_dataset",
            paths_config_dict=paths_dict,
            trainer_config_dict=trainer_dict,
            audio_config_dict=audio_dict,
            text_config_dict=text_dict
        )
        
        # Verify dictionaries were correctly used to create objects
        self.assertEqual(config.paths.data_root, "dict_data")
        self.assertEqual(config.trainer_params.batch_size, 128)
        self.assertEqual(config.trainer_params.learning_rate, 0.01)
        self.assertEqual(config.audio_config.audio_model_name, "dict_audio_model")
        self.assertEqual(config.text_config.text_model_name, "dict_text_model")
        self.assertEqual(config.dataset_name, "dict_dataset")

    def test_init_with_flat_params(self):
        """Test initialization with flat parameters."""
        # Initialize BaseConfig with flat parameters
        config = BaseConfig(
            dataset_name="flat_dataset",
            pc_data_root="flat_data",
            tc_batch_size=256,
            tc_learning_rate=0.002,
            ac_audio_model_name="flat_audio_model",
            txtc_text_model_name="flat_text_model"
        )
        
        # Verify flat parameters were correctly used to create objects
        self.assertEqual(config.paths.data_root, "flat_data")
        self.assertEqual(config.trainer_params.batch_size, 256)
        self.assertEqual(config.trainer_params.learning_rate, 0.002)
        self.assertEqual(config.audio_config.audio_model_name, "flat_audio_model")
        self.assertEqual(config.text_config.text_model_name, "flat_text_model")
        self.assertEqual(config.dataset_name, "flat_dataset")

    def test_precedence_order(self):
        """Test that objects take precedence over dicts which take precedence over flat params."""
        # Create conflicting configurations
        paths_config = PathsConfig(data_root="object_data")
        paths_dict = {"data_root": "dict_data"}
        
        # Test object > dict > flat
        config = BaseConfig(
            pc_data_root="flat_data",
            paths_config_dict=paths_dict,
            paths_config_obj=paths_config
        )
        self.assertEqual(config.paths.data_root, "object_data")
        
        # Test dict > flat
        config = BaseConfig(
            pc_data_root="flat_data",
            paths_config_dict=paths_dict
        )
        self.assertEqual(config.paths.data_root, "dict_data")

    def test_update_from_dict(self):
        """Test updating config from a dictionary."""
        # Create initial config
        config = BaseConfig(
            dataset_name="initial_dataset",
            tc_batch_size=32,
            ac_audio_model_name="initial_audio_model",
            txtc_text_model_name="initial_text_model"
        )
        
        # Update with dictionary
        update_dict = {
            "dataset_name": "updated_dataset",
            "batch_size": 64,  # TrainerConfig field
            "audio_model_name": "updated_audio_model",  # AudioConfig field
            "text_model_name": "updated_text_model"  # TextConfig field
        }
        config.update_from_dict(update_dict)
        
        # Verify updates
        self.assertEqual(config.dataset_name, "updated_dataset")
        self.assertEqual(config.trainer_params.batch_size, 64)
        self.assertEqual(config.audio_config.audio_model_name, "updated_audio_model")
        self.assertEqual(config.text_config.text_model_name, "updated_text_model")

    def test_derived_attributes(self):
        """Test that derived attributes are correctly updated when dependencies change."""
        # Initialize with precision strategy based on mixed_precision_training=True
        config = BaseConfig(tc_mixed_precision_training=True)
        self.assertEqual(config.precision_strategy, "16-mixed")
        
        # Update mixed_precision_training and check that precision_strategy is updated
        update_dict = {"mixed_precision_training": False}
        config.update_from_dict(update_dict)
        self.assertEqual(config.precision_strategy, "32-true")
        
    def test_from_cli_config_basic(self):
        """Test basic functionality of from_cli_config with minimal DictConfig."""
        from omegaconf import OmegaConf
        # Create a simple DictConfig mimicking what LightningCLI might provide
        cli_config = OmegaConf.create({
            "project_settings": {
                "dataset_name": "test_dataset",
                "input_mode": "audio_text",
                "experiment_name": "test_experiment",
                "random_seed": 42,
                "paths": {
                    "data_root": "test_data"
                },
                "trainer_params": {
                    "batch_size": 64,
                    "learning_rate": 0.001
                }
            }
        })
        
        # Mock the architecture system to avoid importing architectures module
        with patch('architectures.get_default_config_class_for_arch', return_value=None):
            # Call the method under test
            config = BaseConfig.from_cli_config(cli_config, "mlp_fusion")
        
        # Verify the config was correctly created
        self.assertEqual(config.dataset_name, "test_dataset")
        self.assertEqual(config.input_mode, "audio_text")
        self.assertEqual(config.experiment_name, "test_experiment")
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.data_root, Path("test_data").absolute())
        self.assertEqual(config.architecture_name, "mlp_fusion")
    
    def test_from_cli_config_empty(self):
        """Test from_cli_config with an empty DictConfig."""
        from omegaconf import OmegaConf
        
        # Create an empty DictConfig
        cli_config = OmegaConf.create({})
        
        # Mock the architecture system to avoid importing architectures module
        with patch('architectures.get_default_config_class_for_arch', return_value=None):
            # Call the method under test with empty config
            config = BaseConfig.from_cli_config(cli_config)
        
        # Verify default values are used
        self.assertEqual(config.dataset_name, "meld")  # Default from BaseConfig.__init__
        self.assertEqual(config.input_mode, "audio_text")  # Default from BaseConfig.__init__
    
    def test_from_cli_config_nested_structure(self):
        """Test from_cli_config with nested structures in project_settings."""
        from omegaconf import OmegaConf
        
        # Create a DictConfig with nested audio and text configs
        cli_config = OmegaConf.create({
            "project_settings": {
                "dataset_name": "iemocap",
                "audio_config": {
                    "sampling_rate": 22050,
                    "n_mels": 128,
                    "audio_model_name": "custom_mel"
                },
                "text_config": {
                    "text_model_name": "bert-base-uncased",
                    "text_max_length": 128
                }
            }
        })
        
        # Mock the architecture system to avoid importing architectures module
        with patch('architectures.get_default_config_class_for_arch', return_value=None):
            # Call the method under test
            config = BaseConfig.from_cli_config(cli_config)
        
        # Verify audio and text configs are correctly processed
        self.assertEqual(config.dataset_name, "iemocap")
        self.assertEqual(config.audio_config.sampling_rate, 22050)
        self.assertEqual(config.audio_config.n_mels, 128)
        self.assertEqual(config.audio_config.audio_model_name, "custom_mel")
        self.assertEqual(config.text_config.text_model_name, "bert-base-uncased")
        self.assertEqual(config.text_config.text_max_length, 128)


if __name__ == "__main__":
    unittest.main()
