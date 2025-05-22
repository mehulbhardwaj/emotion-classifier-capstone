# Emotion Classification from Audio + Text (DLFA Capstone Project)

A streamlined implementation for emotion classification using both audio and text features from the MELD dataset. This project focuses on simplicity, maintainability, and performance.

## Project Goal

Classify speaker emotions from audio and text inputs using the MELD dataset. The system supports multiple model architectures and is designed to be easily extensible.

## Key Features

* **Multiple Model Architectures**:
  - MLP Fusion
  - Teacher (Transformer-based)
  - Student (Distilled model)
  - PaNNs Fusion

* **Streamlined Workflow**:
  - Simple data preparation pipeline
  - Unified training and evaluation scripts
  - Comprehensive testing

* **Easy Configuration**:
  - YAML-based configuration
  - Environment-specific settings
  - Command-line overrides

## Project Structure

```
emotion-classification-dlfa/
├── configs/             # Configuration files
│   └── mlp_fusion_cli.yaml  # Example model configuration
├── docs/                 # Documentation
│   ├── diagrams/         # System architecture diagrams
│   └── results/          # Training results and reports
├── models/               # Model implementations
│   ├── mlp_fusion.py     # MLP Fusion model
│   ├── panns_fusion.py   # PaNNs Fusion model
│   ├── student.py        # Student model
│   └── teacher.py        # Teacher model
├── scripts/              # Data processing scripts
│   └── prepare_dataset/  # Dataset preparation
├── tests/                # Test files
│   ├── test_data/        # Test data
│   ├── test_data_prep.py
│   ├── test_model_init.py
│   └── test_pipeline.py
├── utils/                # Utility functions
│   ├── data_processor.py
│   └── utils.py
├── .gitignore
├── config.py            # Main configuration
├── evaluate.py          # Model evaluation
├── README.md           # This file
├── requirements.txt     # Python dependencies
└── train.py            # Training script
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU training)
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-classification-dlfa.git
   cd emotion-classification-dlfa
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

1. Download the MELD dataset:
   ```bash
   python scripts/download_meld_dataset.py --output_dir data/meld
   ```

2. Prepare the dataset for training:
   ```bash
   python scripts/prepare_dataset/commands.py --action prepare --data_dir data/meld --output_dir data/processed
   ```

### Training

Train a model with default settings:
```bash
python train.py --config configs/mlp_fusion_cli.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py --checkpoint path/to/checkpoint.ckpt --data_dir data/processed
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Documentation

For detailed documentation, see the [docs](docs/) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Environment Setup

Follow these steps to set up the Conda environment. The `setup_env.sh` script auto-detects CPU/GPU capabilities.

**1. Prerequisites:**
*   Git and `bash`.
*   Conda or Mamba (Mamba is recommended for faster environment solving).
*   (Optional, for GPU) NVIDIA drivers installed and `nvidia-smi` operational.

**2. Clone Repository & Install Miniconda (if needed):**
```bash
# If you don't have Conda/Mamba:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh  # Follow prompts
source ~/.bashrc                       # Or ~/.zshrc if using zsh
conda install -n base mamba -c conda-forge # Install Mamba (optional but recommended)

git clone https://github.com/mehulbhardwaj/emotion-classifier-capstone.git
cd emotion-classifier-capstone
```

**3. Create and Activate Environment:**
```bash
chmod +x setup_env.sh
./setup_env.sh
# The script will instruct you to activate the created environment, e.g.:
# conda activate emotion-classification-gpu  OR  conda activate emotion-classification-cpu
```

**4. FFmpeg (for Audio Conversion):**
FFmpeg is required for converting MP4s to WAV files. It's included in the Conda environment files.
If you encounter issues or need to install it manually:
```bash
# On macOS with Homebrew
brew install ffmpeg
# On Linux with apt
sudo apt update && sudo apt install ffmpeg
# Or ensure it's installable via Conda from conda-forge channel if setup_env.sh fails for ffmpeg
# conda install ffmpeg -c conda-forge
```

## Workflow: From Data to Predictions

This section outlines the typical end-to-end workflow for this project, from downloading the raw MELD dataset to making predictions with a trained model.

**1. Download Raw MELD Dataset:**
Fetch the MELD dataset archives and extract them. The `scripts/download_meld_dataset.py` script has been significantly enhanced to:
    *   Robustly handle varying levels of nesting within the `train.tar.gz`, `dev.tar.gz`, and `test.tar.gz` archives by automatically detecting the correct number of path components to strip. This ensures MP4s are correctly placed into their respective `*_videos/` directories.
    *   Automatically generate `train_sent_emo.csv` by locating and concatenating individual CSV files from a `train_splits/` directory (which is expected to be found within the extracted `MELD.Raw.tar.gz` or locally). This is done if `train_sent_emo.csv` is not already present in the output directory.
    *   Ensure that `*_videos/` directories (e.g., `train_videos/`, `dev_videos/`, `test_videos/`) are cleaned (removed and recreated) before each corresponding tarball extraction. This prevents issues from previous partial or incorrect extractions.

```bash
python scripts/download_meld_dataset.py --data_dir ./data/meld_raw [--force_download_main]
```
*   **Functionality**: This script first downloads `MELD.Raw.tar.gz` (if not present or `--force_download_main` is used) to the specified `--data_dir`.
*   It then extracts the contents of `MELD.Raw.tar.gz` directly into `--data_dir` (e.g., CSV files like `dev_sent_emo.csv`, `test_sent_emo.csv`, and potentially tarballs like `train.tar.gz`, `dev.tar.gz`, `test.tar.gz`, and a `train_splits/` directory).
*   Next, it attempts to build `train_sent_emo.csv` using the logic described above.
*   Finally, it looks for `train.tar.gz`, `dev.tar.gz`, and `test.tar.gz` within `--data_dir` and extracts their MP4 video content into correspondingly named subdirectories: `<data_dir>/train_videos/`, `<data_dir>/dev_videos/`, and `<data_dir>/test_videos/`.
*   **Output Structure**: After successful execution, you should have a structure similar to this within your `--data_dir` (e.g., `./data/meld_raw/`):
    ```
    ./data/meld_raw/
    ├── MELD.Raw.tar.gz       # The main downloaded archive
    ├── train_sent_emo.csv    # Generated or from MELD.Raw.tar.gz
    ├── dev_sent_emo.csv      # From MELD.Raw.tar.gz
    ├── test_sent_emo.csv     # From MELD.Raw.tar.gz
    ├── train_videos/         # Contains ~9990 .mp4 files
    │   ├── dia0_utt0.mp4
    │   └── ...
    ├── dev_videos/           # Contains ~1110 .mp4 files
    │   ├── dia0_utt0.mp4
    │   └── ...
    ├── test_videos/          # Contains ~2747 .mp4 files
    │   ├── dia0_utt0.mp4
    │   └── ...
    ├── train.tar.gz          # (if originally inside MELD.Raw.tar.gz)
    ├── dev.tar.gz            # (if originally inside MELD.Raw.tar.gz)
    ├── test.tar.gz           # (if originally inside MELD.Raw.tar.gz)
    └── ... (other files extracted from MELD.Raw.tar.gz like READMEs, etc.)
    ```
*   Use the optional `--force_download_main` flag to re-download `MELD.Raw.tar.gz` even if it already exists and appears to be of a reasonable size.

**2. Extract WAV files from MP4s:**
Convert the downloaded MP4 utterances to WAV format.
```bash
python main.py --prepare_data --action extract_wavs --config configs/mlp_fusion_default.yaml
```
*   Add `--force_wav_conversion` to re-extract WAVs even if they exist.
*   Output directory: `meld_data/processed/audio/` (configurable via `processed_audio_output_dir` in YAML).

**3. Prepare Hugging Face Datasets:**
Process WAVs and CSVs to create Hugging Face datasets, including feature extraction and optional ASR.
```bash
python main.py --prepare_data --action prepare_hf_dataset --config configs/mlp_fusion_default.yaml
```
*   Output directory: `meld_data/processed/features/hf_datasets/` (configurable via `hf_dataset_output_dir` in YAML).
*   Use arguments like `--limit_dialogues_train`, `--limit_dialogues_dev`, `--limit_dialogues_test` to process a subset of data for faster experimentation (these are passed from `main.py` to the underlying scripts if specified on the command line with `main.py`).

**4. Exploratory Data Analysis (EDA):**
Run EDA scripts on raw and/or processed data.
```bash
python main.py --run_eda --config configs/mlp_fusion_default.yaml
```
*   EDA outputs (plots, reports) are typically saved in `results/eda/`.
*   Refer to `scripts/preliminary_eda.py` and `scripts/processed_features_eda.py` for details.

**5. Model Training:**
Train a model using PyTorch Lightning via `main.py`.
```bash
python main.py fit --config configs/mlp_fusion_default.yaml
# To train a different architecture, change the config file:
# python main.py fit --config configs/teacher_default.yaml
```
*   Model checkpoints, logs, and PyTorch Lightning artifacts are saved under `models/<architecture_name>/<experiment_name>/` (paths configurable in `BaseConfig` and YAMLs).
*   You can override YAML settings from the command line, e.g., `python main.py fit --config ... --trainer.max_epochs=10`.

**6. Model Evaluation:**
Evaluate a trained model on the test set.
```bash
python main.py test --config configs/mlp_fusion_default.yaml --ckpt_path path/to/your/model.ckpt
# If you omit --ckpt_path, Lightning will try to find the best checkpoint from the 'fit' stage if run subsequently.
```
*   Evaluation results are typically printed to console and saved in `results/<architecture_name>/<experiment_name>/`.

**7. Inference:**
Perform inference with a trained model.
(The specific command for `cli/inference.py` through `main.py` might need to be added or clarified. Assuming a structure similar to 'test'):
```bash
# Example placeholder - actual command might vary based on cli/inference.py structure
python main.py predict --config configs/mlp_fusion_default.yaml --ckpt_path path/to/your/model.ckpt --input_audio_path path/to/sample.wav --input_text "Sample transcript"
```

## Model Architectures

The project supports the following architectures. Each has a dedicated directory under `architectures/` containing its model definition, trainer, and default configuration.

**1. MLP Fusion (`mlp_fusion`)**
*   **Concept**: Simple concatenation of embeddings from pre-trained audio (e.g., WavLM) and text (e.g., Whisper, DistilRoBERTa) encoders, followed by an MLP classification head. Encoders are typically frozen.
*   **Characteristics**: Lightweight, fast to train, good baseline.
*   **Implementation**: `architectures/mlp_fusion/`
*   **Diagram**: ![MLP Fusion Architecture](docs/diagrams/mlp_fusion_architecture.png) ([Source](docs/diagrams/mlp_fusion_architecture.mmd))

**2. Teacher TODKAT-lite (`teacher`)**
*   **Concept**: A more powerful model, potentially based on architectures like RoBERTa-Large, incorporating topic modeling, COMET commonsense knowledge, and a sophisticated fusion mechanism (e.g., 2-layer encoder-decoder).
*   **Characteristics**: Aims for high performance. Requires more computational resources.
*   **Implementation**: `architectures/teacher/`
*   **Diagram**: ![Teacher Transformer Architecture](docs/diagrams/teacher_transformer_architecture.png) ([Source](docs/diagrams/teacher_transformer_architecture.mmd))

**3. Student Distilled (`student`)**
*   **Concept**: A compact model (e.g., DistilRoBERTa for text, WavLM-Base for audio) designed for efficiency, potentially distilled from the Teacher model. May include additional components like a GRU party-tracker and multi-task learning (MTL) heads.
*   **Characteristics**: Optimized for latency, targeting strong performance with low inference times.
*   **Implementation**: `architectures/student/`
*   **Diagram**: ![Student Distilled GRU Architecture](docs/diagrams/student_distilled_gru_architecture.png) ([Source](docs/diagrams/student_distilled_gru_architecture.mmd))

**4. PANNs Fusion (`panns_fusion`)**
*   **Concept**: Leverages pre-trained PANNs (Large Pre-trained Audio Neural Networks) for rich audio feature extraction from raw waveforms, combined with text embeddings, and fused via an MLP for classification.
*   **Characteristics**: Strong emphasis on audio modality, potentially capturing nuanced acoustic features.
*   **Implementation**: `architectures/panns_fusion/`
*   **Diagram**: ![PANNs Fusion Architecture](docs/diagrams/panns_fusion_architecture.png) ([Source](docs/diagrams/panns_fusion_architecture.mmd))

## Configuration System

The project uses a hierarchical configuration system:

1.  **`configs/base_config.py`**: Defines `BaseConfig` with common parameters (paths, default hyperparameters, feature settings).
2.  **`<architecture>_default.yaml`** (e.g., `configs/mlp_fusion_default.yaml`): Architecture-specific configurations that override or extend `BaseConfig`.
3.  **Command-Line Arguments**: Can override settings from YAML files when running `main.py` (e.g., `--trainer.max_epochs=5`).

PyTorch Lightning CLI orchestrates the loading of these configurations. The flow is visualized below:
![Configuration Loading Flow](docs/diagrams/configuration_loading_flow.png)

## Additional Tools

*   **`run_meld_pipeline_sagemaker.sh`**: An automated script for end-to-end data processing and training, tailored for AWS SageMaker or similar environments.
*   **`colab_runner.ipynb`**: A Jupyter notebook for running experiments in Google Colab, facilitating easy setup and execution in a cloud environment.

---
*This README provides a guide to understanding, setting up, and using the emotion classification project. For more detailed information on specific components, refer to the source code and documentation within the respective directories.*

# Emotion Classification for MELD Dataset

## Overview

This repository contains a simplified implementation of four emotion classification models for the MELD (Multimodal EmotionLines Dataset). The models use audio and text features to classify emotions into 7 classes: neutral, anger, disgust, fear, joy, sadness, and surprise.

## Features

- **Simplified Architecture**: Clean, minimal code structure focused on the core task
- **Multiple Models**: Four different architectures for emotion classification
  - MLP Fusion: Simple multi-layer perceptron fusion of audio and text features
  - Teacher: Transformer-based architecture with attention mechanism
  - Student: Distilled GRU-based model for faster inference
  - PaNNs Fusion: Model using pre-trained PaNNs audio features
- **Easy Data Preparation**: Automated download and preprocessing of MELD dataset
- **PyTorch Lightning Integration**: Standardized training and evaluation
- **Cross-Platform**: Runs on local machines, GCP servers, and Google Colab

## Directory Structure

```
emotion-classification/
  ├── config.py                  # Simple configuration using dataclasses
  ├── prepare_data.py            # Data preparation script
  ├── train.py                   # Training script
  ├── evaluate.py                # Evaluation script
  ├── models/                    # Model implementations
  │   ├── mlp_fusion.py           # MLP fusion model
  │   ├── teacher.py              # Teacher transformer model
  │   ├── student.py              # Student GRU model
  │   └── panns_fusion.py         # PaNNs fusion model
  ├── utils/                     # Utility functions
  │   └── data_processor.py       # Data processing utilities
  ├── data/                      # Directory for dataset storage
  └── output/                    # Directory for outputs and models
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/emotion-classification.git
cd emotion-classification
```

2. Create a conda environment and install dependencies:

```bash
conda create -n emotion-classification python=3.10
conda activate emotion-classification
pip install -r requirements.txt
```

## Usage

### Data Preparation

Run the data preparation script to download and preprocess the MELD dataset:

```bash
python prepare_data.py
```

Options:
- `--data_root`: Root directory for data storage
- `--force_wav_conversion`: Force overwrite of existing WAV files
- `--force_reprocess_hf_dataset`: Force reprocessing of Hugging Face dataset

### Training

Train one of the emotion classification models:

```bash
python train.py --architecture mlp_fusion --experiment_name my_experiment
```

Options:
- `--architecture`: Model architecture (mlp_fusion, teacher, student, panns_fusion)
- `--experiment_name`: Name for the experiment
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--seed`: Random seed for reproducibility

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --checkpoint output/my_experiment/mlp_fusion/checkpoints/best_model.ckpt
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--batch_size`: Batch size for evaluation

## Models

### MLP Fusion

A simple model that extracts features from audio and text and fuses them using a multi-layer perceptron:

```
Audio Input → Audio Encoder → Audio Features \
                                              → Concatenate → MLP → Emotion Prediction
 Text Input → Text Encoder  → Text Features  /
```

### Teacher Transformer

A more powerful model that uses transformer layers for feature fusion:

```
Audio Input → Audio Encoder → Audio Features → Projection \
                                                          → Transformer → Classifier → Emotion Prediction
 Text Input → Text Encoder  → Text Features  → Projection /
```

### Student GRU

A lighter model that can be distilled from the Teacher model:

```
Audio Input → Audio Encoder → Audio Features → Projection \
                                                        → GRU → Classifier → Emotion Prediction
 Text Input → Text Encoder  → Text Features  → Projection /
```

### PaNNs Fusion

A model that uses pre-extracted PaNNs audio features (Pre-trained Audio Neural Networks):

```
 PaNNs Features → Projection \
                              → Fusion Network → Emotion Prediction
Text Input → Text Encoder → Projection /
```

## Configuration

The `config.py` file provides a simple configuration system using Python dataclasses. You can customize settings by:

1. Modifying the default values in `config.py`
2. Passing command-line arguments to the scripts
3. Creating a YAML config file and loading it with `--config your_config.yaml`

## Results

Visualization of results and model performance metrics are saved in the output directory under `<experiment_name>/<architecture_name>/results/`.

## Learnings from Simplification

This project was significantly simplified from a complex codebase with over 10,000 lines of code to a streamlined implementation with approximately 400 lines of core code. Here are the key insights from this simplification process:

### What Caused the Original Complexity

1. **Overengineered Configuration System**
   - Multiple inheritance layers in the configuration system (BaseConfig → ModelConfig → ArchitectureConfig)
   - Properties and getters/setters causing attribute access conflicts
   - Excessive use of dynamic attribute resolution
   - Configuration parameters spread across multiple files and classes

2. **Excessive Abstraction**
   - Too many abstraction layers between model definitions and usage
   - Complex wrapper classes that added indirection without clear benefits
   - Redundant pattern implementations (e.g., multiple ways to load models)

3. **Anticipatory Programming**
   - Building for hypothetical future requirements that never materialized
   - Implementing complex systems for flexibility rarely used in practice
   - Overcompensating for potential variations in environment and use cases

4. **Framework Over-customization**
   - Custom extensions to PyTorch Lightning's CLI that duplicated functionality
   - Complex integration between custom CLI and Lightning CLI
   - Handcrafted solutions for problems already solved by the framework

### How to Avoid These Pitfalls in Future Projects

1. **Start Simple, Add Complexity Only When Needed**
   - Begin with the minimal viable implementation
   - Add abstractions only when patterns clearly emerge
   - Follow the YAGNI principle (You Aren't Gonna Need It)

2. **Leverage Framework Capabilities**
   - Use built-in framework features before creating custom solutions
   - Understand the design patterns of your primary frameworks (PyTorch Lightning, HuggingFace)
   - Extend frameworks through their recommended extension points

3. **Centralize Configuration**
   - Use simple dataclasses or dictionaries for configuration
   - Keep configuration systems flat when possible
   - Provide sensible defaults to reduce required configuration

4. **Maintain Clear Component Boundaries**
   - Ensure clear separation of concerns between components
   - Minimize dependencies between modules
   - Design interfaces that are simple and intuitive

5. **Consistent Testing During Development**
   - Develop with testing in mind from the beginning
   - Create small test datasets for quick validation
   - Test the complete pipeline early and often
   - Pay special attention to edge cases in data processing

6. **Document Architectural Decisions**
   - Keep a record of why certain design choices were made
   - Document known limitations and edge cases
   - Update documentation when patterns or implementations change

By following these principles, future projects can maintain a codebase that is both maintainable and extensible without unnecessary complexity.

## Contributing

Contributions to improve the models or add new features are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Add your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.