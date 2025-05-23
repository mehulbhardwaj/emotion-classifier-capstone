# Emotion Classification from Audio + Text (DLFA Capstone Project)

A streamlined implementation for emotion classification using both audio and text features from the MELD dataset. This project focuses on simplicity, maintainability, and performance comparison between three distinct architectures.

## Project Goal

Classify speaker emotions from audio and text inputs using the MELD dataset. The system compares three different model architectures to provide insights into their relative strengths and performance characteristics.

## Key Features

* **Three Model Architectures**:
  - **MLP Fusion**: Simple baseline with concatenated audio/text features
  - **DialogRNN**: Context-aware model with speaker and emotion tracking
  - **TOD-KAT Lite**: Sophisticated architecture with topic modeling and attention

* **Streamlined Workflow**:
  - Clean YAML-only configuration system
  - Unified training and evaluation scripts
  - Comprehensive architecture comparison tools

* **Easy Testing**:
  - Single architecture testing for quick validation
  - Full comparison suite for research insights
  - Automated performance ranking

## Project Structure

```
emotion-classification-dlfa/
├── configs/                     # YAML configuration files
│   ├── colab_config_mlp_fusion.yaml
│   ├── colab_config_dialog_rnn.yaml
│   └── colab_config_todkat_lite.yaml
├── models/                      # Model implementations
│   ├── mlp_fusion.py           # Simple MLP fusion baseline
│   ├── dialog_rnn.py           # DialogRNN with context modeling
│   └── todkat_lite.py          # TOD-KAT with topic attention
├── utils/                       # Utility functions
│   └── data_processor.py       # MELD data processing
├── tests/                       # Test suite
│   └── debug/                  # Debugging tools
├── train.py                    # Main training script
├── test_single_architecture.py # Quick single model testing
├── test_all_architectures.py  # Full comparison suite
├── evaluate.py                 # Model evaluation
└── requirements.txt            # Dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+ (for GPU training)
- Dependencies in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/emotion-classification-dlfa.git
   cd emotion-classification-dlfa
   ```

2. Create and activate a virtual environment:
   ```bash
   conda create -n emotion-classification python=3.10
   conda activate emotion-classification
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Single Architecture Test

Test a single architecture for quick validation:

```bash
# Test DialogRNN
python test_single_architecture.py dialog_rnn

# Test MLP Fusion
python test_single_architecture.py mlp_fusion

# Test TOD-KAT Lite
python test_single_architecture.py todkat_lite
```

### Full Architecture Comparison

Run comprehensive comparison of all three architectures:

```bash
python test_all_architectures.py
```

This will:
- Train all three models sequentially
- Compare their F1 scores and parameter counts
- Generate a performance ranking
- Save detailed results to a timestamped YAML file

### Manual Training

Train a specific model with custom configuration:

```bash
python train.py --config configs/colab_config_dialog_rnn.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --checkpoint path/to/checkpoint.ckpt
```

## Model Architectures

### 1. MLP Fusion (~427K parameters)
**Simple baseline architecture**
- Concatenates frozen Wav2Vec2 audio features + RoBERTa text features
- Single MLP classification head
- Fast training, good baseline performance

```
Audio → Wav2Vec2 (frozen) → Audio Features \
                                           → Concatenate → MLP → Emotion
Text  → RoBERTa (frozen)  → Text Features  /
```

### 2. DialogRNN (~5.3M parameters)
**Context-aware conversation modeling**
- Three parallel GRU networks: global context, speaker tracking, emotion tracking
- Incorporates conversation history and speaker dynamics
- Designed for multi-turn dialogue understanding

```
Audio + Text Features → GRU Global   \
                     → GRU Speaker   → Fusion → Classifier → Emotion
                     → GRU Emotion  /
```

### 3. TOD-KAT Lite (~24M parameters)
**Topic modeling with attention mechanisms**
- Sophisticated topic embedding and relational encoding
- Multi-head attention for complex feature interactions
- Designed for nuanced emotion understanding

```
Audio + Text → Topic Embeddings → Relational Encoder → Attention → Emotion
```

## Configuration System

The project uses a simple YAML-based configuration system:

- **Central Config Class**: `configs/base_config.py` defines all parameters
- **Architecture Configs**: YAML files override defaults for each model
- **No CLI Overrides**: Clean, predictable configuration from YAML only

Example configuration structure:
```yaml
# configs/colab_config_dialog_rnn.yaml
architecture_name: "dialog_rnn"
experiment_name: "colab_run_dialog_rnn"
batch_size: 8
learning_rate: 1e-5
gru_hidden_size: 256
context_window: 10
```

## Results and Performance

Expected performance characteristics:

| Architecture | Parameters | Expected Strength | Training Time |
|-------------|-----------|------------------|---------------|
| MLP Fusion | ~427K | Simple baseline | Fast (~15 min) |
| DialogRNN | ~5.3M | Context awareness | Medium (~30 min) |
| TOD-KAT | ~24M | Complex reasoning | Slow (~60 min) |

The testing scripts will provide actual F1 scores and performance rankings for your specific dataset and hardware.

## Testing and Debugging

### Debug Tools (in tests/debug/)
- `test_fixed_training.py`: Verify architecture selection works correctly
- `verify_models.py`: Check model parameter counts and components
- `debug_models.py`: Detailed model architecture inspection

### Quick Validation
```bash
# Verify DialogRNN loads correctly
python tests/debug/test_fixed_training.py

# Check all model architectures
python tests/debug/verify_models.py
```

## Data Requirements

This project expects MELD dataset processed with:
- Audio files converted to 16kHz mono WAV
- Text transcripts from MELD CSV files
- Processed Hugging Face datasets with audio/text features

The data processor handles dialogue-level organization and speaker information required for DialogRNN.

## Contributing

Contributions to improve the models or add new architectures are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{emotion-classification-dlfa,
  title={Emotion Classification from Audio and Text: A Comparative Study},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/emotion-classification-dlfa}
}
```