# Emotion Classification from Audio + Text (DLFA Capstone Project)

A streamlined implementation for emotion classification using both audio and text features from the MELD dataset. This project focuses on simplicity, maintainability, and performance comparison between three distinct architectures.

## Project Goal

Classify speaker emotions from audio and text inputs using the MELD dataset. The system compares three different model architectures to provide insights into their relative strengths and performance characteristics.

## Key Features

* **Three Model Architectures**:
  - **MLP Fusion**: Simple baseline with concatenated audio/text features
  - **DialogRNN**: Context-aware model with speaker and emotion tracking
  - **TOD-KAT Lite**: Sequence-to-sequence architecture with topic modeling and knowledge awareness

* **Streamlined Workflow**:
  - Clean YAML-only configuration system
  - Unified training and evaluation scripts
  - Comprehensive architecture comparison tools

* **Advanced Features**:
  - Proper sequence-to-sequence emotion prediction for dialogue contexts
  - Causal masking for autoregressive modeling
  - Topic embeddings and knowledge vector integration
  - Significantly improved data efficiency (5x more training samples)

* **Easy Testing**:
  - Single architecture testing for quick validation
  - Full comparison suite for research insights
  - Automated performance ranking
  - Comprehensive test suite with shape validation

## Project Structure

```
emotion-classification-dlfa/
├── configs/                     # YAML configuration files
│   ├── base_config.py          # Central configuration class
│   ├── colab_config_mlp_fusion.yaml
│   ├── colab_config_dialog_rnn.yaml
│   └── colab_config_todkat_lite.yaml
├── models/                      # Model implementations
│   ├── mlp_fusion.py           # Simple MLP fusion baseline
│   ├── dialog_rnn.py           # DialogRNN with context modeling
│   └── todkat_lite.py          # TOD-KAT with seq2seq prediction
├── utils/                       # Utility functions
│   ├── data_processor.py       # MELD data processing
│   └── sampler.py              # Dialogue-level batch sampling
├── tests/                       # Test suite
│   ├── test_todkat_seq2seq.py  # TOD-KAT validation tests
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

### Validation Testing

Test the TOD-KAT implementation with comprehensive validation:

```bash
python tests/test_todkat_seq2seq.py
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
**Sequence-to-sequence topic and knowledge-aware modeling**
- **Proper seq2seq prediction**: Predicts emotions for all utterances y₁, y₂, ..., yₙ
- **Causal masking**: Each position only attends to previous context (autoregressive)
- **Topic embeddings**: 128-dimensional topic vectors for thematic understanding
- **Knowledge integration**: 64-dimensional knowledge vectors for world knowledge
- **5x data efficiency**: Uses all valid utterances instead of just final ones

```
Audio + Text → [CLS] Features     \
Topic ID → Topic Embeddings      → [uₙ; zₙ; cₙ] → Causal Transformer → Emotion
Knowledge → Knowledge Vectors   /                    (masked attention)
```

**Architecture Details:**
```
P(y₁:ₙ|x₁:ₙ) = ∏ P(yₙ|x≤ₙ, y<ₙ)
```
Where each prediction uses only past context through causal masking.

## Recent Improvements

### TOD-KAT Implementation Overhaul
- **Fixed fundamental architecture**: Changed from "final utterance prediction" to proper sequence-to-sequence
- **Proper causal masking**: Implemented autoregressive constraint via attention masks
- **Dimension compatibility**: Fixed transformer head compatibility (d_model=1728, divisible by heads)
- **Knowledge integration**: Added 64-dim knowledge vectors (was 50-dim, causing dimension issues)
- **Data efficiency**: 5x improvement by using all valid utterances instead of just dialogue endings

### Configuration System Enhancements
- **Simplified config**: Clean YAML-only system with no CLI argument overrides  
- **Centralized parameters**: All settings in `configs/base_config.py` with YAML overrides
- **Better error handling**: Proper None-value handling for class weights and optional parameters

### Comprehensive Testing
- **Shape validation**: Ensures correct tensor dimensions throughout the pipeline
- **Causal masking tests**: Verifies autoregressive constraints work properly
- **Training validation**: End-to-end training loop testing with backward pass
- **Data efficiency measurement**: Quantifies improvement in training sample utilization

## Configuration System

The project uses a simple YAML-based configuration system:

- **Central Config Class**: `configs/base_config.py` defines all parameters
- **Architecture Configs**: YAML files override defaults for each model
- **No CLI Overrides**: Clean, predictable configuration from YAML only

Example configuration structure:
```yaml
# configs/colab_config_todkat_lite.yaml
architecture_name: "todkat_lite"
experiment_name: "colab_run_todkat_lite"
batch_size: 8
learning_rate: 1e-4

# TOD-KAT specific settings
topic_embedding_dim: 128
n_topics: 50
rel_transformer_layers: 2
rel_heads: 4
use_knowledge: true

# Model architecture ensures d_model = 768+768+128+64 = 1728
# which is divisible by rel_heads=4 (1728/4=432)
```

## Results and Performance

Expected performance characteristics:

| Architecture | Parameters | Key Strength | Training Time | Data Efficiency |
|-------------|-----------|--------------|---------------|-----------------|
| MLP Fusion | ~427K | Simple baseline | Fast (~15 min) | Standard |
| DialogRNN | ~5.3M | Context awareness | Medium (~30 min) | Standard |
| TOD-KAT Lite | ~24M | Seq2seq + topics | Slow (~60 min) | **5x improved** |

### TOD-KAT Data Efficiency
- **Old approach**: Used only final utterance per dialogue (~1 sample per dialogue)
- **New approach**: Uses all valid utterances in sequence (~5 samples per dialogue)
- **Result**: 5x more training data from the same dialogues

The testing scripts will provide actual F1 scores and performance rankings for your specific dataset and hardware.

## Testing and Debugging

### Comprehensive Test Suite
```bash
# Test TOD-KAT implementation thoroughly
python tests/test_todkat_seq2seq.py

# Quick dimension validation
python debug_dimensions.py
```

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

The data processor handles:
- **Dialogue-level organization** for context models (DialogRNN, TOD-KAT)
- **Speaker information** for speaker-aware modeling
- **Sequence-to-sequence batching** with proper masking
- **Topic and knowledge vector** integration for TOD-KAT

## Contributing

Contributions to improve the models or add new architectures are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests (follow `tests/test_todkat_seq2seq.py` as example)
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

## Acknowledgments

- TOD-KAT implementation follows the methodology from "Target-Oriented Dialogue and Knowledge-Aware Transformer" 
- DialogRNN based on "DialogueRNN: An Attentive RNN for Emotion Detection in Conversations"
- MELD dataset from "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations"