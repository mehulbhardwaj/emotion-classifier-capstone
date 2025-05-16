#!/bin/bash

echo "Starting MELD data processing pipeline on SageMaker..."
echo "Pipeline Start Time: $(date)"

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Change this if your conda environment name is different
CONDA_ENV_NAME="emotion-classification-dlfa"
# Change this if your project is not in the current directory where the script is run
# PROJECT_DIR=$(pwd) # Assumes script is in project root, or use absolute path
# cd "$PROJECT_DIR"

echo "Current working directory: $(pwd)"

# --- Activate Conda Environment ---
echo "Attempting to activate Conda environment: $CONDA_ENV_NAME..."
CONDA_BASE_DIR=$(conda info --base)

# Check if CONDA_BASE_DIR is found
if [ -z "$CONDA_BASE_DIR" ]; then
    echo "Error: Conda base directory not found. Make sure Conda is installed and initialized."
    exit 1
fi

# Source Conda activation script
# The exact path might vary based on Conda installation (miniconda, anaconda)
# Common paths:
# $CONDA_BASE_DIR/etc/profile.d/conda.sh
# /opt/conda/etc/profile.d/conda.sh (often in SageMaker/Docker)
CONDA_ACTIVATION_SCRIPT="$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_ACTIVATION_SCRIPT" ]; then
    echo "Conda activation script not found at $CONDA_ACTIVATION_SCRIPT."
    # Try an alternative common path for some environments
    ALT_CONDA_ACTIVATION_SCRIPT="/opt/conda/etc/profile.d/conda.sh"
    if [ -f "$ALT_CONDA_ACTIVATION_SCRIPT" ]; then
        CONDA_ACTIVATION_SCRIPT="$ALT_CONDA_ACTIVATION_SCRIPT"
        echo "Using alternative Conda activation script: $ALT_CONDA_ACTIVATION_SCRIPT"
    else
        echo "Error: Could not find Conda activation script. Please check Conda installation."
        exit 1
    fi
fi

source "$CONDA_ACTIVATION_SCRIPT"
if [ $? -ne 0 ]; then
    echo "Failed to source conda.sh. Exiting."
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment '$CONDA_ENV_NAME'. Exiting."
    exit 1
fi
echo "Conda environment '$CONDA_ENV_NAME' activated successfully."
echo "Python version: $(python --version)"
echo "Python executable: $(which python)"

# --- Step 1: Download and Extract MELD Dataset ---
echo ""
echo "--- Step 1: Downloading and extracting MELD dataset (CSVs and MP4s)... ---"
echo "Start time: $(date)"
python scripts/download_meld_dataset.py
if [ $? -ne 0 ]; then
    echo "Error: Dataset download/extraction (scripts/download_meld_dataset.py) failed. Exiting."
    exit 1
fi
echo "Dataset download and initial extraction complete."
echo "End time: $(date)"

# --- Step 2: Data Preparation (MP4 to WAV, WAV to HF Dataset) ---
# This step relies on the config (e.g., mlp_fusion_default.yaml) having:
# run_mp4_to_wav_conversion: true
echo ""
echo "--- Step 2: Running data preparation (MP4 to WAV, WAV to HF Dataset)... ---"
echo "Using 'mlp_fusion' architecture for config loading by default."
echo "Ensure 'run_mp4_to_wav_conversion: true' is set in 'configs/mlp_fusion_default.yaml' or the relevant config."
echo "Start time: $(date)"
python main.py --architecture mlp_fusion --prepare_data
if [ $? -ne 0 ]; then
    echo "Error: Data preparation (main.py --prepare_data) failed. Exiting."
    exit 1
fi
echo "Data preparation (MP4 to WAV, WAV to HF Dataset) complete."
echo "End time: $(date)"

# --- Step 3: Exploratory Data Analysis (EDA) ---
echo ""
echo "--- Step 3: Running Exploratory Data Analysis (EDA)... ---"
echo "Using 'mlp_fusion' architecture for config loading by default for EDA."
echo "Start time: $(date)"
python main.py --architecture mlp_fusion --run_eda
if [ $? -ne 0 ]; then
    echo "Error: EDA (main.py --run_eda) failed. Exiting."
    exit 1
fi
echo "Exploratory Data Analysis complete. Results should be in the 'results/eda/' directory."
echo "End time: $(date)"

echo ""
echo "--- MELD data processing pipeline finished successfully! ---"
echo "Pipeline End Time: $(date)"

exit 0 