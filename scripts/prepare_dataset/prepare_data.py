#!/usr/bin/env python3
"""
Data preparation script for MELD dataset.

Simplified data preparation for emotion classification models.
"""

import os
import argparse
from pathlib import Path
from config import Config
from utils.data_processor import download_meld_dataset, convert_mp4_to_wav, prepare_hf_dataset, get_class_distribution


def main():
    """Main function for preparing the MELD dataset."""
    parser = argparse.ArgumentParser(description="Prepare MELD dataset for emotion classification")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--data_root", type=str, help="Root directory for data")
    parser.add_argument("--force_wav_conversion", action="store_true", help="Force overwrite of existing WAV files")
    parser.add_argument("--force_reprocess_hf_dataset", action="store_true", help="Force reprocessing for Hugging Face dataset creation")
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        # Load from YAML file
        config = Config.from_yaml(args.config)
    else:
        # Create with default values
        config = Config()
    
    # Override with command-line arguments
    if args.data_root:
        config.data_root = Path(args.data_root)
    if args.force_wav_conversion:
        config.force_wav_conversion = args.force_wav_conversion
    if args.force_reprocess_hf_dataset:
        config.force_reprocess_hf_dataset = args.force_reprocess_hf_dataset
    
    # Create necessary directories
    config.create_directories()
    
    # Step 1: Download MELD dataset if not already downloaded
    print("\n===== Step 1: Downloading MELD Dataset =====")
    download_meld_dataset(config.raw_data_dir)
    
    # Step 2: Convert MP4 files to WAV format
    print("\n===== Step 2: Converting MP4 to WAV =====")
    convert_mp4_to_wav(config, force=config.force_wav_conversion)
    
    # Step 3: Prepare Hugging Face dataset
    print("\n===== Step 3: Preparing Hugging Face Dataset =====")
    prepare_hf_dataset(config, force=config.force_reprocess_hf_dataset)
    
    # Step 4: Show class distribution
    print("\n===== Step 4: Class Distribution =====")
    class_distribution = get_class_distribution(config.hf_dataset_dir)
    for split, distribution in class_distribution.items():
        print(f"\n{split.capitalize()} Split:")
        total = sum(distribution.values())
        for emotion, count in distribution.items():
            percentage = (count / total) * 100
            print(f"  {emotion}: {count} ({percentage:.2f}%)")
    
    print("\n===== Data Preparation Complete =====")
    print(f"Raw data: {config.raw_data_dir}")
    print(f"Processed audio: {config.processed_audio_dir}")
    print(f"HuggingFace datasets: {config.hf_dataset_dir}")


if __name__ == "__main__":
    main()
