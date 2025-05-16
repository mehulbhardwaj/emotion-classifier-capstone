# meld_audiotext_emotion_classifier/eda.py -> scripts/processed_features_eda.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk, concatenate_datasets
from collections import Counter
import torch
from pathlib import Path
import sys

# Ensure common modules can be imported from the project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import BaseConfig # Changed

# Instantiate BaseConfig for MELD dataset (default)
# This EDA script is specific to MELD processed features for now.
# THIS WILL BE REPLACED IF CFG IS PASSED TO MAIN
cfg = BaseConfig(dataset_name="meld") # Default config

# Use config properties
ID_TO_EMOTION = {i: name for i, name in enumerate(cfg.class_names)}
MELD_EMOTIONS = cfg.class_names

# EDA_OUTPUTS_DIR should also come from config or be defined consistently.
EDA_OUTPUTS_DIR = cfg.results_dir / "eda" / "processed_features_analysis" / cfg.dataset_name

# from common.utils import plot_to_numpy # If used for tensorboard, else remove
# This function was not in the provided common/utils.py. Assuming not critical for now.

def perform_meld_eda(tensorboard_writer=None):
    """
    Perform exploratory data analysis on the processed MELD dataset (Hugging Face Datasets).
    Creates various plots and statistics about the dataset.
    Outputs plots to EDA_OUTPUTS_DIR and optionally to TensorBoard.
    """
    print("Performing exploratory data analysis on MELD processed Hugging Face datasets...")
    
    Path(EDA_OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"EDA outputs will be saved to: {EDA_OUTPUTS_DIR}")
    
    splits_data = {}
    all_datasets = []
    expected_columns = None

    # Use cfg.processed_features_dir
    base_processed_dir = cfg.processed_features_dir # Changed

    for split in ['train', 'dev', 'test']:
        # The processed_features_dir already includes the dataset_name in its path
        # so we just need to add the split.
        split_path = base_processed_dir / split 
        if split_path.exists() and split_path.is_dir():
            try:
                ds = load_from_disk(str(split_path))
                splits_data[split] = ds
                all_datasets.append(ds)
                print(f"Loaded {split} split with {len(ds)} samples. Columns: {ds.column_names}")
                if expected_columns is None:
                    expected_columns = set(ds.column_names)
                elif set(ds.column_names) != expected_columns:
                    print(f"Warning: Column mismatch in {split} split. Expected {expected_columns}, got {ds.column_names}")
            except Exception as e:
                print(f"Error loading {split} split from {split_path}: {e}")
        else:
            print(f"Processed dataset for split '{split}' not found at {split_path}. Run data preparation first.")
    
    if not splits_data:
        print("No dataset splits could be loaded. EDA cannot proceed.")
        return

    print("\n1. Analyzing Emotion Distribution...")
    # Adjusted subplot count based on available splits_data
    num_valid_splits = len(splits_data)
    if num_valid_splits == 0: 
        print("No valid splits for emotion distribution.")
        return
        
    fig_emotion_dist, axes = plt.subplots(1, num_valid_splits + 1, figsize=(5 * (num_valid_splits +1) , 5), sharey=True) 
    if num_valid_splits == 1: # if only one split, axes is not an array
        axes = [axes, axes] # Make it iterable for the loop
        
    all_emotions_list = []

    for i, (split_name, dataset) in enumerate(splits_data.items()):
        if 'label' not in dataset.column_names: # Common practice is 'label' or 'labels'
            print(f"'label' column not found in {split_name} split. Trying 'labels'...")
            if 'labels' not in dataset.column_names:
                 print(f"Neither 'label' nor 'labels' column found in {split_name}. Skipping emotion distribution.")
                 continue
            label_column_name = 'labels'
        else:
            label_column_name = 'label'
        
        emotion_label_ids = dataset[label_column_name]
        emotion_labels = [ID_TO_EMOTION.get(label_id, "Unknown") for label_id in emotion_label_ids]
        all_emotions_list.extend(emotion_labels)
        emotion_counts = Counter(emotion_labels)
        
        emotions_ordered = MELD_EMOTIONS # Uses cfg.class_names via this variable
        counts_ordered = [emotion_counts.get(emo, 0) for emo in emotions_ordered]

        current_ax = axes[i] if num_valid_splits > 1 else axes[0] # handle single split case for axes
        sns.barplot(x=emotions_ordered, y=counts_ordered, ax=current_ax, palette="viridis")
        current_ax.set_title(f"{split_name.capitalize()} Split")
        current_ax.set_ylabel("Frequency")
        current_ax.tick_params(axis='x', rotation=45)
        for k, bar in enumerate(current_ax.patches):
            current_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                         str(counts_ordered[k]), ha='center', va='bottom')

    if all_emotions_list:
        combined_emotion_counts = Counter(all_emotions_list)
        emotions_ordered = MELD_EMOTIONS # Uses cfg.class_names
        counts_ordered = [combined_emotion_counts.get(emo, 0) for emo in emotions_ordered]
        
        # Determine the correct axis for the combined plot
        combined_ax = axes[num_valid_splits] if num_valid_splits > 0 else axes[0]
        if num_valid_splits == 0 and len(axes) > 1 : combined_ax = axes[0] # Should not happen given earlier check
        
        sns.barplot(x=emotions_ordered, y=counts_ordered, ax=combined_ax, palette="viridis")
        combined_ax.set_title("Overall Distribution")
        if num_valid_splits > 0 : combined_ax.set_ylabel("") # No Y-label if sharing
        else: combined_ax.set_ylabel("Frequency")
        combined_ax.tick_params(axis='x', rotation=45)
        for k, bar in enumerate(combined_ax.patches):
            combined_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                         str(counts_ordered[k]), ha='center', va='bottom')
    
    plt.suptitle("Emotion Class Distribution")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    emotion_dist_path = EDA_OUTPUTS_DIR / "processed_emotion_distribution.png"
    plt.savefig(emotion_dist_path)
    print(f"Saved emotion distribution plot to {emotion_dist_path}")
    # if tensorboard_writer:
    #     tensorboard_writer.add_image("EDA/Processed_Emotion_Distribution", plot_to_numpy(fig_emotion_dist), 0)
    plt.close(fig_emotion_dist)

    print("\n2. Analyzing Audio Duration from Mel Spectrograms...")
    fig_audio_dur, ax_audio_dur = plt.subplots(figsize=(10, 6))
    all_durations_viz = [] # Renamed to avoid conflict with a potential variable from other cells if in notebook

    for split_name, dataset in splits_data.items():
        if 'mel_spectrogram' not in dataset.column_names: # Check for singular form too
             if 'mel_spectrograms' not in dataset.column_names:
                print(f"Neither 'mel_spectrogram' nor 'mel_spectrograms' column found in {split_name}. Skipping audio duration.")
                continue
             feature_col_name = 'mel_spectrograms'
        else:
            feature_col_name = 'mel_spectrogram'
            
        durations = []
        for mel_spec_features in dataset[feature_col_name]:
            if isinstance(mel_spec_features, np.ndarray) or torch.is_tensor(mel_spec_features):
                 # Assuming [Time, Mels] or [Mels, Time] - need to be careful.
                 # The old script assumed [Time, Mels] for .shape[0]
                 # If it's a list of lists, convert to numpy/tensor first.
                if isinstance(mel_spec_features, list):
                    mel_spec_features = np.array(mel_spec_features)
                num_frames = mel_spec_features.shape[0] if mel_spec_features.shape[0] > mel_spec_features.shape[1] else mel_spec_features.shape[1] # Heuristic for time dim
                duration_sec = num_frames * (cfg.hop_length / cfg.sample_rate) # Use cfg for hop_length and sample_rate
                durations.append(duration_sec)
            else:
                print(f"Warning: mel_spectrogram in {split_name} is not an array/tensor, but {type(mel_spec_features)}. Skipping item.")
        
        if durations:
            sns.histplot(durations, ax=ax_audio_dur, label=f"{split_name.capitalize()} (mean: {np.mean(durations):.2f}s)", kde=True, element="step")
            all_durations_viz.extend(durations)
            print(f"{split_name.capitalize()} - Audio duration (s): Min={np.min(durations):.2f}, Max={np.max(durations):.2f}, Mean={np.mean(durations):.2f}, Median={np.median(durations):.2f}")
        else:
            print(f"No valid mel spectrograms found for duration analysis in {split_name} split.")

    ax_audio_dur.set_title("Audio Clip Duration Distribution (from Mel Spectrograms)")
    ax_audio_dur.set_xlabel("Duration (seconds)")
    ax_audio_dur.set_ylabel("Frequency")
    if all_durations_viz: ax_audio_dur.legend()
    plt.tight_layout()
    audio_dur_path = EDA_OUTPUTS_DIR / "processed_audio_duration_distribution.png"
    plt.savefig(audio_dur_path)
    print(f"Saved audio duration plot to {audio_dur_path}")
    # if tensorboard_writer and all_durations_viz:
    #     tensorboard_writer.add_image("EDA/Processed_Audio_Duration_Distribution", plot_to_numpy(fig_audio_dur), 0)
    #     tensorboard_writer.add_histogram("EDA/Processed_Overall_Audio_Durations", np.array(all_durations_viz), 0)
    plt.close(fig_audio_dur)

    print("\n3. Analyzing Text Length (Number of Tokens from input_ids)...")
    fig_text_len, ax_text_len = plt.subplots(figsize=(10, 6))
    all_token_lengths_viz = [] # Renamed

    for split_name, dataset in splits_data.items():
        if 'input_ids' not in dataset.column_names:
            print(f"'input_ids' column not found in {split_name} split. Skipping text length analysis.")
            continue

        token_lengths = []
        if 'attention_mask' in dataset.column_names:
            for mask in dataset['attention_mask']:
                 if isinstance(mask, (list, np.ndarray, torch.Tensor)):
                    token_lengths.append(sum(mask))
                 else:
                    print(f"Warning: attention_mask item in {split_name} is not list/array/tensor. Type: {type(mask)}. Skipping item.")
        else: # Fallback if no attention mask
            for ids in dataset['input_ids']:
                if isinstance(ids, (list, np.ndarray, torch.Tensor)):
                    token_lengths.append(len(ids))
                else:
                    print(f"Warning: input_ids item in {split_name} is not list/array/tensor. Type: {type(ids)}. Skipping item.")

        if token_lengths:
            sns.histplot(token_lengths, ax=ax_text_len, label=f"{split_name.capitalize()} (mean: {np.mean(token_lengths):.1f} tokens)", kde=True, element="step")
            all_token_lengths_viz.extend(token_lengths)
            print(f"{split_name.capitalize()} - Token count: Min={np.min(token_lengths)}, Max={np.max(token_lengths)}, Mean={np.mean(token_lengths):.1f}, Median={np.median(token_lengths)}")
        else:
            print(f"No valid input_ids found for text length analysis in {split_name} split.")

    ax_text_len.set_title("Text Length Distribution (Number of Tokens)")
    ax_text_len.set_xlabel("Number of Tokens")
    ax_text_len.set_ylabel("Frequency")
    if all_token_lengths_viz: ax_text_len.legend()
    plt.tight_layout()
    text_len_path = EDA_OUTPUTS_DIR / "processed_text_token_length_distribution.png"
    plt.savefig(text_len_path)
    print(f"Saved text length plot to {text_len_path}")
    # if tensorboard_writer and all_token_lengths_viz:
    #     tensorboard_writer.add_image("EDA/Processed_Text_Token_Length_Distribution", plot_to_numpy(fig_text_len), 0)
    #     tensorboard_writer.add_histogram("EDA/Processed_Overall_Text_Token_Lengths", np.array(all_token_lengths_viz), 0)
    plt.close(fig_text_len)

    print("\n4. Analyzing Speaker Emotion Patterns (from Training data if available)...")
    train_dataset = splits_data.get('train')
    # Assuming 'speaker' column exists from create_dataset_from_csv in build_hf_dataset.py
    # The column name in HF dataset is usually 'raw_speaker' if not explicitly mapped otherwise.
    # Let's try to be flexible or rely on 'speaker' if present.
    speaker_col_name = 'speaker' 
    if train_dataset and 'speaker' not in train_dataset.column_names and 'raw_speaker' in train_dataset.column_names:
        speaker_col_name = 'raw_speaker'
        
    label_col_for_speaker = 'label'
    if train_dataset and 'label' not in train_dataset.column_names and 'labels' in train_dataset.column_names:
        label_col_for_speaker = 'labels'

    if train_dataset and speaker_col_name in train_dataset.column_names and label_col_for_speaker in train_dataset.column_names:
        df_train = train_dataset.to_pandas()
        # Ensure labels are mapped to names before counting
        df_train['emotion_name'] = df_train[label_col_for_speaker].apply(lambda x: ID_TO_EMOTION.get(x, "Unknown"))
        speaker_emotion_counts = df_train.groupby(speaker_col_name)['emotion_name'].apply(Counter).unstack(fill_value=0)
        
        if not speaker_emotion_counts.empty:
            top_n_speakers = speaker_emotion_counts.sum(axis=1).nlargest(15).index
            speaker_emotion_counts_top_n = speaker_emotion_counts.loc[top_n_speakers, MELD_EMOTIONS]

            if not speaker_emotion_counts_top_n.empty:
                fig_speaker, ax_speaker = plt.subplots(figsize=(12, 7))
                speaker_emotion_counts_top_n.plot(kind='bar', stacked=True, ax=ax_speaker, colormap='viridis')
                ax_speaker.set_title(f"Emotion Distribution for Top {len(top_n_speakers)} Speakers (Train Split)")
                ax_speaker.set_xlabel("Speaker")
                ax_speaker.set_ylabel("Number of Utterances")
                ax_speaker.tick_params(axis='x', rotation=45)
                ax_speaker.legend(title='Emotion')
                plt.tight_layout()
                speaker_path = EDA_OUTPUTS_DIR / "processed_speaker_emotion_distribution.png"
                plt.savefig(speaker_path)
                print(f"Saved speaker emotion distribution plot to {speaker_path}")
                # if tensorboard_writer:
                #     tensorboard_writer.add_image("EDA/Processed_Speaker_Emotion_Distribution", plot_to_numpy(fig_speaker), 0)
                plt.close(fig_speaker)
            else:
                print("Not enough speaker data for top N to generate speaker emotion plot.")
        else:
            print("Speaker emotion counts are empty.")
    else:
        missing_cols = []
        if not train_dataset: missing_cols.append("train_dataset")
        else:
            if speaker_col_name not in train_dataset.column_names: missing_cols.append(speaker_col_name)
            if label_col_for_speaker not in train_dataset.column_names: missing_cols.append(label_col_for_speaker)
        print(f"Missing data/columns for speaker analysis: {', '.join(missing_cols)}. Skipping speaker analysis.")

    print(f"\nProcessed Features EDA finished. Plots saved to {EDA_OUTPUTS_DIR}")

def main(cfg_param: BaseConfig = None): # Added cfg_param
    global cfg # Declare cfg as global to modify it
    active_cfg = None # To be used if main creates its own cfg, not strictly needed if perform_meld_eda uses global cfg

    if cfg_param:
        cfg = cfg_param # Update global cfg for perform_meld_eda
        # active_cfg = cfg_param # Not strictly needed if perform_meld_eda directly uses the updated global cfg
        print(f"Processed Features EDA using configuration for dataset: {cfg.dataset_name}, input_mode: {cfg.input_mode}")
    else:
        # This branch is for when the script is run directly and no config is passed.
        # The global cfg is already initialized at the top.
        # active_cfg = BaseConfig(dataset_name="meld") # No need to create another local instance if global is used.
        # cfg = active_cfg # Ensure global cfg is this default one if we were to re-assign it here.
        print(f"Processed Features EDA using default (globally initialized) configuration for dataset: {cfg.dataset_name}")

    # perform_meld_eda will use the global cfg, which is now set based on cfg_param or the initial global default.
    perform_meld_eda()
    print("\nEDA completed.")

if __name__ == "__main__":
    # The script currently instantiates its own BaseConfig.
    # If called from main.py, main.py would create and pass the config.
    # main() # Old call
    default_run_cfg = BaseConfig(dataset_name="meld") # Example of creating a config for standalone run
    main(cfg_param=default_run_cfg) 