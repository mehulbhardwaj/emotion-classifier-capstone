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

from configs.base_config import BaseConfig

# EDA_OUTPUTS_DIR will be defined within perform_meld_eda using the config

def perform_meld_eda(config: BaseConfig, markdown_report_path: Path):
    """
    Perform exploratory data analysis on the processed MELD dataset (Hugging Face Datasets).
    Creates various plots and statistics about the dataset.
    Outputs plots to EDA_OUTPUTS_DIR and generates a markdown report.
    """
    md_lines = []
    md_lines.append(f"# EDA Report for Processed {config.dataset_name.upper()} Dataset")
    md_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("\n---\n")

    print(f"Performing exploratory data analysis on {config.dataset_name.upper()} processed Hugging Face datasets...")
    md_lines.append("## Overview")
    md_lines.append(f"This report provides an exploratory data analysis of the processed {config.dataset_name.upper()} Hugging Face datasets.")
    
    EDA_OUTPUTS_DIR = config.results_dir / "eda" / "processed_features_analysis" / config.dataset_name
    Path(EDA_OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"EDA plot outputs will be saved to: {EDA_OUTPUTS_DIR}")
    # Ensure paths in markdown are relative to the markdown file in the root.
    try:
        eda_output_dir_relative = EDA_OUTPUTS_DIR.relative_to(PROJECT_ROOT)
    except ValueError: # Handle cases where PROJECT_ROOT might not be an ancestor (e.g. symlinks, unusual setup)
        eda_output_dir_relative = EDA_OUTPUTS_DIR 
    md_lines.append(f"Plot images are saved in `{eda_output_dir_relative}`.")
    md_lines.append("\n")

    splits_data = {}
    base_processed_dir = config.processed_hf_dataset_dir

    md_lines.append("### Dataset Loading Status")
    for split in ['train', 'dev', 'test']:
        split_path = base_processed_dir / split
        if split_path.exists() and split_path.is_dir():
            try:
                ds = load_from_disk(str(split_path))
                splits_data[split] = ds
                print(f"Loaded {split} split with {len(ds)} samples. Columns: {ds.column_names}")
                md_lines.append(f"*   Loaded **{split}** split with **{len(ds)}** samples.")
                md_lines.append(f"    *   Columns: `{ds.column_names}`")
            except Exception as e:
                error_msg = f"Error loading {split} split from {split_path}: {e}"
                print(error_msg)
                md_lines.append(f"*   <span style='color:red;'>{error_msg}</span>")
        else:
            not_found_msg = f"Processed dataset for split '{split}' not found at {split_path}."
            print(not_found_msg)
            md_lines.append(f"*   <span style='color:red;'>{not_found_msg}</span>")
    
    if not splits_data:
        print("No dataset splits could be loaded. EDA cannot proceed.")
        md_lines.append("\n**No dataset splits could be loaded. EDA cannot proceed.**")
        with open(markdown_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(md_lines))
        print(f"Markdown report (aborted) saved to {markdown_report_path}")
        return

    # Use config properties for ID_TO_EMOTION and MELD_EMOTIONS
    ID_TO_EMOTION_FROM_CFG = {i: name for i, name in enumerate(config.class_names)}
    MELD_EMOTIONS_FROM_CFG = config.class_names

    # --- 1. Emotion Distribution ---
    md_lines.append("\n## 1. Emotion Distribution")
    print("\n1. Analyzing Emotion Distribution...")
    num_valid_splits = len(splits_data)
    
    if num_valid_splits > 0:
        fig_emotion_dist, axes = plt.subplots(1, num_valid_splits + 1, figsize=(5 * (num_valid_splits +1) , 5), sharey=True)
        if num_valid_splits == 1: # Handle single split case for axes
            axes = [axes, axes] 
            
        all_emotions_list = []
        emotion_stats_md = ["| Split   | Emotion    | Count | Percentage |", "|---------|------------|-------|------------|"]

        for i, (split_name, dataset) in enumerate(splits_data.items()):
            label_column_name = 'label' if 'label' in dataset.column_names else 'labels' if 'labels' in dataset.column_names else None
            if not label_column_name:
                msg = f"Neither 'label' nor 'labels' column found in {split_name} split. Skipping emotion distribution for this split."
                print(msg)
                md_lines.append(f"\n- <span style='color:orange;'>{msg}</span>")
                continue
            
            emotion_label_ids = dataset[label_column_name]
            emotion_labels = [ID_TO_EMOTION_FROM_CFG.get(label_id, "Unknown") for label_id in emotion_label_ids]
            all_emotions_list.extend(emotion_labels)
            emotion_counts = Counter(emotion_labels)
            
            emotions_ordered = MELD_EMOTIONS_FROM_CFG
            counts_ordered = [emotion_counts.get(emo, 0) for emo in emotions_ordered]
            total_in_split = sum(counts_ordered)

            for emo_idx, emo_name in enumerate(emotions_ordered):
                count = counts_ordered[emo_idx]
                percentage = (count / total_in_split * 100) if total_in_split > 0 else 0
                emotion_stats_md.append(f"| {split_name.capitalize()} | {emo_name} | {count} | {percentage:.2f}% |")

            current_ax = axes[i]
            sns.barplot(x=emotions_ordered, y=counts_ordered, ax=current_ax, palette="viridis")
            current_ax.set_title(f"{split_name.capitalize()} Split")
            current_ax.set_ylabel("Frequency")
            current_ax.tick_params(axis='x', rotation=45)
            for k, bar_patch in enumerate(current_ax.patches): # Renamed bar to bar_patch
                current_ax.text(bar_patch.get_x() + bar_patch.get_width()/2, bar_patch.get_height(), 
                             str(counts_ordered[k]), ha='center', va='bottom')

        if all_emotions_list:
            combined_emotion_counts = Counter(all_emotions_list)
            emotions_ordered = MELD_EMOTIONS_FROM_CFG
            counts_ordered = [combined_emotion_counts.get(emo, 0) for emo in emotions_ordered]
            total_overall = sum(counts_ordered)
            
            for emo_idx, emo_name in enumerate(emotions_ordered):
                count = counts_ordered[emo_idx]
                percentage = (count / total_overall * 100) if total_overall > 0 else 0
                emotion_stats_md.append(f"| **Overall** | {emo_name} | {count} | {percentage:.2f}% |")

            combined_ax = axes[num_valid_splits]
            sns.barplot(x=emotions_ordered, y=counts_ordered, ax=combined_ax, palette="viridis")
            combined_ax.set_title("Overall Distribution")
            combined_ax.set_ylabel("") 
            combined_ax.tick_params(axis='x', rotation=45)
            for k, bar_patch in enumerate(combined_ax.patches): # Renamed bar to bar_patch
                combined_ax.text(bar_patch.get_x() + bar_patch.get_width()/2, bar_patch.get_height(), 
                             str(counts_ordered[k]), ha='center', va='bottom')
        
        plt.suptitle("Emotion Class Distribution")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        emotion_dist_path = EDA_OUTPUTS_DIR / "processed_emotion_distribution.png"
        plt.savefig(emotion_dist_path)
        print(f"Saved emotion distribution plot to {emotion_dist_path}")
        md_lines.append(f"![Emotion Distribution]({emotion_dist_path.relative_to(PROJECT_ROOT)})")
        md_lines.append("\n")
        md_lines.extend(emotion_stats_md)
        plt.close(fig_emotion_dist)
    else:
        md_lines.append("\n- No valid splits found to analyze emotion distribution.")

    # --- 2. Audio Duration ---
    md_lines.append("\n## 2. Audio Duration (from Mel Spectrograms)")
    print("\n2. Analyzing Audio Duration from Mel Spectrograms...")
    fig_audio_dur, ax_audio_dur = plt.subplots(figsize=(10, 6))
    all_durations_viz = [] 
    audio_duration_stats_md = ["| Split   | Min (s) | Max (s) | Mean (s) | Median (s) |", "|:--------|--------:|--------:|---------:|-----------:|"]

    for split_name, dataset in splits_data.items():
        feature_col_name = 'input_features' 
        if feature_col_name not in dataset.column_names:
            msg = f"Column '{feature_col_name}' not found in {split_name}. Skipping audio duration."
            print(msg)
            md_lines.append(f"\n- <span style='color:orange;'>{msg}</span>")
            continue
            
        durations = []
        for mel_spec_features in dataset[feature_col_name]:
            if isinstance(mel_spec_features, list): 
                mel_spec_features = np.array(mel_spec_features)

            if isinstance(mel_spec_features, (np.ndarray, torch.Tensor)):
                # MelSpectrogram transform from torchaudio produces (..., n_mels, time)
                num_frames = mel_spec_features.shape[-1] # Time is the last dimension
                duration_sec = num_frames * (config.hop_length / config.sample_rate)
                durations.append(duration_sec)
            else:
                print(f"Warning: {feature_col_name} in {split_name} is not an array/tensor, but {type(mel_spec_features)}. Skipping item.")
        
        if durations:
            mean_dur, min_dur, max_dur, median_dur = np.mean(durations), np.min(durations), np.max(durations), np.median(durations)
            sns.histplot(durations, ax=ax_audio_dur, label=f"{split_name.capitalize()} (mean: {mean_dur:.2f}s)", kde=True, element="step")
            all_durations_viz.extend(durations)
            print(f"{split_name.capitalize()} - Audio duration (s): Min={min_dur:.2f}, Max={max_dur:.2f}, Mean={mean_dur:.2f}, Median={median_dur:.2f}")
            audio_duration_stats_md.append(f"| {split_name.capitalize()} | {min_dur:.2f} | {max_dur:.2f} | {mean_dur:.2f} | {median_dur:.2f} |")
        else:
            print(f"No valid mel spectrograms for duration analysis in {split_name} split.")
            audio_duration_stats_md.append(f"| {split_name.capitalize()} | N/A | N/A | N/A | N/A |")

    if all_durations_viz:
        overall_min, overall_max, overall_mean, overall_median = np.min(all_durations_viz), np.max(all_durations_viz), np.mean(all_durations_viz), np.median(all_durations_viz)
        audio_duration_stats_md.append(f"| **Overall** | {overall_min:.2f} | {overall_max:.2f} | {overall_mean:.2f} | {overall_median:.2f} |")

    ax_audio_dur.set_title("Audio Clip Duration Distribution (from Mel Spectrograms)")
    ax_audio_dur.set_xlabel("Duration (seconds)")
    ax_audio_dur.set_ylabel("Frequency")
    if all_durations_viz: ax_audio_dur.legend()
    plt.tight_layout()
    audio_dur_path = EDA_OUTPUTS_DIR / "processed_audio_duration_distribution.png"
    plt.savefig(audio_dur_path)
    print(f"Saved audio duration plot to {audio_dur_path}")
    md_lines.append(f"![Audio Duration Distribution]({audio_dur_path.relative_to(PROJECT_ROOT)})")
    md_lines.append("\n")
    md_lines.extend(audio_duration_stats_md)
    plt.close(fig_audio_dur)

    # --- 3. Text Length ---
    md_lines.append("\n## 3. Text Length (Number of Tokens)")
    print("\n3. Analyzing Text Length (Number of Tokens from input_ids)...")
    fig_text_len, ax_text_len = plt.subplots(figsize=(10, 6))
    all_token_lengths_viz = []
    text_length_stats_md = ["| Split   | Min Tokens | Max Tokens | Mean Tokens | Median Tokens |", "|---------|------------|------------|-------------|---------------|"]

    for split_name, dataset in splits_data.items():
        if 'input_ids' not in dataset.column_names:
            msg = f"'input_ids' column not found in {split_name} split. Skipping text length analysis."
            print(msg)
            md_lines.append(f"\n- <span style='color:orange;'>{msg}</span>")
            continue

        token_lengths = []
        if 'attention_mask' in dataset.column_names:
            for mask in dataset['attention_mask']:
                 if isinstance(mask, (list, np.ndarray, torch.Tensor)): token_lengths.append(sum(mask))
                 else: print(f"Warning: attention_mask item in {split_name} is not list/array/tensor. Type: {type(mask)}. Skipping.")
        elif 'input_ids' in dataset.column_names: 
             for ids in dataset['input_ids']:
                if isinstance(ids, (list, np.ndarray, torch.Tensor)): token_lengths.append(len(ids))
                else: print(f"Warning: input_ids item in {split_name} is not list/array/tensor. Type: {type(ids)}. Skipping.")
        
        if token_lengths:
            mean_len, min_len, max_len, median_len = np.mean(token_lengths), np.min(token_lengths), np.max(token_lengths), np.median(token_lengths)
            sns.histplot(token_lengths, ax=ax_text_len, label=f"{split_name.capitalize()} (mean: {mean_len:.1f} tokens)", kde=True, element="step")
            all_token_lengths_viz.extend(token_lengths)
            print(f"{split_name.capitalize()} - Token count: Min={min_len}, Max={max_len}, Mean={mean_len:.1f}, Median={median_len:.1f}")
            text_length_stats_md.append(f"| {split_name.capitalize()} | {min_len} | {max_len} | {mean_len:.1f} | {median_len:.1f} |")
        else:
            print(f"No valid input_ids/attention_mask for text length analysis in {split_name} split.")
            text_length_stats_md.append(f"| {split_name.capitalize()} | N/A | N/A | N/A | N/A |")

    if all_token_lengths_viz:
        overall_min_len, overall_max_len, overall_mean_len, overall_median_len = np.min(all_token_lengths_viz), np.max(all_token_lengths_viz), np.mean(all_token_lengths_viz), np.median(all_token_lengths_viz)
        text_length_stats_md.append(f"| **Overall** | {overall_min_len} | {overall_max_len} | {overall_mean_len:.1f} | {overall_median_len:.1f} |")

    ax_text_len.set_title("Text Length Distribution (Number of Tokens)")
    ax_text_len.set_xlabel("Number of Tokens")
    ax_text_len.set_ylabel("Frequency")
    if all_token_lengths_viz: ax_text_len.legend()
    plt.tight_layout()
    text_len_path = EDA_OUTPUTS_DIR / "processed_text_token_length_distribution.png"
    plt.savefig(text_len_path)
    print(f"Saved text length plot to {text_len_path}")
    md_lines.append(f"![Text Token Length Distribution]({text_len_path.relative_to(PROJECT_ROOT)})")
    md_lines.append("\n")
    md_lines.extend(text_length_stats_md)
    plt.close(fig_text_len)

    # --- 4. Speaker Emotion Patterns ---
    md_lines.append("\n## 4. Speaker Emotion Patterns (Training Split)")
    print("\n4. Analyzing Speaker Emotion Patterns (from Training data if available)...")
    train_dataset = splits_data.get('train')
    speaker_col_for_analysis = 'speaker' if train_dataset and 'speaker' in train_dataset.column_names else 'raw_speaker' if train_dataset and 'raw_speaker' in train_dataset.column_names else None
    label_col_for_speaker_analysis = 'label' if train_dataset and 'label' in train_dataset.column_names else 'labels' if train_dataset and 'labels' in train_dataset.column_names else None

    if train_dataset and speaker_col_for_analysis and label_col_for_speaker_analysis:
        df_train = train_dataset.to_pandas()
        df_train['emotion_name'] = df_train[label_col_for_speaker_analysis].apply(lambda x: ID_TO_EMOTION_FROM_CFG.get(x, "Unknown"))
        
        # Group by speaker and then by emotion_name to get counts
        speaker_emotion_grouped = df_train.groupby([speaker_col_for_analysis, 'emotion_name']).size().unstack(fill_value=0)
        
        # Ensure all MELD emotions are columns, even if some speakers don't have them
        for emo in MELD_EMOTIONS_FROM_CFG:
            if emo not in speaker_emotion_grouped.columns:
                speaker_emotion_grouped[emo] = 0
        speaker_emotion_pivot = speaker_emotion_grouped[MELD_EMOTIONS_FROM_CFG] # Order columns

        speaker_utterance_counts = df_train[speaker_col_for_analysis].value_counts()

        if not speaker_emotion_pivot.empty:
            top_n_speakers = 15 
            top_speakers_names = speaker_utterance_counts.nlargest(top_n_speakers).index
            
            speaker_emotion_pivot_top_n = speaker_emotion_pivot.loc[speaker_emotion_pivot.index.isin(top_speakers_names)]

            if not speaker_emotion_pivot_top_n.empty:
                fig_speaker_emo, ax_speaker_emo = plt.subplots(figsize=(15, 8))
                speaker_emotion_pivot_top_n.plot(kind='bar', stacked=True, ax=ax_speaker_emo, colormap='viridis')
                ax_speaker_emo.set_title(f"Emotion Distribution for Top {len(speaker_emotion_pivot_top_n)} Speakers (Train Split)")
                ax_speaker_emo.set_xlabel("Speaker")
                ax_speaker_emo.set_ylabel("Number of Utterances")
                ax_speaker_emo.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
                speaker_emo_path = EDA_OUTPUTS_DIR / "speaker_emotion_patterns.png"
                plt.savefig(speaker_emo_path)
                print(f"Saved speaker emotion patterns plot to {speaker_emo_path}")
                md_lines.append(f"![Speaker Emotion Patterns]({speaker_emo_path.relative_to(PROJECT_ROOT)})")
                
                md_lines.append("\n### Top Speaker Emotion Counts (Train Split)")
                md_lines.append("| Speaker | Total Utterances | Most Frequent Emotion | Count | Less Frequent Emotions (sample) |")
                md_lines.append("|---------|-----------------:|:----------------------|------:|:--------------------------------|")
                for speaker_name in top_speakers_names[:5]: # Show top 5
                    total_utts = speaker_utterance_counts.get(speaker_name, 0)
                    if speaker_name in speaker_emotion_pivot.index:
                        speaker_data = speaker_emotion_pivot.loc[speaker_name]
                        most_frequent_emotion = speaker_data.idxmax()
                        most_frequent_count = speaker_data.max()
                        other_emotions = ", ".join([f"{emo}({count})" for emo, count in speaker_data[speaker_data > 0].sort_values(ascending=False).items() if emo != most_frequent_emotion][:2])
                        md_lines.append(f"| {speaker_name} | {total_utts} | {most_frequent_emotion} | {most_frequent_count} | {other_emotions} |")
                    else:
                        md_lines.append(f"| {speaker_name} | {total_utts} | N/A | N/A | N/A |")
            else:
                msg = "Not enough data for top speakers to plot speaker emotion patterns."
                print(msg)
                md_lines.append(f"\n- <span style='color:orange;'>{msg}</span>")
        else:
            msg = "Could not generate speaker emotion pivot table (possibly no speaker data)."
            print(msg)
            md_lines.append(f"\n- <span style='color:orange;'>{msg}</span>")
    else:
        msg = "Training data or required 'speaker'/'label' columns not available for speaker emotion pattern analysis."
        print(msg)
        md_lines.append(f"\n- <span style='color:orange;'>{msg}</span>")

    # --- 5. Dataset Summary ---
    md_lines.append("\n## 5. Dataset Summary")
    print("\n5. Dataset Summary...")
    md_lines.append("| Split   | Number of Samples | Features (Columns) |")
    md_lines.append("|---------|-------------------|--------------------|")
    for split_name, dataset in splits_data.items():
        md_lines.append(f"| {split_name.capitalize()} | {len(dataset)} | `{', '.join(dataset.column_names)}` |")

    md_lines.append("\n## 6. Correlation Analysis (Placeholder)")
    md_lines.append("Further analysis could explore correlations, e.g., between text length and audio duration.")
    
    print("\nEDA (Processed Features) complete.")
    md_lines.append("\n---\nReport End.")

    try:
        # Check if file exists and has content to decide on mode and prepending newline/separator
        file_exists_and_not_empty = markdown_report_path.exists() and markdown_report_path.stat().st_size > 0
        mode = 'a' if file_exists_and_not_empty else 'w' # Append if exists and not empty, else write new
        
        with open(markdown_report_path, mode, encoding='utf-8') as f:
            if file_exists_and_not_empty:
                f.write("\n\n---\n\n") # Add a separator if appending
            f.write("\n".join(md_lines))
        print(f"Markdown EDA report {'appended to' if mode == 'a' else 'saved to'} {markdown_report_path}")
    except Exception as e:
        print(f"Error writing markdown report to {markdown_report_path}: {e}")

def main(cfg_param: BaseConfig = None):
    """
    Main function to run the EDA on processed MELD features.
    Can be called with a specific config.
    """
    current_cfg = cfg_param
    if current_cfg is None:
        print("No config passed to EDA main, using default BaseConfig for MELD.")
        # Ensure default params for audio are set if not in a YAML
        # BaseConfig now has these defaults, so direct instantiation is fine.
        current_cfg = BaseConfig(dataset_name="meld") # Removed **default_audio_params
    
    markdown_report_file_path = PROJECT_ROOT / "docs" / "results" / "eda" / "meld_data_eda.md"
    perform_meld_eda(config=current_cfg, markdown_report_path=markdown_report_file_path)

if __name__ == '__main__':
    main()
