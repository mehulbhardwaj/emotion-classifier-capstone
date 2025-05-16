# Content from meld_audiotext_emotion_classifier/scripts/preliminary_eda.py will be pasted here.
# For now, this is a placeholder action. The actual content transfer will be done by reading the old file and writing to new.

import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json
import subprocess
from tqdm import tqdm

# Import BaseConfig
from configs.base_config import BaseConfig

# Instantiate BaseConfig - this script is primarily for MELD's raw data.
# If it needs to be generic, dataset_name could be an arg to its main().
# THIS WILL BE REPLACED IF CFG IS PASSED TO MAIN
cfg = BaseConfig(dataset_name="meld") # Default config

# MELD_RAW_DATA_PARENT_DIR should now point to the parent of all raw datasets
# The specific dataset directory (like MELD.Raw) is handled by cfg.raw_data_dir
# MELD_RAW_DATA_PARENT_DIR = cfg.raw_data_parent_dir # This is data/raw
# meld_root_actual will be cfg.raw_data_dir for "meld"

def analyze_csv(csv_path: Path, file_description: str, output_dir: Path = None):
    """Analyzes a single MELD CSV file, including dialog-level statistics."""
    print(f"\n--- Analyzing: {file_description} ({csv_path.name}) ---")
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)

        print("\n1. Shape of the dataset:")
        print(df.shape)

        print("\n2. Column names:")
        print(df.columns.tolist())

        print("\n3. First 5 rows:")
        print(df.head())

        print("\n4. Emotion distribution:")
        if 'Emotion' in df.columns:
            emotion_counts = df['Emotion'].value_counts()
            print(emotion_counts)
            emotion_percentages = df['Emotion'].value_counts(normalize=True)
            print(f"\n   Proportion of emotions:\n{emotion_percentages}")
        else:
            print("'Emotion' column not found.")

        print("\n5. Utterance statistics:")
        if 'Utterance' in df.columns:
            print(df['Utterance'].describe())
            df['UtteranceLength'] = df['Utterance'].astype(str).apply(len)
            utterance_stats = df['UtteranceLength'].agg(['min', 'max', 'mean', 'median', 'std'])
            print("\n   Utterance length statistics (number of characters):")
            print(utterance_stats)
        else:
            print("'Utterance' column not found.")
            
        print("\n6. Dialog-level statistics:")
        if 'Dialogue_ID' in df.columns:
            dialog_count = df['Dialogue_ID'].nunique()
            print(f"   Total number of dialogs: {dialog_count}")
            
            utterances_per_dialog = df.groupby('Dialogue_ID').size()
            print(f"   Utterances per dialog: min={utterances_per_dialog.min()}, max={utterances_per_dialog.max()}, mean={utterances_per_dialog.mean():.2f}, median={utterances_per_dialog.median()}")
            
            top_dialogs = utterances_per_dialog.sort_values(ascending=False).head(5)
            print(f"\n   Top 5 dialogs by utterance count:")
            for dialog, count in top_dialogs.items():
                print(f"     Dialog {dialog}: {count} utterances")
                
            print("\n   Emotions per dialog:")
            emotions_per_dialog = df.groupby('Dialogue_ID')['Emotion'].apply(lambda x: dict(Counter(x)))
            dominant_emotions = df.groupby('Dialogue_ID')['Emotion'].apply(lambda x: Counter(x).most_common(1)[0][0])
            print(f"   Most common dominant emotion: {dominant_emotions.value_counts().idxmax()}")
            
            if output_dir:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(utterances_per_dialog, kde=True, ax=ax)
                ax.set_title(f"Distribution of Utterances per Dialog - {file_description}")
                ax.set_xlabel("Number of Utterances")
                ax.set_ylabel("Number of Dialogs")
                plot_path = output_dir / f"{csv_path.stem}_utterances_per_dialog.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"   Saved plot to {plot_path}")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                dominant_emotions.value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Distribution of Dominant Emotions per Dialog - {file_description}")
                ax.set_xlabel("Emotion")
                ax.set_ylabel("Number of Dialogs")
                plot_path = output_dir / f"{csv_path.stem}_dominant_emotions.png"
                plt.savefig(plot_path)
                plt.close()
                print(f"   Saved plot to {plot_path}")
        else:
            print("'Dialogue_ID' column not found.")
            
        print("\n7. Speaker statistics:")
        if 'Speaker' in df.columns:
            speaker_count = df['Speaker'].nunique()
            print(f"   Total number of unique speakers: {speaker_count}")
            
            top_speakers = df['Speaker'].value_counts().head(10)
            print("\n   Top 10 speakers by utterance count:")
            print(top_speakers)
            
            speaker_emotions = df.groupby('Speaker')['Emotion'].apply(lambda x: dict(Counter(x)))
            speaker_dominant_emotions = df.groupby('Speaker')['Emotion'].apply(lambda x: Counter(x).most_common(1)[0][0])
            print(f"\n   Most common dominant emotion among speakers: {speaker_dominant_emotions.value_counts().idxmax()}")
        else:
            print("'Speaker' column not found.")

        return {
            "filename": csv_path.name,
            "num_samples": len(df),
            "emotions": emotion_counts.to_dict() if 'Emotion' in df.columns else {},
            "emotion_percentages": emotion_percentages.to_dict() if 'Emotion' in df.columns else {},
            "utterance_stats": utterance_stats.to_dict() if 'Utterance' in df.columns else {},
            "dialog_count": dialog_count if 'Dialogue_ID' in df.columns else 0,
            "utterances_per_dialog_stats": {
                "min": utterances_per_dialog.min(),
                "max": utterances_per_dialog.max(),
                "mean": utterances_per_dialog.mean(),
                "median": utterances_per_dialog.median()
            } if 'Dialogue_ID' in df.columns else {},
            "speaker_count": speaker_count if 'Speaker' in df.columns else 0,
            "df": df
        }
    except Exception as e:
        print(f"Could not process {csv_path}: {e}")
        return None
    finally:
        print("--- End of CSV Analysis ---\n")

def analyze_audio_files(df, video_dir_path: Path, file_description: str, sample_limit=100, output_dir: Path = None):
    """Analyzes audio files (mp4) using ffprobe."""
    print(f"\n--- Analyzing Audio Files for: {file_description} ---")
    # video_dir_path is constructed in main() using cfg.raw_data_dir
    if not video_dir_path.exists() or not video_dir_path.is_dir():
        print(f"ERROR: Directory not found or is not a directory: {video_dir_path}")
        return None

    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("ERROR: ffprobe not found. Please install ffmpeg and ensure it's in your PATH.")
        return None

    if 'Dialogue_ID' not in df.columns or 'Utterance_ID' not in df.columns:
        print("ERROR: Required columns 'Dialogue_ID' or 'Utterance_ID' not found in the dataframe.")
        return None

    df_sampled = df.sample(min(sample_limit, len(df))) if sample_limit > 0 else df
    audio_stats = []
    
    print(f"Analyzing {len(df_sampled)} audio samples...")
    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled)):
        try:
            dia_id = row['Dialogue_ID']
            utt_id = row['Utterance_ID']
            video_file = video_dir_path / f"dia{dia_id}_utt{utt_id}.mp4"
            
            if not video_file.exists():
                if "Development" in file_description and (video_dir_path / "dev_splits_video").exists():
                     video_file = video_dir_path / "dev_splits_video" / f"dia{dia_id}_utt{utt_id}.mp4"
                     if not video_file.exists():
                        continue
                else:
                    continue
                
            file_size_bytes = video_file.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", 
                "format=duration : stream=codec_type,codec_name,sample_rate,channels", 
                "-of", "json", str(video_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            duration = float(info['format']['duration']) if 'format' in info and 'duration' in info['format'] else None
            
            audio_streams = [s for s in info['streams'] if s['codec_type'] == 'audio'] if 'streams' in info else []
            audio_codec = audio_streams[0]['codec_name'] if audio_streams else None
            sample_rate = audio_streams[0].get('sample_rate') if audio_streams else None
            channels = audio_streams[0].get('channels') if audio_streams else None
            
            audio_stats.append({
                "dialogue_id": dia_id,
                "utterance_id": utt_id,
                "file_size_mb": file_size_mb,
                "duration_seconds": duration,
                "audio_codec": audio_codec,
                "sample_rate": sample_rate,
                "channels": channels
            })
        except Exception as e:
            print(f"Error analyzing audio file for dia{dia_id}_utt{utt_id} ({video_file}): {e}")
            continue
    
    if not audio_stats:
        print("No audio files were successfully analyzed.")
        return None
        
    audio_df = pd.DataFrame(audio_stats)
    
    print("\nAudio File Statistics:")
    print(f"Total files analyzed: {len(audio_df)}")
    
    print("\nFile Size (MB):")
    print(audio_df['file_size_mb'].describe())
    
    print("\nDuration (seconds):")
    print(audio_df['duration_seconds'].describe())
    
    print("\nAudio Codec Distribution:")
    print(audio_df['audio_codec'].value_counts())
    
    print("\nSample Rate Distribution:")
    print(audio_df['sample_rate'].value_counts())
    
    print("\nChannels Distribution:")
    print(audio_df['channels'].value_counts())
    
    if output_dir:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(audio_df['file_size_mb'], kde=True, ax=ax)
        ax.set_title(f"File Size Distribution - {file_description}")
        ax.set_xlabel("File Size (MB)")
        ax.set_ylabel("Count")
        plot_path = output_dir / f"{file_description.replace(' ', '_').lower()}_file_size_dist.png"
        plt.savefig(plot_path)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(audio_df['duration_seconds'], kde=True, ax=ax)
        ax.set_title(f"Duration Distribution - {file_description}")
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Count")
        plot_path = output_dir / f"{file_description.replace(' ', '_').lower()}_duration_dist.png"
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Plots saved to {output_dir}")
    
    return {
        "file_count": len(audio_df),
        "file_size_stats": audio_df['file_size_mb'].describe().to_dict(),
        "duration_stats": audio_df['duration_seconds'].describe().to_dict(),
        "audio_codec": audio_df['audio_codec'].value_counts().to_dict(),
        "sample_rate": audio_df['sample_rate'].value_counts().to_dict(),
        "channels": audio_df['channels'].value_counts().to_dict(),
        "df": audio_df
    }

def inspect_video_dirs(video_data_path: Path, dir_description: str, num_files_to_list: int = 5):
    """Lists a few files/subdirectories from a video directory."""
    print(f"\n--- Inspecting Video Directory: {dir_description} ({video_data_path.name}) ---")
    if not video_data_path.exists() or not video_data_path.is_dir():
        print(f"ERROR: Directory not found or is not a directory: {video_data_path}")
        return None

    try:
        print(f"Path: {video_data_path.resolve()}")
        print(f"First {num_files_to_list} items in this directory:")
        items = []
        count = 0
        for item in video_data_path.iterdir():
            if count < num_files_to_list:
                item_type = '[D]' if item.is_dir() else '[F]'
                print(f"  - {item_type} {item.name}")
                items.append({"name": item.name, "is_dir": item.is_dir()})
                count += 1
            else:
                break
        
        if count == 0:
            print("  (Directory is empty or contains no listable items)")
            
        total_files = sum(1 for f in video_data_path.glob('**/*') if f.is_file())
        total_dirs = sum(1 for d in video_data_path.glob('**/*') if d.is_dir())
        print(f"\nTotal files (recursively): {total_files}")
        print(f"Total subdirectories (recursively): {total_dirs}")
        
        return {
            "path": str(video_data_path),
            "sample_items": items,
            "total_files": total_files,
            "total_dirs": total_dirs
        }
    except Exception as e:
        print(f"Could not inspect {video_data_path}: {e}")
        return None
    finally:
        print("--- End of Video Directory Inspection ---\n")

def save_results_to_json(results, output_file: Path):
    """Save analysis results to a JSON file, converting numpy types to native Python types."""
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        return obj

    clean_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            clean_dict = {}
            for k, v in value.items():
                if k != 'df':
                    clean_dict[k] = convert_numpy_types(v)
            clean_results[key] = clean_dict
        else:
            clean_results[key] = convert_numpy_types(value)
            
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=4)
    print(f"Results saved to {output_file}")

def main(cfg_param: BaseConfig = None): # Added cfg_param
    """Main function to run the enhanced EDA script."""
    global cfg # Declare cfg as global to modify it
    if cfg_param:
        cfg = cfg_param # Use passed config if available
        print(f"Preliminary EDA using configuration for dataset: {cfg.dataset_name}, input_mode: {cfg.input_mode}")
    else:
        # This branch will be taken if the script is run directly without passing a config
        print(f"Preliminary EDA using default (globally initialized) configuration for dataset: {cfg.dataset_name}")


    print("Starting Enhanced MELD Dataset Analysis...")
    
    # Create a Path object for the output directory and ensure it exists.
    # output_dir should be passed from main or be based on cfg.results_dir
    # For now, assuming it's passed or handled locally as before, but main() will use cfg.
    base_output_dir = cfg.results_dir / "eda" / "raw_data_analysis" / cfg.dataset_name
    ensure_dir(base_output_dir) # Ensure main EDA output dir exists

    overall_summary = {}

    # Define the root for MELD raw data using cfg
    meld_root_for_script = cfg.raw_data_dir # This will be DATA_DIR / "raw" / "MELD.Raw"
    print(f"Using MELD raw data root: {meld_root_for_script}")

    # Analyze CSVs
    csv_files_to_analyze = {
        "Train Data": meld_root_for_script / "train_sent_emo.csv",
        "Development Data": meld_root_for_script / "dev_sent_emo.csv",
        "Test Data": meld_root_for_script / "test_sent_emo.csv"
    }
    all_dfs = {}
    for desc, path in csv_files_to_analyze.items():
        analysis_result = analyze_csv(path, desc, output_dir=base_output_dir)
        if analysis_result:
            overall_summary[desc + " CSV Analysis"] = {
                k: v for k, v in analysis_result.items() if k != 'df'
            }
            all_dfs[desc] = analysis_result['df']
    
    # Analyze Audio (MP4s)
    # Video directories are typically within meld_root_for_script, e.g., 'train_videos'
    video_dirs_to_analyze = {
        "Train Videos": (meld_root_for_script / "train_videos", all_dfs.get("Train Data")),
        "Dev Videos": (meld_root_for_script / "dev_splits_video", all_dfs.get("Development Data")), # MELD specific name for dev videos
        "Test Videos": (meld_root_for_script / "test_videos", all_dfs.get("Test Data")) 
    }

    for desc, (video_dir, df_for_audio) in video_dirs_to_analyze.items():
        if df_for_audio is not None and not df_for_audio.empty:
            print(f"Analyzing audio for {desc} based on its CSV data...")
            audio_analysis_result = analyze_audio_files(df_for_audio, video_dir, desc, sample_limit=100, output_dir=base_output_dir)
            if audio_analysis_result:
                overall_summary[desc + " Audio Analysis"] = audio_analysis_result
        else:
            print(f"Skipping audio analysis for {desc} as corresponding DataFrame is missing or empty.")

    # Analyze Video Directory Structure
    structure_summary = inspect_video_dirs(meld_root_for_script, "MELD.Raw Video Root", num_files_to_list=2)
    if structure_summary:
        overall_summary["Video Directory Structure"] = structure_summary

    # Save overall summary
    summary_output_file = base_output_dir / "preliminary_eda_summary.json"
    save_results_to_json(overall_summary, summary_output_file)
    print(f"\nOverall EDA summary saved to: {summary_output_file}")
    print("Preliminary EDA finished.")

if __name__ == "__main__":
    # main() # Old call
    # For standalone execution, create a default config or parse args if needed
    # This allows the script to be run directly using its default internal config,
    # or a specific one if main() is called with cfg_param from elsewhere.
    default_run_cfg = BaseConfig(dataset_name="meld") # Example of creating a config for standalone run
    main(cfg_param=default_run_cfg)
    # The path used ("data/raw/") for MELD_RAW_DATA_PARENT_DIR should align with common/config.py's RAW_DATA_DIR.
    # The output_dir for results is now "results/eda/raw_data_analysis" relative to project root.
    # After this change, paths will be derived from BaseConfig instance `cfg`.