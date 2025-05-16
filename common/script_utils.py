import torch
from pathlib import Path
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import subprocess
import shutil
import filetype

from transformers import Wav2Vec2FeatureExtractor, WavLMModel, AutoProcessor, WhisperModel
from common.utils import ensure_dir
from configs.base_config import BaseConfig

# Get label encoder from BaseConfig
# cfg = BaseConfig() # Commented out: Global config instantiation here can cause issues / hide dependencies
# EMOTION_TO_ID = cfg.label_encoder # Commented out

# Globals for feature extractors - these will be initialized by init_feature_extractors_globals
wavlm_model_global = None
wavlm_feature_extractor_global = None
whisper_model_global = None
whisper_processor_global = None

def convert_mp4_to_wav_meld(
    mp4_base_dir: Path, 
    wav_output_base_dir: Path, 
    meld_csv_df: pd.DataFrame, 
    split_name: str, 
    ffmpeg_path: str = "ffmpeg", 
    force_conversion: bool = False
) -> pd.DataFrame:
    """
    Converts MP4 files from MELD dataset to WAV format (16kHz, mono, 16-bit PCM).
    Updates and returns the DataFrame with an 'audio_path' column pointing to the WAV files.
    (Adapted from old data_loader.py)
    """
    print(f"\nStarting MP4 to WAV conversion for {split_name} split...")
    print(f"  Source MP4s directory: {mp4_base_dir}")
    print(f"  Target WAVs directory: {wav_output_base_dir}")

    if not shutil.which(ffmpeg_path):
        print(f"ERROR: ffmpeg executable not found at '{ffmpeg_path}'. Please install ffmpeg or provide correct path.")
        if 'audio_path' not in meld_csv_df.columns:
            meld_csv_df['audio_path'] = pd.NA 
        return meld_csv_df

    ensure_dir(wav_output_base_dir)

    if 'audio_path' not in meld_csv_df.columns:
        meld_csv_df['audio_path'] = pd.NA

    new_audio_paths = [pd.NA] * len(meld_csv_df)
    total_files_to_process = len(meld_csv_df)
    converted_count = 0
    skipped_existing_wav_count = 0
    missing_mp4_count = 0
    conversion_errors = 0

    with tqdm(total=total_files_to_process, desc=f"Converting MP4s for {split_name}") as pbar:
        for index, row in meld_csv_df.iterrows():
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            
            source_mp4_filename = f"dia{dialogue_id}_utt{utterance_id}.mp4"
            source_mp4_path = mp4_base_dir / source_mp4_filename
            target_wav_filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
            target_wav_path = wav_output_base_dir / target_wav_filename
            
            new_audio_paths[index] = str(target_wav_path)
            pbar.update(1)

            if target_wav_path.exists() and not force_conversion:
                skipped_existing_wav_count += 1
                continue 

            if not source_mp4_path.exists():
                if split_name == "test": # Specific MELD test set naming quirk
                    alt_source_mp4_filename = f"final_videos_testdia{dialogue_id}_utt{utterance_id}.mp4"
                    alt_source_mp4_path = mp4_base_dir / alt_source_mp4_filename
                    if alt_source_mp4_path.exists():
                        source_mp4_path = alt_source_mp4_path
                    else:
                        missing_mp4_count += 1
                        new_audio_paths[index] = pd.NA
                        continue
                else:
                    missing_mp4_count += 1
                    new_audio_paths[index] = pd.NA
                    continue
            
            try:
                kind = filetype.guess(str(source_mp4_path))
                if kind is None or kind.mime != 'video/mp4':
                    print(f"Warning: Source file {source_mp4_path} is not a valid MP4. MIME type: {kind.mime if kind else 'None'}. Skipping.")
                    missing_mp4_count +=1 
                    new_audio_paths[index] = pd.NA
                    continue
            except Exception as e:
                print(f"Warning: Could not determine file type for {source_mp4_path}: {e}. Error: {type(e)}. Skipping.")
                missing_mp4_count += 1
                new_audio_paths[index] = pd.NA
                continue

            try:
                command = [
                    str(ffmpeg_path), "-y", 
                    "-i", str(source_mp4_path),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(target_wav_path)
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    converted_count += 1
                else:
                    print(f"ERROR converting {source_mp4_path}. ffmpeg stderr: {result.stderr}")
                    conversion_errors += 1
                    new_audio_paths[index] = pd.NA
            except Exception as e:
                print(f"Exception during ffmpeg conversion for {source_mp4_path}: {e}")
                conversion_errors += 1
                new_audio_paths[index] = pd.NA
        
    meld_csv_df['audio_path'] = new_audio_paths

    print(f"\nMP4 to WAV conversion summary for {split_name}:")
    print(f"  Total utterances: {total_files_to_process}, Converted: {converted_count}, Skipped existing: {skipped_existing_wav_count}")
    print(f"  MP4s missing/untypeable: {missing_mp4_count}, Conversion errors: {conversion_errors}")
    print(f"  DataFrame now has {meld_csv_df['audio_path'].notna().sum()} valid audio paths for {split_name} split.")
    return meld_csv_df

def load_meld_csvs(csv_base_dir: Path) -> dict[str, pd.DataFrame]:
    """Loads MELD CSVs. (Adapted from old data_loader.py)"""
    datasets_dfs = {}
    print(f"\nLoading MELD CSVs from directory: {csv_base_dir}")
    expected_csv_files = 0
    found_csv_files = 0
    for split in ["train", "dev", "test"]:
        expected_csv_files +=1
        csv_file_name = f"{split}_sent_emo.csv"
        csv_path = csv_base_dir / csv_file_name
        if csv_path.exists():
            found_csv_files +=1
            try:
                df = pd.read_csv(csv_path)
                df = df.rename(columns=lambda x: x.strip())
                required_cols = {'Dialogue_ID', 'Utterance_ID', 'Utterance', 'Emotion'}
                if not required_cols.issubset(df.columns):
                    print(f"Warning: CSV {csv_path} missing required columns. Skipping.")
                    datasets_dfs[split] = pd.DataFrame()
                    continue
                df['split'] = split
                datasets_dfs[split] = df
                print(f"  Loaded MELD {split} data: {len(df)} utterances from {csv_file_name}")
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                datasets_dfs[split] = pd.DataFrame()
        else:
            print(f"Warning: MELD CSV {csv_path} not found.")
            datasets_dfs[split] = pd.DataFrame() 
    if found_csv_files < expected_csv_files:
        print(f"Warning: Only {found_csv_files}/{expected_csv_files} MELD CSV files found.")
    elif found_csv_files == expected_csv_files and all(not df.empty for df in datasets_dfs.values()):
        print("Successfully loaded all expected MELD CSV files.")
    return datasets_dfs

def init_feature_extractors_globals(device="cpu", audio_model_name="microsoft/wavlm-base", text_model_name="openai/whisper-base"):
    """Initializes global models for feature extraction. (Adapted from old data_loader.py)"""
    global wavlm_model_global, wavlm_feature_extractor_global, whisper_model_global, whisper_processor_global
    print(f"Initializing feature extractors on device: {device}")
    try:
        wavlm_feature_extractor_global = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)
        wavlm_model_global = WavLMModel.from_pretrained(audio_model_name).to(device)
        wavlm_model_global.eval()
        print("WavLM model and extractor loaded.")
    except Exception as e:
        print(f"Error loading WavLM model ({audio_model_name}): {e}")
        wavlm_model_global, wavlm_feature_extractor_global = None, None
    try:
        whisper_processor_global = AutoProcessor.from_pretrained(text_model_name)
        whisper_model_global = WhisperModel.from_pretrained(text_model_name).to(device)
        whisper_model_global.eval()
        print("Whisper model and processor loaded.")
    except Exception as e:
        print(f"Error loading Whisper model ({text_model_name}): {e}")
        whisper_model_global, whisper_processor_global = None, None
    if wavlm_model_global is None or whisper_model_global is None:
        print("Critical Error: One or more feature extraction models failed to load.")

def extract_audio_features_wavlm(audio_path: str, device="cpu") -> np.ndarray | None:
    """Extracts WavLM features. (Adapted from old data_loader.py)"""
    if wavlm_model_global is None or wavlm_feature_extractor_global is None:
        return None
    try:
        waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        inputs = wavlm_feature_extractor_global(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None
        with torch.no_grad():
            outputs = wavlm_model_global(input_values, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error extracting WavLM for {Path(audio_path).name}: {e}")
        return None

def extract_text_features_whisper(text: str, device="cpu") -> np.ndarray | None:
    """Extracts Whisper text encoder features. (Adapted from old data_loader.py)"""
    if whisper_model_global is None or whisper_processor_global is None:
        return None
    try:
        processed_inputs = whisper_processor_global(text=text, return_tensors="pt", padding="longest", truncation=True).to(device)
        with torch.no_grad():
            encoder_outputs = whisper_model_global.get_encoder()(processed_inputs.input_ids if hasattr(processed_inputs, "input_ids") else processed_inputs.input_features)
            embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy().squeeze()
    except Exception as e:
        print(f"Error extracting Whisper text features for '{text[:30]}...': {e}")
        return None

def preprocess_and_cache_meld_features(datasets_dfs, feature_dir, config: BaseConfig, force_reprocess=False, device="cpu"):
    """Caches features. (Adapted from old data_loader.py)"""
    for split_name, df in datasets_dfs.items():
        if df.empty:
            continue
        print(f"\nProcessing and caching MELD features for {split_name} set into {feature_dir}...")
        split_specific_feature_dir = Path(feature_dir) / split_name
        ensure_dir(split_specific_feature_dir)
        processed_count, skipped_count, error_count, missing_audio_path_count = 0, 0, 0, 0
        with tqdm(total=len(df), desc=f"Caching features for {split_name}") as pbar:
            for index, row in df.iterrows():
                pbar.update(1)
                dialogue_id = row.get('Dialogue_ID', 'UnknownDia')
                utterance_id_val = row.get('Utterance_ID', f'UnknownUtt{index}')
                utterance_id_str = f"dia{dialogue_id}_utt{utterance_id_val}"
                cached_feature_file = split_specific_feature_dir / f"{utterance_id_str}.npz"
                if cached_feature_file.exists() and not force_reprocess:
                    skipped_count += 1
                    continue
                audio_path_val = row.get('audio_path')
                text = str(row.get('Utterance', ''))
                emotion_label_str = str(row.get('Emotion', ''))
                if pd.isna(audio_path_val) or not Path(audio_path_val).exists():
                    missing_audio_path_count +=1
                    continue
                try:
                    acoustic_feat = extract_audio_features_wavlm(str(audio_path_val), device=device)
                    text_feat = extract_text_features_whisper(text, device=device)
                    if acoustic_feat is not None and text_feat is not None:
                        emotion_id = config.label_encoder.get(emotion_label_str)
                        if emotion_id is None:
                            print(f"Warning: Emotion '{emotion_label_str}' not in label encoder for {utterance_id_str}. Skipping.")
                            error_count +=1
                            continue
                        np.savez_compressed(cached_feature_file, audio_features=acoustic_feat, text_features=text_feat, emotion_label=emotion_id, utterance_id=utterance_id_str)
                        processed_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
        print(f"Finished caching MELD features for {split_name}. Summary: {processed_count} processed, {skipped_count} skipped, {missing_audio_path_count} missing audio, {error_count} errors.") 