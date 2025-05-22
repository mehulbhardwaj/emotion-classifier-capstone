import pandas as pd
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

# Assuming BaseConfig and ensure_dir are accessible, e.g., from utils.utils or configs.base_config
# You'll need to adjust these imports based on your actual project structure.
# from configs.base_config import BaseConfig # Placeholder
# from utils.utils import ensure_dir # Placeholder

# It's good practice to use a logger instead of print for production code.
# import logging
# logger = logging.getLogger(__name__)

def load_metadata(split: str, config, limit_dialogues: int = None) -> pd.DataFrame:
    """
    Load metadata from CSV for a given split.
    """
    csv_path = config.raw_data_dir / f"{split}_sent_emo.csv"
    if not csv_path.exists():
        # logger.error(f"CSV file not found for {split} split: {csv_path}")
        print(f"ERROR: CSV file not found for {split} split: {csv_path}")
        # Return an empty DataFrame or raise an error, depending on desired handling
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        # logger.error(f"Error reading CSV file {csv_path}: {e}")
        print(f"ERROR: Error reading CSV file {csv_path}: {e}")
        return pd.DataFrame()

    if limit_dialogues is not None and 'Dialogue_ID' in df.columns:
        unique_dialogues = df['Dialogue_ID'].unique()[:limit_dialogues]
        df = df[df['Dialogue_ID'].isin(unique_dialogues)]
    elif limit_dialogues is not None:
        # logger.warning("limit_dialogues specified but 'Dialogue_ID' column not found in CSV.")
        print("Warning: limit_dialogues specified but 'Dialogue_ID' column not found in CSV.")
        
    return df

def create_dataset_from_csv(split: str, config, limit_dialogues: int = None) -> Dataset:
    """
    Create a Dataset object from MELD CSV and (pre-existing) audio files.
    Checks for file existence of WAV files in cfg.processed_audio_output_dir.
    
    Args:
        split (str): Dataset split ('train', 'dev', or 'test').
        config (BaseConfig): Configuration object.
        limit_dialogues (int, optional): If set, limits processing to this many dialogues.

    Returns:
        Dataset object or None if critical errors occur.
    """
    df = load_metadata(split, config, limit_dialogues)
    if df.empty:
        print(f"Metadata for {split} is empty. Cannot create dataset.")
        return Dataset.from_list([]) # Return empty dataset

    data = []
    missing_audio_count = 0
    total_rows_for_split = len(df)

    # Ensure the processed_audio_output_dir exists, though files within it are checked individually
    # ensure_dir(config.processed_audio_output_dir / split) # Moved from original, ensure_dir might be in utils

    for idx, row in tqdm(df.iterrows(), total=total_rows_for_split, desc=f"Creating dataset items from CSV for {split}"):
        try:
            # Validate essential columns exist
            required_cols = ['Dialogue_ID', 'Utterance_ID', 'Speaker', 'Utterance', 'Emotion', 'Sentiment']
            if not all(col in row for col in required_cols):
                # logger.warning(f"Skipping row {idx} due to missing one or more required columns: {required_cols}")
                print(f"Warning: Skipping row {idx} due to missing one or more required columns: {required_cols}")
                missing_audio_count +=1 # Count as effectively missing for reporting
                continue

            dia_id_val = row.get('Dialogue_ID')
            utt_id_val = row.get('Utterance_ID')
            
            # Attempt conversion to int, handle potential errors
            try:
                dia_id = int(dia_id_val)
                utt_id = int(utt_id_val)
            except (ValueError, TypeError):
                # logger.warning(f"Skipping row {idx} due to invalid Dialogue_ID ('{dia_id_val}') or Utterance_ID ('{utt_id_val}').")
                if missing_audio_count < 5: # Limit verbose logging
                    print(f"Warning: Skipping row {idx} in {split} CSV due to invalid/missing Dialogue_ID ('{dia_id_val}') or Utterance_ID ('{utt_id_val}').")
                missing_audio_count += 1
                continue
            
            audio_path = config.processed_audio_output_dir / split / f"dia{dia_id}_utt{utt_id}.wav"
            
            if not audio_path.exists():
                # logger.warning(f"Audio file not found: {audio_path}, skipping item for dataset.")
                if missing_audio_count < 5: # Limit verbose logging
                    print(f"Warning: Audio file not found: {audio_path}, skipping item for dataset.")
                missing_audio_count += 1
                continue
                
            item = {
                'dialogue_id': dia_id,
                'utterance_id': utt_id,
                'speaker': str(row['Speaker']), # Ensure string type
                'text': str(row['Utterance']),   # Ensure string type
                'emotion': str(row['Emotion']), # Ensure string type
                'sentiment': str(row['Sentiment']), # Ensure string type
                'audio_path': str(audio_path), 
                'label': config.label_encoder.get(str(row['Emotion']), -1) 
            }
            
            if item['label'] == -1 and config.label_encoder: # only warn if map is populated
                 # logger.warning(f"Emotion '{row['Emotion']}' for dia{dia_id}_utt{utt_id} not in label_encoder map. Assigning label -1.")
                 print(f"Warning: Emotion '{row['Emotion']}' for dia{dia_id}_utt{utt_id} not in label_encoder map. Assigning label -1.")
            data.append(item)
        
        except Exception as e: # Catch any other unexpected errors per row
            # logger.error(f"Unexpected error processing row {idx} for {split}: {e}. Row data: {row.to_dict()}", exc_info=True)
            print(f"Unexpected error processing row {idx} for {split}: {e}. Skipping.")
            missing_audio_count +=1
            continue # Skip this item
    
    if missing_audio_count > 0:
        # logger.warning(f"{missing_audio_count}/{total_rows_for_split} items were skipped for the {split} split due to missing audio or data issues.")
        print(f"Warning: {missing_audio_count}/{total_rows_for_split} items were skipped for the {split} split due to missing audio or data issues.")
    
    if not data:
        # logger.warning(f"No data collected for {split} split. Returning empty dataset. Check audio paths, CSV content, and config.")
        print(f"Warning: No data collected for {split} split. Returning empty dataset. Check audio paths, CSV content, and config.")
        return Dataset.from_list([]) # HF expects a list, even if empty

    return Dataset.from_list(data)

# Placeholder for BaseConfig if not imported from elsewhere
# class BaseConfig:
#     def __init__(self, raw_data_dir, processed_audio_output_dir, label_encoder):
#         self.raw_data_dir = Path(raw_data_dir)
#         self.processed_audio_output_dir = Path(processed_audio_output_dir)
#         self.label_encoder = label_encoder
#         # Add other necessary attributes used by the functions, e.g., logger
#         # self.logger = logging.getLogger(__name__) # Example

if __name__ == '__main__':
    # This is a basic test stub. You'll want to expand this or move to a proper test suite.
    print("Testing csv_loader.py...")

    # Create a dummy BaseConfig for testing
    # You'll need to create dummy data files and directories for this to run.
    class DummyConfig:
        def __init__(self):
            self.raw_data_dir = Path("temp_test_data/raw")
            self.processed_audio_output_dir = Path("temp_test_data/processed_wavs")
            self.label_encoder = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "surprise": 4, "fear": 5, "disgust": 6}
            # self.logger = logging.getLogger("test_csv_loader")
            # logging.basicConfig(level=logging.INFO)

            # Create dummy directories and files
            (self.raw_data_dir).mkdir(parents=True, exist_ok=True)
            (self.processed_audio_output_dir / "train").mkdir(parents=True, exist_ok=True)
            
            # Dummy CSV
            dummy_train_csv_path = self.raw_data_dir / "train_sent_emo.csv"
            pd.DataFrame({
                'Dialogue_ID': [0, 0, 1], 
                'Utterance_ID': [0, 1, 0], 
                'Speaker': ['Chandler', 'Monica', 'Ross'],
                'Utterance': ['Hi there!', 'How are you?', 'Fine.'], 
                'Emotion': ['neutral', 'joy', 'neutral'], 
                'Sentiment': ['neutral', 'positive', 'neutral']
            }).to_csv(dummy_train_csv_path, index=False)

            # Dummy audio files (empty files for path existence check)
            (self.processed_audio_output_dir / "train" / "dia0_utt0.wav").touch(exist_ok=True)
            # dia0_utt1.wav will be missing to test that path
            (self.processed_audio_output_dir / "train" / "dia1_utt0.wav").touch(exist_ok=True)

    cfg = DummyConfig()
    
    print("\n--- Testing load_metadata ---")
    train_df = load_metadata("train", cfg, limit_dialogues=1)
    if not train_df.empty:
        print(f"Loaded train metadata for 1 dialogue (should be 2 utterances): {len(train_df)} rows")
        print(train_df.head())
    else:
        print("load_metadata for train returned empty DataFrame.")

    train_df_all = load_metadata("train", cfg)
    if not train_df_all.empty:
        print(f"Loaded train metadata for all dialogues: {len(train_df_all)} rows")
    else:
        print("load_metadata for train (all) returned empty DataFrame.")

    print("\n--- Testing create_dataset_from_csv ---")
    train_dataset = create_dataset_from_csv("train", cfg)
    if train_dataset and len(train_dataset) > 0:
        print(f"Created train dataset with {len(train_dataset)} items.")
        print("First item:", train_dataset[0])
        # Expected: 2 items, one for dia0_utt0 and one for dia1_utt0. dia0_utt1 should be skipped.
        assert len(train_dataset) == 2, f"Expected 2 items, got {len(train_dataset)}"
        assert train_dataset[0]['dialogue_id'] == 0 and train_dataset[0]['utterance_id'] == 0
        assert train_dataset[1]['dialogue_id'] == 1 and train_dataset[1]['utterance_id'] == 0
    else:
        print("create_dataset_from_csv for train returned None or empty dataset.")

    # Clean up dummy files (optional, but good practice for tests)
    # import shutil
    # shutil.rmtree("temp_test_data", ignore_errors=True)
    print("\ncsv_loader.py test run finished.") 