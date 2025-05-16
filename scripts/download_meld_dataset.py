import os
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm
import argparse
import sys

# Adjust sys.path to allow importing from common modules
SCRIPT_DIR = Path(__file__).resolve().parent  # .../scripts
PROJECT_WORKSPACE_ROOT = SCRIPT_DIR.parent      # .../emotion-classification-dlfa
if str(PROJECT_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_WORKSPACE_ROOT))

from configs.base_config import BaseConfig
# from common.script_utils import convert_mp4_to_wav_meld # If we decide to call conversion from here

cfg = BaseConfig()
DEFAULT_RAW_DATA_DIR = cfg.raw_data_dir

def download_meld_raw_tar_gz(data_dir: Path, force_download: bool = False):
    """
    Downloads and extracts MELD.Raw.tar.gz which contains CSVs and further tarballs for videos.
    (Adapted from old data_loader.py - download_dataset function)
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    download_path = data_dir / "MELD.Raw.tar.gz"
    # Marker for successful extraction of the main tarball (e.g., a key CSV file)
    extracted_marker_file = data_dir / "train_sent_emo.csv" 
    # MELD.Raw.tar.gz also contains train.tar.gz, dev.tar.gz, test.tar.gz for videos.
    train_videos_tar = data_dir / "train.tar.gz"
    dev_videos_tar = data_dir / "dev.tar.gz"
    test_videos_tar = data_dir / "test.tar.gz"

    url = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz" # Original MELD source
    # Alternative if above is down: "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz"
    # However, HF one might have a different internal structure (e.g., already split).
    # The original URL is preferred if the goal is to match the old script's behavior.

    MIN_EXPECTED_TAR_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB (rough estimate for MELD.Raw.tar.gz)

    if extracted_marker_file.exists() and train_videos_tar.exists() and not force_download:
        print(f"Dataset main archive already extracted at '{data_dir}'. CSVs and video tarballs present.")
        if download_path.exists():
            print(f"Removing leftover main archive: '{download_path}'")
            try: download_path.unlink() 
            except OSError as e: print(f"Warning: Could not remove '{download_path}': {e}")
        return True # Indicate main archive processed

    # Download MELD.Raw.tar.gz
    if not download_path.exists() or force_download or download_path.stat().st_size < MIN_EXPECTED_TAR_SIZE_BYTES:
        print(f"Downloading {url} to {download_path}...")
        current_pos = download_path.stat().st_size if download_path.exists() and not force_download else 0
        headers = {'Range': f'bytes={current_pos}-'} if current_pos > 0 else {}
        mode = 'ab' if current_pos > 0 else 'wb'
        if force_download and download_path.exists(): download_path.unlink(); current_pos =0; mode = 'wb'
        
        try:
            with requests.get(url, headers=headers, stream=True, timeout=3600) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                if total_size > 0 and current_pos >= total_size: # Already complete
                    print(f"File {download_path} already complete.")
                else:
                    with open(download_path, mode) as f, tqdm(
                        total=total_size, initial=current_pos, unit='iB', unit_scale=True, desc=url.split('/')[-1]
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            size = f.write(chunk)
                            pbar.update(size)
            print(f"Download of '{download_path}' complete.")
        except requests.exceptions.RequestException as e:
            print(f"ERROR during download of {url}: {e}. Partial file may exist.")
            return False # Indicate failure
    else:
        print(f"Dataset archive '{download_path}' already exists and meets size criteria.")

    # Extract MELD.Raw.tar.gz
    if not extracted_marker_file.exists() or not train_videos_tar.exists() or force_download:
        print(f"Extracting dataset from '{download_path}' to '{data_dir}'...")
        try:
            with tarfile.open(download_path, 'r:gz') as tar:
                # Check members to avoid extracting into MELD.Raw/MELD.Raw if not intended
                # The original MELD.Raw.tar.gz from umich usually extracts its contents (CSVs, other tars) directly.
                members_to_extract = []
                for member in tar.getmembers():
                    # Avoid path traversal vulnerabilities, though less critical for trusted source
                    if member.name.startswith(("../", "/")):
                        print(f"Skipping potentially unsafe member: {member.name}")
                        continue
                    # If files are nested under a single directory like 'MELD.Raw' in the tar,
                    # we might want to strip that leading component upon extraction.
                    # For now, assume direct extraction or handle nesting post-extraction.
                    members_to_extract.append(member)
                
                for member in tqdm(members_to_extract, desc="Extracting main archive contents"):
                    tar.extract(member, path=data_dir)
            print("Main archive extraction complete.")
            
            # Handle potential MELD.Raw/MELD.Raw nesting if it occurs
            # This structure was seen with some MELD downloads.
            nested_meld_raw_dir = data_dir / "MELD.Raw" / "MELD.Raw"
            if nested_meld_raw_dir.exists() and nested_meld_raw_dir.is_dir():
                print(f"Detected nested MELD.Raw directory: {nested_meld_raw_dir}")
                print(f"Moving contents from {nested_meld_raw_dir} to {data_dir}...")
                for item in nested_meld_raw_dir.iterdir():
                    target_item_path = data_dir / item.name
                    if target_item_path.exists():
                        print(f"  Item {item.name} already exists in {data_dir}, skipping move.")
                    else:
                        item.rename(target_item_path)
                        print(f"  Moved {item.name} to {data_dir}")
                # Attempt to remove the now empty MELD.Raw/MELD.Raw and then MELD.Raw if also empty
                try: nested_meld_raw_dir.rmdir(); (data_dir / "MELD.Raw").rmdir() 
                except OSError: print("Could not remove all nested MELD.Raw dirs, may not be empty.")

            if extracted_marker_file.exists() and train_videos_tar.exists():
                print(f"Main archive extraction verified. Removing archive: '{download_path}'")
                try: download_path.unlink()
                except OSError as e: print(f"Warning: Could not remove archive '{download_path}': {e}")
            else:
                print(f"CRITICAL ERROR: Post-extraction check failed for main archive. Markers not found.")
                return False
        except Exception as e:
            print(f"ERROR during extraction of '{download_path}': {e}. Archive kept.")
            return False
    return True

def extract_video_archives(data_dir: Path, force_extract: bool = False):
    """Extracts train.tar.gz, dev.tar.gz, test.tar.gz into respective _videos folders."""
    video_tars_info = {
        "train": {"tar": data_dir / "train.tar.gz", "extract_to": data_dir / "train_videos", "marker": data_dir / "train_videos" / "dia0_utt0.mp4"},
        "dev":   {"tar": data_dir / "dev.tar.gz",   "extract_to": data_dir / "dev_videos",   "marker": data_dir / "dev_videos" / "dia0_utt0.mp4"},
        # Test set videos might have different naming or structure; this is a common case.
        "test":  {"tar": data_dir / "test.tar.gz",  "extract_to": data_dir / "test_videos",  "marker": data_dir / "test_videos" / "final_videos_testdia0_utt0.mp4"}
    }

    all_successful = True
    for split_name, info in video_tars_info.items():
        tar_path = info["tar"]
        extract_dir = info["extract_to"]
        # A more robust marker would be checking for a significant number of files or a specific file known to exist.
        # For simplicity, we check if the directory is non-empty or a sample file exists if known.
        # The current marker in `info` is a placeholder for a known file.

        if not tar_path.exists():
            print(f"Video archive {tar_path.name} not found. Skipping extraction for {split_name}.")
            all_successful = False
            continue

        # Check if extraction seems complete (directory exists and is not empty)
        extraction_seems_complete = extract_dir.exists() and any(extract_dir.iterdir())
        if extraction_seems_complete and not force_extract:
            print(f"Video files for {split_name} seem already extracted to {extract_dir}. Skipping.")
            continue

        print(f"Extracting {tar_path.name} to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                for member in tqdm(tar.getmembers(), desc=f"Extracting {split_name} videos"):
                    tar.extract(member, path=extract_dir)
            print(f"Extraction of {split_name} videos complete.")
            # Optionally, remove the tarball after successful extraction
            # tar_path.unlink()
        except Exception as e:
            print(f"ERROR extracting {tar_path.name}: {e}")
            all_successful = False
    return all_successful

def main():
    parser = argparse.ArgumentParser(description="Download and perform initial extraction of MELD dataset.")
    parser.add_argument(
        "--data_dir", type=str, default=str(DEFAULT_RAW_DATA_DIR),
        help=f"Directory to download and extract MELD data. Default: {DEFAULT_RAW_DATA_DIR}"
    )
    parser.add_argument(
        "--force_download_main", action="store_true", 
        help="Force re-download and extraction of the main MELD.Raw.tar.gz."
    )
    parser.add_argument(
        "--force_extract_videos", action="store_true",
        help="Force re-extraction of train/dev/test video tarballs even if target folders exist."
    )
    args = parser.parse_args()

    target_data_dir = Path(args.data_dir)
    print(f"MELD Dataset Setup Script")
    print(f"Target Raw Data Directory: {target_data_dir}")

    if download_meld_raw_tar_gz(target_data_dir, args.force_download_main):
        print("\nProceeding to extract video archives...")
        if extract_video_archives(target_data_dir, args.force_extract_videos):
            print("\nMELD dataset download and video archive extraction complete.")
            print(f"CSVs and video tarballs should be in: {target_data_dir}")
            print(f"Video files (MP4s) should be in: {target_data_dir}/<split>_videos/")
            print("Next, run scripts/preprocess_meld.py to convert MP4s to WAVs.")
            print("Then, run scripts/extract_features_meld.py to cache features.")
        else:
            print("\nVideo archive extraction faced issues. Please check logs.")
    else:
        print("\nDownload and extraction of MELD.Raw.tar.gz failed. Please check logs.")

if __name__ == "__main__":
    main() 