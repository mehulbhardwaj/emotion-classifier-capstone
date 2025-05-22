import os
import tarfile
from pathlib import Path
import requests
from tqdm import tqdm
import argparse
import sys
import shutil
import subprocess
import tempfile
import hashlib

# Adjust sys.path to allow importing from common modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_WORKSPACE_ROOT = SCRIPT_DIR.parent
if str(PROJECT_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_WORKSPACE_ROOT))

from configs.base_config import BaseConfig

def safe_strip_depth(tar_path: Path) -> int:
    """Return how many leading components are common to every file path."""
    with tarfile.open(tar_path) as tf:
        parts = [m.name.split('/') for m in tf.getmembers() if m.isfile()]
    d = 0
    while len({p[d] for p in parts if len(p) > d}) == 1:
        d += 1
    return d

def dedupe_mp4_dir(mp4_dir: Path):
    """Remove byte-for-byte duplicate MP4s inside mp4_dir."""
    seen: dict[str, Path] = {}
    for f in mp4_dir.glob("*.mp4"):
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen:
            print(f"    • removing duplicate {f.name}")
            f.unlink()
        else:
            seen[h] = f
    # optional: drop any nested dirs accidentally left over
    for d in mp4_dir.iterdir():
        if d.is_dir():
            shutil.rmtree(d)

def strip_depth(archive: Path) -> int:
    """Return minimal number of leading components common to all members."""
    print(f"    Determining strip depth for {archive.name}...")
    try:
        with tarfile.open(archive) as tf:
            names = [m.name for m in tf.getmembers() if m.isfile() and m.name != '.']
        if not names:
            print(f"    ⚠️ No files found in {archive.name} to determine depth. Assuming depth 0.")
            return 0
        
        first_parts_list = [n.split('/') for n in names]
        first_parts_list = [fp for fp in first_parts_list if fp and fp[0]] 
        if not first_parts_list:
            print(f"    ⚠️ No valid path structures found in {archive.name} after splitting. Assuming depth 0.")
            return 0

        depth = 0
        while True:
            current_level_parts = []
            for p in first_parts_list:
                if len(p) > depth:
                    current_level_parts.append(p[depth])
                else:
                    return depth 
            
            if not current_level_parts: 
                break

            unique_parts = set(current_level_parts)
            if len(unique_parts) == 1:
                depth += 1
            else:
                break
        print(f"    Calculated strip depth for {archive.name}: {depth}")
        return depth
    except tarfile.ReadError as e:
        print(f"    ⚠️ Error reading {archive.name} to determine depth: {e}. Assuming depth 0.")
        return 0 
    except Exception as e_strip:
        print(f"    ⚠️ Unexpected error determining depth for {archive.name}: {e_strip}. Assuming depth 0.")
        return 0

def extract_flat(archive: Path, dest: Path, depth: int):
    """tar -xzf archive -C dest --strip-components=depth"""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"    Executing: tar -xzf {archive} -C {dest} --strip-components={depth}")
    subprocess.check_call(
        ["tar", "-xzf", str(archive), "-C", str(dest),
         f"--strip-components={depth}"]
    )

def build_train_csv(data_dir: Path):
    """Concat train_splits/*.csv → train_sent_emo.csv (if not already)."""
    out = data_dir / "train_sent_emo.csv"
    if out.exists():
        print(f"✔ {out.name} already exists ({out.stat().st_size} bytes). Skipping build.")
        return
    
    print(f"Attempting to build {out.name} from CSVs in a 'train_splits' directory...")
    
    possible_train_splits_paths_in_archive = [
        "MELD.Raw/train_splits/", 
        "train_splits/"           
    ]

    extracted_train_splits_dir = None
    main_archive = data_dir / "MELD.Raw.tar.gz"

    if not main_archive.exists():
        print(f"⚠️ {main_archive.name} not found in {data_dir}. Cannot build {out.name} from it.")
        local_train_splits_dir = data_dir / "train_splits"
        if local_train_splits_dir.is_dir():
            print(f"  Found local directory: {local_train_splits_dir}. Will attempt to use CSVs from here.")
            extracted_train_splits_dir = local_train_splits_dir
        else:
            print(f"  Local 'train_splits' directory also not found. Cannot proceed to build {out.name}.")
            return
    else:
        with tempfile.TemporaryDirectory(prefix="meld_train_csv_") as tmp:
            tmp_path = Path(tmp)
            print(f"  Created temporary directory for extraction: {tmp_path}")
            try:
                found_in_archive = False
                for archive_path_to_extract in possible_train_splits_paths_in_archive:
                    print(f"    Attempting to extract '{archive_path_to_extract}' from {main_archive.name} into {tmp_path}...")
                    tar_command = [
                        "tar", "-xzf", str(main_archive),
                        "-C", str(tmp_path), 
                        archive_path_to_extract.rstrip('/') 
                    ]
                    subprocess.check_call(tar_command)
                    
                    potential_extracted_path = tmp_path / archive_path_to_extract.rstrip('/')
                    if potential_extracted_path.is_dir():
                        extracted_train_splits_dir = potential_extracted_path
                        print(f"    Successfully extracted '{archive_path_to_extract}' to {extracted_train_splits_dir}")
                        found_in_archive = True
                        break 
                    else:
                        print(f"    Extraction of '{archive_path_to_extract}' did not result in directory {potential_extracted_path}.")

                if not found_in_archive:
                    print(f"⚠️  'train_splits' (or variants) not found within {main_archive.name} at expected locations.")
                    local_train_splits_dir = data_dir / "train_splits"
                    if local_train_splits_dir.is_dir():
                        print(f"  Found local directory: {local_train_splits_dir}. Will use CSVs from here.")
                        extracted_train_splits_dir = local_train_splits_dir
                    else:
                        print(f"  Local 'train_splits' also not found. No train CSV produced for {out.name}.")
                        return 

            except subprocess.CalledProcessError as e:
                print(f"  ERROR: 'tar' command failed during build_train_csv extraction: {e}")
                print(f"    Command: {' '.join(e.cmd)}")
                print(f"    Stderr: {e.stderr.decode(errors='ignore') if e.stderr else 'N/A'}")
                print(f"  Cannot proceed to build {out.name}.")
                return 
            except Exception as e_extract:
                print(f"  An unexpected error occurred during extraction for build_train_csv: {e_extract}")
                print(f"  Cannot proceed to build {out.name}.")
                return
    
    if not extracted_train_splits_dir or not extracted_train_splits_dir.is_dir():
        print(f"Logic error: extracted_train_splits_dir not set or not a directory. Path: {extracted_train_splits_dir}")
        return

    parts = sorted(extracted_train_splits_dir.glob("*.csv"))
    if not parts:
        print(f"⚠️  No CSV files found in {extracted_train_splits_dir}; no train CSV produced for {out.name}.")
        return
    
    print(f"  Found {len(parts)} CSV files in {extracted_train_splits_dir} for concatenation: {[p.name for p in parts]}")
    try:
        with out.open("w", encoding='utf-8') as fout:
            header = ""
            with open(parts[0], 'r', encoding='utf-8') as first_file:
                header = first_file.readline().strip()
            
            if not header or "Dialogue_ID" not in header: 
                print(f"⚠️  Could not read a valid header from {parts[0]}. Expected MELD CSV format. Aborting {out.name} build.")
                if out.exists(): out.unlink() 
                return
            fout.write(header + "\n")
            
            total_rows_written = 0
            with open(parts[0], 'r', encoding='utf-8') as first_file_data:
                first_file_data.readline() 
                for line in first_file_data:
                    fout.write(line)
                    total_rows_written += 1
            
            for p in parts[1:]:
                with open(p, 'r', encoding='utf-8') as part_file:
                    part_file.readline() 
                    for line in part_file:
                        fout.write(line)
                        total_rows_written += 1
        print(f"✔ Wrote {out.name} with {total_rows_written} data rows (plus 1 header row). Final size: {out.stat().st_size} bytes.")
    except Exception as e_concat:
        print(f"❌ Error during CSV concatenation for {out.name}: {e_concat}")
        if out.exists(): 
            print(f"  Cleaning up partially written or erroneous file: {out.name}")
            out.unlink()

cfg = BaseConfig()
DEFAULT_RAW_DATA_DIR = cfg.raw_data_dir

def extract_and_flatten(archive_path: Path, dest_dir: Path):
    """
    Extracts archive_path into dest_dir, removing the top-level folder
    (e.g., MELD.Raw/) so everything lands directly under dest_dir.
    Uses the system's 'tar' command for robust --strip-components behavior.
    """
    print(f"Extracting and flattening '{archive_path.name}' into '{dest_dir}'...")
    dest_dir.mkdir(parents=True, exist_ok=True) 
    try:
        subprocess.check_call([
            "tar",
            "-xzf", str(archive_path), 
            "-C", str(dest_dir),       
            "--strip-components=1"     
        ])
        print(f"  Successfully extracted and flattened {archive_path.name} to {dest_dir}")
        
        extracted_files = list(dest_dir.glob('*'))
        if extracted_files:
            print(f"    -> Contents of {dest_dir} after extraction (first 5 of {len(extracted_files)}):")
            for i, f_path in enumerate(extracted_files[:5]): 
                print(f"      - {f_path.name}")
            if len(extracted_files) > 5:
                print(f"      ... and {len(extracted_files) - 5} more items.")
        else:
            print(f"    -> WARNING: {dest_dir} is empty after claimed successful extraction of {archive_path.name}.")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: 'tar' command failed with exit code {e.returncode} for {archive_path.name}.")
        print(f"  Command: {' '.join(e.cmd)}")
        if e.stdout:
            print(f"  Stdout: {e.stdout.decode(errors='ignore')}")
        if e.stderr:
            print(f"  Stderr: {e.stderr.decode(errors='ignore')}")
        return False
    except FileNotFoundError:
        print(f"  ERROR: 'tar' command not found. Please ensure tar is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"  An unexpected error occurred during tar extraction of {archive_path.name}: {e}")
        return False

def download_meld_raw_tar_gz(data_dir: Path, force_download: bool = False):
    """
    Downloads MELD.Raw.tar.gz, extracts its primary contents (CSVs, video tarballs),
    using a robust flattening method, and then extracts individual video tarballs.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    main_archive_path = data_dir / "MELD.Raw.tar.gz"
    
    if not main_archive_path.exists() or force_download or main_archive_path.stat().st_size < (500 * 1024 * 1024): 
        if main_archive_path.exists() and force_download:
            print(f"Force download: Removing existing main archive '{main_archive_path}'.")
            main_archive_path.unlink()
        elif main_archive_path.exists() and main_archive_path.stat().st_size < (500 * 1024 * 1024):
            print(f"Existing main archive '{main_archive_path}' is too small ({main_archive_path.stat().st_size} bytes). Re-downloading.")
            main_archive_path.unlink()
        
        if not main_archive_path.exists() or force_download: 
            print(f"Downloading MELD.Raw.tar.gz to '{main_archive_path}'...")
            url = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status() 
                total_size = int(response.headers.get('content-length', 0))
                with open(main_archive_path, 'wb') as f, tqdm(
                    desc=main_archive_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        bar.update(size)
                print(f"Download complete: '{main_archive_path}' ({main_archive_path.stat().st_size} bytes).")
                if main_archive_path.stat().st_size < (500 * 1024 * 1024):
                    print(f"WARNING: Downloaded file size is unexpectedly small. Please check the file.")
            except requests.exceptions.RequestException as e:
                print(f"ERROR: Download failed: {e}")
                if main_archive_path.exists(): 
                    main_archive_path.unlink()
                return False
    else:
        print(f"Main archive '{main_archive_path}' already exists and meets size criteria. Skipping download.")

    print(f"Extracting main archive from '{main_archive_path}' to '{data_dir}' using flattening method...")
    if not extract_and_flatten(main_archive_path, data_dir):
        print(f"CRITICAL ERROR: Failed to extract and flatten '{main_archive_path}'. Cannot proceed.")
        return False
    print("Initial extraction of main archive complete.")
    build_train_csv(data_dir) # Call build_train_csv right after flattening MELD.Raw.tar.gz

    print("\n► Extracting each split tarball")
    for tar_path in sorted(data_dir.glob("*.tar.gz")):
        if tar_path.name == main_archive_path.name:
            continue
        split = tar_path.stem.split(".")[0]          # train/dev/test
        out_dir = data_dir / f"{split}_videos"
        shutil.rmtree(out_dir, ignore_errors=True)   # always start fresh

        depth = safe_strip_depth(tar_path)
        print(f"  · {tar_path.name}: strip-components={depth} → {out_dir.name}/")
        extract_flat(tar_path, out_dir, depth)

        # merge/dedupe if previous bad extractions left nested folders
        for nested in out_dir.rglob("*.mp4"):
            if nested.parent != out_dir:
                nested.rename(out_dir / nested.name)
        dedupe_mp4_dir(out_dir)
        count = len(list(out_dir.glob("*.mp4")))
        print(f"    ✔ final MP4 count: {count}")

    print("\n► Final check for essential video directories and MP4 content...")
    essential_video_dirs_names = ["train_videos", "dev_videos", "test_videos"]
    all_essential_dirs_ok = True
    for dir_name in essential_video_dirs_names:
        video_dir_path = data_dir / dir_name
        if not video_dir_path.is_dir():
            print(f"  ❌ CRITICAL: Expected video directory '{video_dir_path}' is MISSING.")
            all_essential_dirs_ok = False
            continue 
        
        mp4_files_in_dir = list(video_dir_path.rglob("*.mp4")) 
        if not mp4_files_in_dir:
            print(f"  ❌ CRITICAL: Video directory '{video_dir_path}' exists BUT CONTAINS NO MP4 files.")
            all_essential_dirs_ok = False
        else:
            print(f"  ✔ Found {len(mp4_files_in_dir)} MP4 files in '{video_dir_path}'.")

    if not all_essential_dirs_ok:
        print(f"\n❌ CRITICAL FAILURE: Not all essential video directories ({', '.join(essential_video_dirs_names)}) under '{data_dir}' are present and populated with MP4s.")
        print(f"  Please check the source tarballs (MELD.Raw.tar.gz and individual split tarballs like train.tar.gz) and any extraction errors above.")
        return False

    print(f"\n✅ Download, extraction, and train CSV generation appear complete under {data_dir.resolve()}")
    csv_files_present = list(data_dir.glob("*_sent_emo.csv"))
    print(f"CSVs expected/found in '{data_dir}': {[f.name for f in csv_files_present]}")
    print(f"Video files should be in respective subdirectories like: '{data_dir / 'train_videos'}', '{data_dir / 'dev_videos'}', etc.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and perform initial extraction of MELD dataset.")
    parser.add_argument(
        "--data_dir", type=str, default=str(DEFAULT_RAW_DATA_DIR),
        help=f"Directory to download and extract MELD data. Default: {DEFAULT_RAW_DATA_DIR}"
    )
    parser.add_argument(
        "--force_download_main", action="store_true", 
        help="Force re-download of MELD.Raw.tar.gz. Extraction will also be attempted."
    )
    args = parser.parse_args()

    target_data_dir = Path(args.data_dir)
    print(f"MELD Dataset Setup Script")
    print(f"Target Raw Data Directory: {target_data_dir}")

    if download_meld_raw_tar_gz(target_data_dir, args.force_download_main):
        print("MELD dataset download and all archive extractions (CSVs, video tarballs into _videos folders) complete.")
        print("Next Steps (typically handled by main.py --prepare_data or similar):")
        print("1. Convert MP4s (in <split>_videos folders) to WAVs.")
        print("2. Proceed with feature extraction or Hugging Face dataset creation as per your pipeline.")
    else:
        print("Download and/or processing of the MELD dataset failed. Please check logs above.")

if __name__ == "__main__":
    main()