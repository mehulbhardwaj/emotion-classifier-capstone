#!/usr/bin/env python3
"""Debug script to check HuggingFace dataset fields and structure."""

import sys
from pathlib import Path
import datasets as hf
from collections import defaultdict

def check_dataset_structure():
    """Check the actual structure of the HuggingFace dataset."""
    
    # Try to load a dataset split
    try:
        # Assume we're in Colab environment with mounted drive
        root = Path("/content/drive/MyDrive/dlfa_capstone/meld_data/processed/features/hf_datasets")
        if not root.exists():
            # Fallback for local testing
            root = Path("meld_data/processed/features/hf_datasets")
        
        print(f"🔍 Looking for dataset at: {root}")
        
        for split in ["train", "dev", "test"]:
            split_path = root / split
            if split_path.exists():
                print(f"\n📂 Checking {split} split...")
                try:
                    ds = hf.load_from_disk(str(split_path))
                    print(f"   ✅ Loaded {split}: {len(ds)} samples")
                    
                    # Check first sample
                    if len(ds) > 0:
                        sample = ds[0]
                        print(f"   📋 Available fields: {list(sample.keys())}")
                        
                        # Check dialogue_id variants
                        dialogue_fields = [k for k in sample.keys() if 'dialog' in k.lower() or 'dialogue' in k.lower()]
                        print(f"   🎭 Dialogue-related fields: {dialogue_fields}")
                        
                        # Check a few samples for dialogue_id mapping
                        if len(ds) >= 10:
                            dialogue_mapping = defaultdict(list)
                            print(f"   🔢 Checking dialogue ID distribution...")
                            
                            for idx in range(min(len(ds), 100)):  # Check first 100
                                row = ds[idx]
                                
                                # Try different field names
                                dialogue_id = None
                                for field in ['dialogue_id', 'Dialogue_ID', 'dialog_id', 'Dialog_ID']:
                                    if field in row:
                                        dialogue_id = row[field]
                                        break
                                
                                if dialogue_id is not None:
                                    dialogue_mapping[dialogue_id].append(idx)
                                else:
                                    print(f"      ⚠️  No dialogue_id found in sample {idx}")
                                    break
                            
                            print(f"   📊 Found {len(dialogue_mapping)} unique dialogues in first 100 samples")
                            if dialogue_mapping:
                                # Show distribution
                                lengths = [len(indices) for indices in dialogue_mapping.values()]
                                print(f"      Utterances per dialogue: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
                                
                                # Show first few dialogue IDs
                                first_few = list(dialogue_mapping.keys())[:5]
                                print(f"      First few dialogue IDs: {first_few}")
                                
                                for did in first_few[:3]:
                                    indices = dialogue_mapping[did][:3]  # First 3 utterances
                                    print(f"      Dialogue {did}: indices {indices}")
                            else:
                                print("      ❌ No valid dialogue mappings found!")
                
                except Exception as e:
                    print(f"   ❌ Error loading {split}: {e}")
            else:
                print(f"   ❌ {split} split not found at {split_path}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🕵️ Debugging HuggingFace dataset structure...")
    success = check_dataset_structure()
    if not success:
        print("\n💡 Suggestions:")
        print("   1. Make sure you're running this in Colab with drive mounted")
        print("   2. Check that the dataset was properly preprocessed")
        print("   3. Verify the data paths in your config")
    else:
        print("\n✅ Dataset structure check completed!") 