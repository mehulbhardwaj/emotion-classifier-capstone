#!/usr/bin/env python3
"""Test script to verify dialogue mapping works correctly."""

import sys
from pathlib import Path
from collections import defaultdict

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

def test_dialogue_mapping():
    """Test the dialogue mapping logic."""
    try:
        from utils.data_processor import MELDDataModule
        from configs.base_config import BaseConfig
        
        # Create a minimal config
        config = BaseConfig()
        config.data_root = "/content/drive/MyDrive/dlfa_capstone/meld_data"
        config.text_encoder_model_name = "roberta-base"
        config.architecture_name = "dialog_rnn"
        config.batch_size = 4
        config.output_dim = 7
        
        print("🧪 Testing dialogue mapping...")
        
        # Create data module
        dm = MELDDataModule(config)
        dm.setup()
        
        print("✅ DataModule setup completed")
        
        # Test dialogue mapping for each split
        for split in ["dev"]:  # Start with dev (smaller)
            print(f"\n📊 Testing {split} split...")
            ds = dm.ds[split]
            print(f"   Dataset size: {len(ds)}")
            
            # Test the mapping logic
            mapping = defaultdict(list)
            for idx in range(min(len(ds), 100)):  # Test first 100 samples
                raw_row = ds.ds[idx]
                did = raw_row.get("Dialogue_ID", raw_row.get("dialogue_id"))
                dialogue_id = int(did) if did is not None else -1
                if dialogue_id != -1:
                    mapping[dialogue_id].append(idx)
            
            print(f"   ✅ Found {len(mapping)} dialogues in first 100 samples")
            
            if mapping:
                lengths = [len(indices) for indices in mapping.values()]
                print(f"   📈 Utterances per dialogue: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
                
                # Test a few dialogue samples
                first_dialogues = list(mapping.keys())[:3]
                for did in first_dialogues:
                    indices = mapping[did]
                    print(f"   🎭 Dialogue {did}: {len(indices)} utterances at indices {indices[:5]}...")
                    
                    # Test actual data loading for one utterance
                    if indices:
                        try:
                            sample = ds[indices[0]]
                            print(f"      ✅ Sample loaded: dialogue_id={sample['dialogue_id']}, labels shape={sample['labels'].shape}")
                        except Exception as e:
                            print(f"      ❌ Error loading sample: {e}")
                
                print(f"   ✅ Dialogue mapping test passed for {split}")
            else:
                print(f"   ❌ No valid dialogues found in {split}")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🔧 Testing dialogue mapping fix...")
    success = test_dialogue_mapping()
    if success:
        print("\n🎉 Dialogue mapping test completed successfully!")
    else:
        print("\n💥 Dialogue mapping test failed!") 