#!/usr/bin/env python3
"""Test script to verify the debug data processor fix works."""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_debug_data_fix():
    """Test that the debug data processor loads data even with missing audio files."""
    try:
        print("🔧 Testing debug data processor fix...")
        
        # Import after path setup
        from utils.data_processor_debug import MELDDataset, MELDDataModule
        from configs.base_config_debug import Config
        
        # Create minimal config
        config = Config()
        config.data_root = "/content/drive/MyDrive/dlfa_capstone/meld_data"
        config.architecture_name = "dialog_rnn"
        config.batch_size = 2
        config.max_sequence_length = 20
        
        print("✅ Config created")
        
        # Test dataset loading
        print("📊 Testing dataset loading...")
        dataset = MELDDataset(config, "dev")  # Use dev split (smaller)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        if len(dataset) == 0:
            print("❌ Dataset is empty - fix didn't work")
            return False
        
        # Test first sample
        print("📦 Testing first sample...")
        sample = dataset[0]
        print(f"✅ Sample loaded: {type(sample)}")
        
        if hasattr(sample, 'get') and 'utterances' in sample:
            print(f"   Dialog has {len(sample['utterances'])} utterances")
            if len(sample['utterances']) > 0:
                print("🎉 SUCCESS: Found non-empty dialog!")
            else:
                print("⚠️  WARNING: Dialog is still empty")
        
        # Test data module
        print("🔍 Testing data module...")
        data_module = MELDDataModule(config)
        data_module.setup()
        print("✅ Data module setup completed")
        
        # Test dataloader
        print("📦 Testing dataloader...")
        val_loader = data_module.val_dataloader()
        print(f"✅ Val dataloader created: {len(val_loader)} batches")
        
        # Test one batch
        try:
            batch = next(iter(val_loader))
            print(f"✅ Batch loaded successfully!")
            
            if 'dialog_mask' in batch:
                valid_utterances = batch['dialog_mask'].sum().item()
                print(f"   Valid utterances in batch: {valid_utterances}")
                if valid_utterances > 0:
                    print("🎉 SUCCESS: Found valid dialogue data in batch!")
                    return True
                else:
                    print("⚠️  WARNING: Batch still has no valid utterances")
                    return False
        except Exception as e:
            print(f"❌ Error loading batch: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing debug data processor fix...")
    success = test_debug_data_fix()
    if success:
        print("\n🏆 Debug data processor fix PASSED!")
    else:
        print("\n💥 Debug data processor fix FAILED.") 