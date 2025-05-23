#!/usr/bin/env python3
"""
Debug script to test model instantiation and data loading for all three architectures.
"""

import torch
from configs import Config
from utils.data_processor import MELDDataModule
from models.mlp_fusion import MultimodalFusionMLP
from models.dialog_rnn import DialogRNNMLP
from models.todkat_lite import TodkatLiteMLP

def test_architecture(arch_name):
    """Test a specific architecture."""
    print(f"\n{'='*50}")
    print(f"Testing {arch_name.upper()}")
    print(f"{'='*50}")
    
    # Create config
    config = Config()
    config.architecture_name = arch_name
    config.batch_size = 4  # Small batch for testing
    
    # Create data module
    try:
        data_module = MELDDataModule(config)
        data_module.setup()
        train_loader = data_module.train_dataloader()
        print(f"✅ Data module created successfully")
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"✅ Batch loaded successfully")
        print(f"   Batch type: {type(batch)}")
        
        if isinstance(batch, dict):
            print(f"   Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {type(value)}")
        else:
            print(f"   Batch length: {len(batch)}")
            for i, item in enumerate(batch):
                if torch.is_tensor(item):
                    print(f"   Item {i}: {item.shape}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Create model
    try:
        if arch_name == "mlp_fusion":
            model = MultimodalFusionMLP(config)
        elif arch_name == "dialog_rnn":
            model = DialogRNNMLP(config)
        elif arch_name == "todkat_lite":
            model = TodkatLiteMLP(config)
        else:
            raise ValueError(f"Unknown architecture: {arch_name}")
        
        print(f"✅ Model created successfully")
        print(f"   Model type: {type(model).__name__}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            if arch_name == "mlp_fusion":
                # Expects 5-tuple
                wav, wav_mask, txt, txt_mask, labels = batch
                logits = model(wav, wav_mask, txt, txt_mask)
            else:
                # Expects dict
                logits = model(batch)
            
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {logits.shape}")
            print(f"   Expected classes: {config.output_dim}")
            
            if logits.shape[-1] != config.output_dim:
                print(f"⚠️  Warning: Output dimension mismatch!")
                
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"✅ {arch_name.upper()} test completed successfully!")
    return True

def main():
    """Test all architectures."""
    architectures = ["mlp_fusion", "dialog_rnn", "todkat_lite"]
    
    print("🧪 Testing all model architectures...")
    
    results = {}
    for arch in architectures:
        results[arch] = test_architecture(arch)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    for arch, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{arch:<15}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All architectures working correctly!")
        print("You should now get different F1 scores for each model.")
    else:
        print("\n⚠️  Some architectures failed. Check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()