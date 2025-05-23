#!/usr/bin/env python3
"""Debug script to test config loading from YAML."""

from configs import Config

def test_config_loading():
    """Test loading config from YAML file."""
    try:
        print("🔧 Testing config loading...")
        
        # Load from YAML
        config_path = "configs/colab_config_dialog_rnn.yaml"
        print(f"📋 Loading config from: {config_path}")
        
        config = Config.from_yaml(config_path)
        
        print(f"✅ Config loaded successfully!")
        print(f"📊 Config attributes:")
        print(f"   architecture_name: '{config.architecture_name}'")
        print(f"   type: {type(config.architecture_name)}")
        print(f"   repr: {repr(config.architecture_name)}")
        
        # Test the model selection logic
        print(f"\n🤖 Testing model selection logic...")
        
        if config.architecture_name == "mlp_fusion":
            print("   ➡️  Would select: MultimodalFusionMLP")
        elif config.architecture_name == "dialog_rnn":
            print("   ➡️  Would select: DialogRNNMLP ✅")
        elif config.architecture_name == "todkat_lite":
            print("   ➡️  Would select: TodkatLiteMLP")
        else:
            print(f"   ❌ Unknown architecture: '{config.architecture_name}'")
        
        # Show other relevant config values
        print(f"\n📊 Other config values:")
        print(f"   batch_size: {config.batch_size}")
        print(f"   learning_rate: {config.learning_rate}")
        print(f"   output_dim: {config.output_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing config loading...")
    success = test_config_loading()
    if success:
        print("\n✅ Config loading test PASSED!")
    else:
        print("\n❌ Config loading test FAILED!") 