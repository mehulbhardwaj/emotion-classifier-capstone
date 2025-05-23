#!/usr/bin/env python3
"""Debug script to test config loading issues."""

import yaml
from configs import Config

def test_config_loading():
    """Test how the Config class handles different architecture names."""
    print("🔍 DEBUGGING CONFIG LOADING")
    print("=" * 50)
    
    # Test 1: Direct YAML loading
    print("\n📁 Testing direct YAML loading...")
    try:
        with open("configs/simple_dialog_rnn.yaml", "r") as f:
            yaml_content = yaml.safe_load(f)
        print(f"YAML content: {yaml_content}")
        print(f"architecture_name in YAML: {yaml_content.get('architecture_name', 'NOT FOUND')}")
    except Exception as e:
        print(f"❌ YAML loading failed: {e}")
        return False
    
    # Test 2: Config.from_yaml()
    print("\n🔧 Testing Config.from_yaml()...")
    try:
        config = Config.from_yaml("configs/simple_dialog_rnn.yaml")
        print(f"Config.architecture_name: {config.architecture_name}")
        print(f"Config type: {type(config)}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False
    
    # Test 3: Test different architecture names
    print("\n🧪 Testing different architecture names...")
    test_configs = {
        "mlp_fusion": "mlp_fusion",
        "dialog_rnn": "dialog_rnn", 
        "todkat_lite": "todkat_lite",
        "invalid_arch": "invalid_arch"
    }
    
    for name, arch in test_configs.items():
        try:
            test_dict = {"architecture_name": arch}
            config = Config.from_dict(test_dict)
            print(f"  {name:15} → {config.architecture_name}")
        except Exception as e:
            print(f"  {name:15} → ERROR: {e}")
    
    # Test 4: Check if Config class validates architecture names
    print("\n🔍 Checking Config class source...")
    import inspect
    config_source = inspect.getsource(Config)
    if "dialog_rnn" in config_source:
        print("✅ dialog_rnn found in Config class")
    else:
        print("❌ dialog_rnn NOT found in Config class")
    
    if "todkat_lite" in config_source:
        print("✅ todkat_lite found in Config class")
    else:
        print("❌ todkat_lite NOT found in Config class")
    
    # Test 5: Manual override test
    print("\n🛠️  Testing manual override...")
    config = Config()
    print(f"Default architecture_name: {config.architecture_name}")
    config.architecture_name = "dialog_rnn"
    print(f"After manual set: {config.architecture_name}")
    
    return True

def test_train_selection():
    """Test the model selection logic from train.py"""
    print("\n🎭 Testing train.py model selection logic...")
    
    config = Config.from_yaml("configs/simple_dialog_rnn.yaml")
    print(f"Loaded config.architecture_name: '{config.architecture_name}'")
    
    # Replicate train.py logic
    if config.architecture_name == "mlp_fusion":
        result = "MultimodalFusionMLP"
    elif config.architecture_name == "todkat_lite":
        result = "TodkatLiteMLP"
    elif config.architecture_name == "dialog_rnn":
        result = "DialogRNNMLP"
    else:
        result = f"ERROR: Unknown architecture '{config.architecture_name}'"
    
    print(f"Model selection result: {result}")
    
    # Check for whitespace or other issues
    arch_repr = repr(config.architecture_name)
    print(f"architecture_name repr: {arch_repr}")
    print(f"architecture_name length: {len(config.architecture_name)}")

if __name__ == "__main__":
    success = test_config_loading()
    if success:
        test_train_selection()
    else:
        print("❌ Config loading tests failed!") 