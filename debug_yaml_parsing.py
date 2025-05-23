#!/usr/bin/env python3
"""Debug the YAML parsing process to find why architecture_name is wrong."""

import yaml
from dataclasses import fields
from configs import Config

def debug_yaml_parsing():
    """Debug step by step what happens during YAML parsing."""
    print("🔧 Debugging YAML parsing process...")
    
    yaml_path = "configs/colab_config_dialog_rnn.yaml"
    print(f"📋 Loading: {yaml_path}")
    
    # Step 1: Raw YAML loading
    print("\n1️⃣ Raw YAML loading:")
    with open(yaml_path, 'r') as f:
        raw_content = f.read()
    
    print(f"   Raw content (first 200 chars):")
    print(f"   {repr(raw_content[:200])}")
    
    # Step 2: YAML parsing
    print("\n2️⃣ YAML parsing:")
    config_dict = yaml.safe_load(raw_content)
    print(f"   Parsed dict type: {type(config_dict)}")
    print(f"   Parsed dict: {config_dict}")
    
    if config_dict and 'architecture_name' in config_dict:
        print(f"   ✅ Found architecture_name: '{config_dict['architecture_name']}'")
    else:
        print(f"   ❌ architecture_name not found!")
        print(f"   Available keys: {list(config_dict.keys()) if config_dict else 'None'}")
    
    # Step 3: Dataclass field filtering
    print("\n3️⃣ Dataclass field filtering:")
    valid_keys = {f.name for f in fields(Config)}
    print(f"   Valid dataclass fields: {sorted(valid_keys)}")
    
    filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
    print(f"   Filtered dict: {filtered_dict}")
    
    if 'architecture_name' in filtered_dict:
        print(f"   ✅ architecture_name in filtered dict: '{filtered_dict['architecture_name']}'")
    else:
        print(f"   ❌ architecture_name missing from filtered dict!")
    
    # Step 4: Config creation
    print("\n4️⃣ Config creation:")
    config = Config(**filtered_dict)
    print(f"   Final config.architecture_name: '{config.architecture_name}'")
    
    # Step 5: Compare with manual creation
    print("\n5️⃣ Manual test:")
    manual_config = Config(architecture_name="dialog_rnn")
    print(f"   Manual config.architecture_name: '{manual_config.architecture_name}'")
    
    return config.architecture_name == "dialog_rnn"

if __name__ == "__main__":
    success = debug_yaml_parsing()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: YAML parsing debug completed") 