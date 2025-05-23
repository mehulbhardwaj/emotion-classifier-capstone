#!/usr/bin/env python3
"""
Quick test script for a single architecture.

Usage: python test_single_architecture.py <architecture_name>
Examples:
  python test_single_architecture.py mlp_fusion
  python test_single_architecture.py dialog_rnn
  python test_single_architecture.py todkat_lite
"""

import sys
import subprocess
import time
from datetime import datetime


def test_architecture(arch_name):
    """Test a single architecture."""
    
    # Config file mapping
    config_map = {
        'mlp_fusion': 'configs/colab_config_mlp_fusion.yaml',
        'dialog_rnn': 'configs/colab_config_dialog_rnn.yaml',
        'todkat_lite': 'configs/colab_config_todkat_lite.yaml'
    }
    
    if arch_name not in config_map:
        print(f"❌ Unknown architecture: {arch_name}")
        print(f"Available: {list(config_map.keys())}")
        return False
        
    config_path = config_map[arch_name]
    
    print(f"🚀 Testing {arch_name.upper()}")
    print(f"📋 Config: {config_path}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run training with real-time output
        result = subprocess.run(
            ["python", "train.py", "--config", config_path],
            timeout=3600  # 1 hour timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print(f"✅ {arch_name.upper()} training completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return True
        else:
            print(f"\n❌ {arch_name.upper()} training failed!")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {arch_name.upper()} training timed out after 1 hour")
        return False
    except KeyboardInterrupt:
        print(f"\n🛑 {arch_name.upper()} training interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 {arch_name.upper()} training crashed: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_single_architecture.py <architecture_name>")
        print("Available architectures: mlp_fusion, dialog_rnn, todkat_lite")
        sys.exit(1)
        
    arch_name = sys.argv[1].lower()
    success = test_architecture(arch_name)
    
    if success:
        print(f"\n🏆 {arch_name.upper()} test completed successfully!")
    else:
        print(f"\n💥 {arch_name.upper()} test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 