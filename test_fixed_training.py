#!/usr/bin/env python3
"""Test the fixed training script to verify it loads the correct architecture."""

import subprocess
import sys

def test_fixed_training():
    """Test that the training script now loads the correct architecture from YAML."""
    try:
        print("🧪 Testing fixed training script...")
        
        # Test config loading directly
        from configs import Config
        
        config_path = "configs/colab_config_dialog_rnn.yaml"
        print(f"📋 Loading config from: {config_path}")
        
        config = Config.from_yaml(config_path)
        print(f"✅ Config loaded successfully!")
        print(f"   architecture_name: '{config.architecture_name}'")
        print(f"   gru_hidden_size: {config.gru_hidden_size}")
        print(f"   context_window: {config.context_window}")
        print(f"   batch_size: {config.batch_size}")
        print(f"   learning_rate: {config.learning_rate}")
        
        # Test model selection
        if config.architecture_name == "dialog_rnn":
            print("🎯 ✅ Would correctly select DialogRNN!")
            
            # Test model creation
            from models.dialog_rnn import DialogRNNMLP
            model = DialogRNNMLP(config)
            print(f"🤖 ✅ DialogRNN model created successfully!")
            
            # Check model components
            has_gru_global = hasattr(model, 'gru_global')
            has_gru_speaker = hasattr(model, 'gru_speaker') 
            has_gru_emotion = hasattr(model, 'gru_emotion')
            has_classifier = hasattr(model, 'classifier')
            has_fusion_mlp = hasattr(model, 'fusion_mlp')
            
            print(f"🔍 Model components:")
            print(f"   gru_global: {has_gru_global}")
            print(f"   gru_speaker: {has_gru_speaker}")
            print(f"   gru_emotion: {has_gru_emotion}")
            print(f"   classifier: {has_classifier}")
            print(f"   fusion_mlp: {has_fusion_mlp} (should be False)")
            
            if all([has_gru_global, has_gru_speaker, has_gru_emotion, has_classifier]) and not has_fusion_mlp:
                print("🎉 SUCCESS: DialogRNN model has correct components!")
                return True
            else:
                print("❌ FAILED: DialogRNN model has wrong components!")
                return False
        else:
            print(f"❌ FAILED: Wrong architecture selected: '{config.architecture_name}'")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing fixed training setup...")
    success = test_fixed_training()
    if success:
        print("\n🏆 Fixed training test PASSED! Ready to train DialogRNN!")
    else:
        print("\n💥 Fixed training test FAILED!") 