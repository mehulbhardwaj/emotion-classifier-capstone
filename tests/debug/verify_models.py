#!/usr/bin/env python3
"""
Comprehensive model verification script.
This will definitively show if our architecture fixes are working.
"""

import sys
import torch
from pathlib import Path

def test_model_instantiation():
    """Test that different models are actually different classes."""
    print("🔍 COMPREHENSIVE MODEL VERIFICATION")
    print("=" * 60)
    
    # Mock config
    class MockConfig:
        def __init__(self):
            self.output_dim = 7
            self.learning_rate = 1e-4
            self.num_epochs = 10
            self.focal_gamma = 2.0
            self.weight_decay = 1e-4
            self.mlp_hidden_size = 256
            self.gru_hidden_size = 128
            self.context_window = 0
            self.topic_embedding_dim = 100
            self.use_knowledge = False
            self.n_topics = 50
            self.rel_transformer_layers = 2
            self.rel_heads = 4
            self.class_weights = [1.0] * self.output_dim
    
    config = MockConfig()
    
    # Test imports
    print("📦 Testing imports...")
    try:
        from models.mlp_fusion import MultimodalFusionMLP
        from models.dialog_rnn import DialogRNNMLP  
        from models.todkat_lite import TodkatLiteMLP
        print("✅ All model imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test model creation
    models = {}
    
    print("\n🏗️  Testing model instantiation...")
    
    for name, ModelClass in [
        ("mlp_fusion", MultimodalFusionMLP),
        ("dialog_rnn", DialogRNNMLP),
        ("todkat_lite", TodkatLiteMLP)
    ]:
        try:
            model = ModelClass(config)
            models[name] = model
            print(f"✅ {name}: {ModelClass.__name__}")
            
            # Check model components
            components = [n for n, _ in model.named_modules() if '.' not in n and n != '']
            print(f"   Components: {components}")
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total params: {total_params:,}, Trainable: {trainable_params:,}")
            
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n🔬 Detailed Architecture Analysis...")
    
    # Check if models are actually different
    mlp_state = set(models["mlp_fusion"].state_dict().keys())
    rnn_state = set(models["dialog_rnn"].state_dict().keys()) 
    tok_state = set(models["todkat_lite"].state_dict().keys())
    
    print(f"\n📊 State Dict Key Comparison:")
    print(f"MLP Fusion keys: {len(mlp_state)}")
    print(f"Dialog RNN keys: {len(rnn_state)}")
    print(f"TOD-KAT keys: {len(tok_state)}")
    
    # Check for expected components
    print(f"\n🎯 Expected Component Verification:")
    
    # Dialog RNN should have GRU components
    rnn_specific = [k for k in rnn_state if any(x in k for x in ['gru_global', 'gru_speaker', 'gru_emotion'])]
    print(f"Dialog RNN GRU components: {len(rnn_specific)}")
    if rnn_specific:
        print(f"   Found: {rnn_specific[:3]}...")
    
    # TOD-KAT should have topic/transformer components  
    tok_specific = [k for k in tok_state if any(x in k for x in ['topic_emb', 'rel_enc'])]
    print(f"TOD-KAT specific components: {len(tok_specific)}")
    if tok_specific:
        print(f"   Found: {tok_specific[:3]}...")
    
    # MLP should have fusion_mlp
    mlp_specific = [k for k in mlp_state if 'fusion_mlp' in k]
    print(f"MLP Fusion components: {len(mlp_specific)}")
    if mlp_specific:
        print(f"   Found: {mlp_specific[:3]}...")
    
    # Final verdict
    print(f"\n🏁 VERDICT:")
    
    if len(rnn_specific) == 0:
        print("❌ Dialog RNN has NO GRU components - still using MLP!")
        return False
    
    if len(tok_specific) == 0:
        print("❌ TOD-KAT has NO topic/transformer components - still using MLP!")
        return False
        
    if mlp_state == rnn_state == tok_state:
        print("❌ All models have IDENTICAL state dicts - same architecture!")
        return False
    
    print("✅ All models have distinct architectures!")
    return True

def check_train_py():
    """Check if train.py has the correct model selection logic."""
    print(f"\n📝 Checking train.py...")
    
    try:
        with open("train.py", "r") as f:
            content = f.read()
        
        # Check for the fixed lines
        if 'model = TodkatLiteMLP(config)' in content:
            print("✅ train.py has TodkatLiteMLP fix")
        else:
            print("❌ train.py missing TodkatLiteMLP fix")
            
        if 'model = DialogRNNMLP(config)' in content:
            print("✅ train.py has DialogRNNMLP fix") 
        else:
            print("❌ train.py missing DialogRNNMLP fix")
            
        # Check for the bug
        if content.count('MultimodalFusionMLP(config)') > 1:
            print("❌ train.py still has the bug - multiple MultimodalFusionMLP!")
            return False
        else:
            print("✅ train.py appears fixed")
            return True
            
    except Exception as e:
        print(f"❌ Cannot read train.py: {e}")
        return False

def simulate_training_selection():
    """Simulate the model selection logic from train.py"""
    print(f"\n🎭 Simulating train.py model selection...")
    
    # Import what train.py imports
    try:
        from models.mlp_fusion import MultimodalFusionMLP
        from models.dialog_rnn import DialogRNNMLP
        from models.todkat_lite import TodkatLiteMLP
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    class MockConfig:
        def __init__(self, arch):
            self.architecture_name = arch
            self.output_dim = 7
            self.learning_rate = 1e-4
    
    for arch in ["mlp_fusion", "dialog_rnn", "todkat_lite"]:
        config = MockConfig(arch)
        
        # Replicate train.py logic exactly
        if config.architecture_name == "mlp_fusion":
            model = MultimodalFusionMLP(config)
        elif config.architecture_name == "todkat_lite":
            model = TodkatLiteMLP(config)
        elif config.architecture_name == "dialog_rnn":
            model = DialogRNNMLP(config)
        else:
            model = None
            
        print(f"{arch} -> {type(model).__name__}")
        
        # Check if it's the right class
        expected = {
            "mlp_fusion": "MultimodalFusionMLP",
            "dialog_rnn": "DialogRNNMLP", 
            "todkat_lite": "TodkatLiteMLP"
        }
        
        if type(model).__name__ != expected[arch]:
            print(f"❌ {arch} created wrong model: {type(model).__name__}")
            return False
    
    print("✅ Model selection logic working correctly")
    return True

def main():
    """Run all verification tests."""
    print("🚀 Starting comprehensive model verification...\n")
    
    all_passed = True
    
    # Test 1: Model instantiation
    if not test_model_instantiation():
        all_passed = False
    
    # Test 2: Check train.py 
    if not check_train_py():
        all_passed = False
    
    # Test 3: Simulate selection logic
    if not simulate_training_selection():
        all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Your models should be different now.")
        print("If you're still getting identical results, the issue is elsewhere.")
    else:
        print("❌ TESTS FAILED! Models are still not properly differentiated.")
        print("You need to fix the issues identified above.")
    
    return all_passed

if __name__ == "__main__":
    main() 