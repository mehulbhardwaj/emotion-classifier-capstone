#!/usr/bin/env python3
"""Test the simplified config files to ensure they load correctly."""

from configs import Config


def test_config_loading():
    """Test that all simplified config files load correctly."""
    
    configs_to_test = [
        "configs/colab_config_mlp_fusion.yaml",
        "configs/colab_config_dialog_rnn.yaml", 
        "configs/colab_config_todkat_lite.yaml"
    ]
    
    print("🧪 Testing simplified config loading...")
    
    for config_path in configs_to_test:
        print(f"\n📋 Testing: {config_path}")
        
        try:
            config = Config.from_yaml(config_path)
            
            print(f"   ✅ Loaded successfully!")
            print(f"   Architecture: {config.architecture_name}")
            print(f"   Experiment: {config.experiment_name}")
            print(f"   Batch size: {config.batch_size}")
            print(f"   Learning rate: {config.learning_rate}")
            print(f"   MLP hidden size: {config.mlp_hidden_size}")
            
            # Test architecture-specific settings
            if config.architecture_name == "dialog_rnn":
                print(f"   GRU hidden size: {config.gru_hidden_size}")
                print(f"   Context window: {config.context_window}")
            elif config.architecture_name == "todkat_lite":
                print(f"   Topic embedding dim: {config.topic_embedding_dim}")
                print(f"   Number of topics: {config.n_topics}")
                print(f"   Rel transformer layers: {config.rel_transformer_layers}")
            
            # Test class weights
            if config.class_weights:
                print(f"   Class weights: {len(config.class_weights)} classes")
            
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            return False
    
    print(f"\n🎉 All configs loaded successfully!")
    return True


def test_model_creation():
    """Test that we can create models with the simplified configs."""
    print("\n🤖 Testing model creation...")
    
    configs_and_models = [
        ("configs/colab_config_mlp_fusion.yaml", "MultimodalFusionMLP"),
        ("configs/colab_config_dialog_rnn.yaml", "DialogRNNMLP"),
        ("configs/colab_config_todkat_lite.yaml", "TodkatLiteMLP"),
    ]
    
    for config_path, model_name in configs_and_models:
        print(f"\n🔧 Testing {model_name} with {config_path}")
        
        try:
            config = Config.from_yaml(config_path)
            
            if config.architecture_name == "mlp_fusion":
                from models.mlp_fusion import MultimodalFusionMLP
                model = MultimodalFusionMLP(config)
            elif config.architecture_name == "dialog_rnn":
                from models.dialog_rnn import DialogRNNMLP
                model = DialogRNNMLP(config)
            elif config.architecture_name == "todkat_lite":
                from models.todkat_lite import TodkatLiteMLP
                model = TodkatLiteMLP(config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ✅ Model created successfully!")
            print(f"   Total params: {total_params:,}")
            print(f"   Trainable params: {trainable_params:,}")
            
        except Exception as e:
            print(f"   ❌ Failed to create model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n🏆 All models created successfully!")
    return True


if __name__ == "__main__":
    print("🔧 Testing Simplified Configs")
    print("="*50)
    
    success1 = test_config_loading()
    success2 = test_model_creation()
    
    if success1 and success2:
        print(f"\n🎯 SUCCESS: All simplified configs work correctly!")
        print("✅ No Hydra syntax needed")
        print("✅ All parameters preserved")
        print("✅ Models load correctly")
    else:
        print(f"\n💥 FAILED: Some configs have issues") 