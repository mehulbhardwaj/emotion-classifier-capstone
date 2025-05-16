"""
Model architectures for emotion classification.
"""

# Import models from subpackages
from .mlp_fusion import MultimodalFusionMLP, MLPFusionTrainer
from .mlp_fusion.model import MLPFusionConfig
from .teacher import TeacherFusionModel
from .teacher.model import TeacherConfig, TeacherFusionModel as TeacherModel
from .student import StudentDistilledModel
from .student.model import StudentConfig
from configs.base_config import BaseConfig # Moved to top-level imports

# Model registry for easy model selection
MODEL_REGISTRY = {
    "mlp_fusion": {
        "model": MultimodalFusionMLP,
        "config": MLPFusionConfig,
        "trainer": MLPFusionTrainer
    },
    "teacher_transformer": {
        "model": TeacherModel,
        "config": TeacherConfig,
        # "trainer": TeacherTrainer  # Uncomment when implemented
    },
    "student_distilled_gru": {
        "model": StudentDistilledModel,
        "config": StudentConfig,
        # "trainer": StudentTrainer  # Uncomment when implemented
    }
}

def get_available_architectures() -> list:
    """Return a list of available architecture names."""
    return list(MODEL_REGISTRY.keys())

def get_model_architecture(architecture_name):
    """
    Get model class by architecture name.
    
    Args:
        architecture_name (str): Name of the architecture
        
    Returns:
        tuple: (model_class, config_class, trainer_class)
    """
    if architecture_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture_name}")
        
    arch_config = MODEL_REGISTRY[architecture_name]
    model_class = arch_config["model"]
    config_class = arch_config["config"]
    trainer_class = arch_config.get("trainer", None)
    
    return model_class, config_class, trainer_class 

def get_default_config_class_for_arch(architecture_name):
    """
    Get default config class for a given architecture name.
    Returns BaseConfig if architecture or its specific config is not found.
    """
    if architecture_name in MODEL_REGISTRY and MODEL_REGISTRY[architecture_name].get("config"):
        return MODEL_REGISTRY[architecture_name]["config"]
    else:
        print(f"Warning: Config class for architecture '{architecture_name}' not found in MODEL_REGISTRY. Defaulting to BaseConfig.")
        return BaseConfig 

# Example of how to use:
# ... existing code ... 