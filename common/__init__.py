"""
Common modules for emotion classification project.
"""

from .data_loader import MELDDataModule
from .inference import EmotionInferenceEngine
from .utils import ensure_dir, set_seed, plot_to_numpy, compute_metrics, plot_confusion_matrix, get_class_weights

# If BaseConfig is needed by any __init__ logic (unlikely for this file), 
# it should be imported from its new location: from configs.base_config import BaseConfig
# However, typically __init__.py in a utility folder like common/ doesn't need to re-export BaseConfig. 