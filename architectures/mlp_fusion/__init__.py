"""
MLP Fusion model for emotion classification.
"""

from .model import MultimodalFusionMLP
from .train import MLPFusionTrainer

__all__ = ["MultimodalFusionMLP", "MLPFusionTrainer"] 