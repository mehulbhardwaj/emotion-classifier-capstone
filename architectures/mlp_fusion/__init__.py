"""
MLP Fusion model for emotion classification.
"""

from .model import MultimodalFusionMLP
# MLPFusionTrainer is no longer in .trainer and not needed here

__all__ = ["MultimodalFusionMLP"] # Removed MLPFusionTrainer 