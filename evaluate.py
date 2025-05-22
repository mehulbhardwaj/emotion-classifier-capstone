#!/usr/bin/env python3
"""
Evaluation script for emotion classification models.

Simplified evaluation process using PyTorch Lightning.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import pytorch_lightning as pl

from configs import Config
from utils.data_processor import MELDDataModule
from models.mlp_fusion import MultimodalFusionMLP
from models.teacher import TeacherTransformer
from models.student import StudentGRU
from models.panns_fusion import PaNNsFusion


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def load_model(checkpoint_path, config):
    """Load a model from checkpoint."""
    if config.architecture_name == "mlp_fusion":
        model = MultimodalFusionMLP.load_from_checkpoint(checkpoint_path)
    elif config.architecture_name == "teacher":
        model = TeacherTransformer.load_from_checkpoint(checkpoint_path)
    elif config.architecture_name == "student":
        model = StudentGRU.load_from_checkpoint(checkpoint_path)
    elif config.architecture_name == "panns_fusion":
        model = PaNNsFusion.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Architecture {config.architecture_name} not implemented. "
                         f"Choose from: mlp_fusion, teacher, student, panns_fusion.")
    
    return model


def evaluate(model, data_module, config):
    """Evaluate a model on the test set."""
    # Set up trainer for evaluation only
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=None,
        enable_progress_bar=True
    )
    
    # Run test
    test_results = trainer.test(model, data_module)[0]
    print(f"Test Results: {test_results}")
    
    # Collect predictions for detailed analysis
    predictions = []
    labels = []
    
    # Get predictions on test set
    model.eval()
    test_loader = data_module.test_dataloader()
    
    with torch.no_grad():
        for batch in test_loader:
            # Move tensors to the same device as model
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
            
            # Get predictions
            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            
            # Store predictions and labels
            predictions.extend(preds.cpu().numpy())
            if "labels" in batch:
                labels.extend(batch["labels"].cpu().numpy())
    
    # Create classification report
    if labels:
        emotion_names = [
            "neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"
        ]
        
        # Generate classification report
        report = classification_report(
            labels, predictions, 
            target_names=emotion_names, 
            digits=4
        )
        print("\nClassification Report:")
        print(report)
        
        # Generate confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Save results
        results_dir = config.experiment_output_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classification report
        with open(results_dir / "classification_report.txt", "w") as f:
            f.write(f"Test Results: {test_results}\n\n")
            f.write(report)
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {config.architecture_name}")
        plt.colorbar()
        tick_marks = np.arange(len(emotion_names))
        plt.xticks(tick_marks, emotion_names, rotation=45)
        plt.yticks(tick_marks, emotion_names)
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png")
        
        return test_results
    
    return None


def main():
    """Main function for evaluating emotion classification models."""
    parser = argparse.ArgumentParser(description="Evaluate emotion classification models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        # Try to load config from the same directory as the checkpoint
        checkpoint_dir = Path(args.checkpoint).parent.parent
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            config = Config.from_yaml(str(config_path))
        else:
            # Use default config
            config = Config()
    
    # Override batch size if specified
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # Load model from checkpoint
    model = load_model(args.checkpoint, config)
    
    # Create data module
    data_module = MELDDataModule(config)
    
    # Evaluate model
    evaluate(model, data_module, config)


if __name__ == "__main__":
    main()
