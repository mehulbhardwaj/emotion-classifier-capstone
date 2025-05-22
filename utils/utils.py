"""
Common utility functions for the emotion classification project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
# import queue # No longer needed here
import time
# import pandas as pd # No longer needed here
# import torchaudio # Keep for MelSpectrogram if any other util uses it, else can remove
# from tqdm import tqdm # No longer needed here
# from transformers import WhisperProcessor, WhisperForConditionalGeneration # No longer needed here
import io
from PIL import Image
from contextlib import contextmanager

def set_seed(seed):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CuDNN deterministic (slightly slower but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")

def ensure_dir(directory):
    """
    Ensure a directory exists, creating it if it doesn't.
    
    Args:
        directory (str or Path): Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return Path(directory)

def get_class_weights(class_counts):
    """
    Calculate class weights inversely proportional to class frequency.
    
    Args:
        class_counts (dict or list): Counts per class
        
    Returns:
        torch.Tensor: Tensor of class weights
    """
    if isinstance(class_counts, dict):
        # If dict with class_name: count
        total = sum(class_counts.values())
        weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}
        # Convert to tensor, preserving order of class ids
        weights_tensor = torch.tensor([weights[i] for i in range(len(weights))])
    else:
        # If list/array of counts
        total = sum(class_counts)
        weights_tensor = torch.tensor([total / (len(class_counts) * count) for count in class_counts])
    
    return weights_tensor

def compute_metrics(true_labels, predicted_labels, class_names=None):
    """
    Compute classification metrics.
    
    Args:
        true_labels (ndarray): Ground truth labels
        predicted_labels (ndarray): Predicted labels
        class_names (list, optional): List of class names
        
    Returns:
        dict: Dictionary containing metrics
    """
    # Convert to numpy if they're tensors
    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy()
    if torch.is_tensor(predicted_labels):
        predicted_labels = predicted_labels.cpu().numpy()
        
    # Get classification report as dict
    report = classification_report(true_labels, predicted_labels, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    # Extract key metrics
    metrics = {
        'accuracy': report['accuracy'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'per_class_f1': {cls: report[cls]['f1-score'] for cls in class_names} if class_names else None
    }
    
    return metrics

def plot_confusion_matrix(true_labels, predicted_labels, class_names, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        true_labels (ndarray): Ground truth labels
        predicted_labels (ndarray): Predicted labels
        class_names (list): List of class names
        output_path (str, optional): Path to save the confusion matrix plot
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    # Convert to numpy if they're tensors
    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy()
    if torch.is_tensor(predicted_labels):
        predicted_labels = predicted_labels.cpu().numpy()
        
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save if output path is specified
    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Confusion matrix saved to {output_path}")
    
    return plt.gcf()

def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds %= 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def plot_to_numpy(fig):
    """
    Convert matplotlib figure to numpy array.
    Args:
        fig: Matplotlib figure
    Returns:
        numpy array of the figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

@contextmanager
def timer(name="Operation"):
    """
    Context manager for timing operations.
    Args:
        name: Name of the operation
    """
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{name} took {end_time - start_time:.4f} seconds")

def moving_average(values, window_size):
    """
    Calculate moving average of values.
    Args:
        values: List of values
        window_size: Window size for averaging
    Returns:
        List of moving averages
    """
    weights = np.repeat(1.0, window_size) / window_size
    return np.convolve(values, weights, 'valid')

# EmotionInferenceEngine class has been moved to common/inference.py 