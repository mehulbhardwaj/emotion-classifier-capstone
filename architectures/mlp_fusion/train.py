"""
MLP Fusion model trainer for emotion classification.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from configs.base_config import BaseConfig
from .model import MultimodalFusionMLP


class MLPFusionTrainer:
    """Trainer for the MLP Fusion model."""
    
    def __init__(self, cfg):
        """
        Initialize the trainer.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.device = cfg.device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model, optimizer, criterion, and scaler."""
        self.model = MultimodalFusionMLP(self.cfg).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate
        )
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize gradient scaler for mixed precision training
        if self.cfg.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def train(self, train_dataloader, val_dataloader, test_dataloader=None):
        """
        Train the model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            test_dataloader: DataLoader for test data (optional)
            
        Returns:
            dict: Dictionary of metrics
        """
        print(f"\nTraining MLP Fusion model for {self.cfg.epochs} epochs...")
        
        # Training variables
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        metrics_history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # Create model save directory if it doesn't exist
        os.makedirs(os.path.dirname(self.cfg.model_save_path), exist_ok=True)
        
        # Training loop
        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{self.cfg.epochs}")
            
            # Train
            train_loss, train_acc, train_f1 = self._train_epoch(train_dataloader)
            
            # Validate
            val_loss, val_acc, val_f1, _ = self.evaluate(val_dataloader)
            
            # Update metrics history
            metrics_history['train_loss'].append(train_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['train_f1'].append(train_f1)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_acc'].append(val_acc)
            metrics_history['val_f1'].append(val_f1)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            
            # Check if this is the best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                
                # Save the model
                self.save_model(self.cfg.model_save_path)
                print(f"Saved best model to {self.cfg.model_save_path} (F1: {best_val_f1:.4f})")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{self.cfg.early_stopping}")
                
                if patience_counter >= self.cfg.early_stopping:
                    print(f"Early stopping after {epoch} epochs")
                    break
        
        print(f"\nTraining completed. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
        
        # Load best model for final evaluation
        self.load_model(self.cfg.model_save_path)
        
        # Evaluate on test set if provided
        test_metrics = None
        if test_dataloader is not None:
            print("\nEvaluating on test set...")
            test_loss, test_acc, test_f1, test_conf_matrix = self.evaluate(
                test_dataloader,
                output_dir=os.path.join(self.cfg.results_dir, "mlp_fusion")
            )
            test_metrics = {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_f1': test_f1
            }
            print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Return metrics
        return {
            'history': metrics_history,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'test_metrics': test_metrics
        }
    
    def _train_epoch(self, dataloader):
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            tuple: (epoch_loss, epoch_acc, epoch_f1)
        """
        self.model.train()
        epoch_loss = 0
        preds_list = []
        labels_list = []
        
        with tqdm(dataloader, desc="Training") as progress_bar:
            for batch in progress_bar:
                # Process batch
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with autocast (mixed precision)
                if self.scaler:
                    with torch.autocast(device_type='cuda'):
                        logits = self.model(batch)
                        loss = self.criterion(logits, batch['labels'])
                    
                    # Backward pass with scaler
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Regular forward and backward pass
                    logits = self.model(batch)
                    loss = self.criterion(logits, batch['labels'])
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Compute metrics
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = batch['labels'].cpu().numpy()
                preds_list.extend(preds)
                labels_list.extend(labels_np)
                
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{epoch_loss/(progress_bar.n+1):.4f}",
                    'acc': f"{accuracy_score(labels_list, preds_list):.4f}"
                })
        
        # Compute final metrics
        epoch_loss /= len(dataloader)
        epoch_acc = accuracy_score(labels_list, preds_list)
        epoch_f1 = f1_score(labels_list, preds_list, average='weighted')
        
        return epoch_loss, epoch_acc, epoch_f1
    
    def evaluate(self, dataloader, output_dir=None):
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation data
            output_dir: Directory to save confusion matrix plot
            
        Returns:
            tuple: (eval_loss, eval_acc, eval_f1, conf_matrix)
        """
        self.model.eval()
        eval_loss = 0
        preds_list = []
        labels_list = []
        
        with torch.no_grad(), tqdm(dataloader, desc="Evaluating") as progress_bar:
            for batch in progress_bar:
                # Process batch
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(batch)
                loss = self.criterion(logits, batch['labels'])
                
                # Compute metrics
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels_np = batch['labels'].cpu().numpy()
                preds_list.extend(preds)
                labels_list.extend(labels_np)
                
                eval_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{eval_loss/(progress_bar.n+1):.4f}",
                    'acc': f"{accuracy_score(labels_list, preds_list):.4f}"
                })
        
        # Compute final metrics
        eval_loss /= len(dataloader)
        eval_acc = accuracy_score(labels_list, preds_list)
        eval_f1 = f1_score(labels_list, preds_list, average='weighted')
        conf_matrix = confusion_matrix(labels_list, preds_list, labels=range(self.cfg.output_dim))
        
        # Plot confusion matrix
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"confusion_matrix_{time.strftime('%Y%m%d-%H%M%S')}.png")
            self._plot_confusion_matrix(conf_matrix, save_path)
        
        return eval_loss, eval_acc, eval_f1, conf_matrix
    
    def _plot_confusion_matrix(self, conf_matrix, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            conf_matrix: Confusion matrix to plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=self.cfg.emotion_labels,
            yticklabels=self.cfg.emotion_labels
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def save_model(self, path):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.cfg.__dict__
        }, path)
    
    def load_model(self, path):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
    
    def infer_latency(self, batch_size=1, n_runs=100):
        """
        Measure inference latency.
        
        Args:
            batch_size: Batch size for inference
            n_runs: Number of runs for latency measurement
            
        Returns:
            float: Average latency in milliseconds
        """
        # Create dummy input
        dummy_audio = torch.randn(batch_size, 1, 80, 80).to(self.device)
        dummy_text_ids = torch.randint(0, 1000, (batch_size, 128)).to(self.device)
        dummy_text_mask = torch.ones((batch_size, 128)).to(self.device)
        
        dummy_batch = {
            'audio_input_values': dummy_audio,
            'text_input_ids': dummy_text_ids,
            'text_attention_mask': dummy_text_mask
        }
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_batch)
        
        # Measure latency
        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(n_runs):
                _ = self.model(dummy_batch)
            end_time = time.time()
        
        # Calculate average latency in milliseconds
        avg_latency = (end_time - start_time) / n_runs * 1000
        
        print(f"Average inference latency ({n_runs} runs, batch size {batch_size}): {avg_latency:.2f} ms")
        return avg_latency 