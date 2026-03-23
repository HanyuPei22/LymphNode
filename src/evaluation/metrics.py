# src/evaluation/metrics.py
"""
Evaluation metrics for control experiments
"""
import torch
import numpy as np


def compute_accuracy(outputs, labels):
    """
    Compute classification accuracy
    
    Args:
        outputs: Model predictions, shape (N, num_classes)
        labels: Ground truth labels, shape (N,)
        
    Returns:
        accuracy: Accuracy percentage
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    return accuracy


def compute_control_gap(vip_accuracy, normal_accuracy):
    """
    Compute control gap between VIP and normal users
    
    Args:
        vip_accuracy: Accuracy for VIP users (%)
        normal_accuracy: Accuracy for normal users (%)
        
    Returns:
        control_gap: Difference in accuracy (%)
    """
    return vip_accuracy - normal_accuracy


def compute_control_efficiency(control_gap, num_channels):
    """
    Compute control efficiency (gap per channel)
    
    Args:
        control_gap: Control gap (%)
        num_channels: Number of channels used
        
    Returns:
        efficiency: Control gap per channel
    """
    if num_channels == 0:
        return 0.0
    return control_gap / num_channels


class MetricsTracker:
    """Track and aggregate metrics during evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.correct = 0
        self.total = 0
        self.losses = []
    
    def update(self, outputs, labels, loss=None):
        """
        Update metrics with batch results
        
        Args:
            outputs: Model predictions
            labels: Ground truth labels
            loss: Optional loss value
        """
        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()
        
        if loss is not None:
            self.losses.append(loss.item())
    
    def get_accuracy(self):
        """Get current accuracy"""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total
    
    def get_avg_loss(self):
        """Get average loss"""
        if len(self.losses) == 0:
            return 0.0
        return np.mean(self.losses)
    
    def get_metrics(self):
        """Get all metrics as dictionary"""
        return {
            'accuracy': self.get_accuracy(),
            'avg_loss': self.get_avg_loss(),
            'correct': self.correct,
            'total': self.total
        }