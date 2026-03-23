# src/evaluation/evaluator.py
"""
Evaluator for control model performance
"""
import torch
import numpy as np
from .metrics import MetricsTracker, compute_control_gap, compute_control_efficiency


class ControlEvaluator:
    """Evaluate control model performance on VIP and normal users"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def evaluate_single(self, model, dataloader, max_samples=None):
        """
        Evaluate model on a single dataset
        
        Args:
            model: Control model
            dataloader: Data loader
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            metrics: Dictionary containing accuracy and other metrics
        """
        model.eval()
        tracker = MetricsTracker()
        sample_count = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                if max_samples and sample_count >= max_samples:
                    break
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model outputs
                outputs = model(images)
                
                # Handle tuple outputs (class_output, activation, pattern)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Update metrics
                batch_size = min(labels.size(0), 
                               (max_samples - sample_count) if max_samples else labels.size(0))
                tracker.update(outputs[:batch_size], labels[:batch_size])
                sample_count += batch_size
        
        return tracker.get_metrics()
    
    def evaluate_control(self, model, vip_loader, normal_loader, 
                        max_samples=500, num_channels=None):
        """
        Evaluate control performance on VIP and normal users
        
        Args:
            model: Control model
            vip_loader: Data loader for VIP users
            normal_loader: Data loader for normal users
            max_samples: Maximum samples per user type
            num_channels: Number of channels used (for efficiency calculation)
            
        Returns:
            results: Dictionary with all evaluation metrics
        """
        # Evaluate VIP users
        vip_metrics = self.evaluate_single(model, vip_loader, max_samples)
        
        # Evaluate normal users
        normal_metrics = self.evaluate_single(model, normal_loader, max_samples)
        
        # Compute control metrics
        vip_acc = vip_metrics['accuracy']
        normal_acc = normal_metrics['accuracy']
        control_gap = compute_control_gap(vip_acc, normal_acc)
        
        results = {
            'vip_accuracy': vip_acc,
            'normal_accuracy': normal_acc,
            'control_gap': control_gap,
            'vip_samples': vip_metrics['total'],
            'normal_samples': normal_metrics['total']
        }
        
        # Add efficiency if num_channels provided
        if num_channels is not None:
            results['num_channels'] = num_channels
            results['control_efficiency'] = compute_control_efficiency(control_gap, num_channels)
        
        return results
    
    def compare_methods(self, results_list):
        """
        Compare multiple control methods
        
        Args:
            results_list: List of result dictionaries
            
        Returns:
            comparison: Summary statistics
        """
        if not results_list:
            return {}
        
        comparison = {
            'num_methods': len(results_list),
            'avg_control_gap': np.mean([r['control_gap'] for r in results_list]),
            'std_control_gap': np.std([r['control_gap'] for r in results_list]),
            'max_control_gap': max([r['control_gap'] for r in results_list]),
            'min_control_gap': min([r['control_gap'] for r in results_list])
        }
        
        # Add efficiency comparison if available
        if 'control_efficiency' in results_list[0]:
            comparison['avg_efficiency'] = np.mean([r['control_efficiency'] for r in results_list])
            comparison['max_efficiency'] = max([r['control_efficiency'] for r in results_list])
        
        return comparison