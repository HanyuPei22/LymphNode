# src/selection/weight_norm.py
"""
Weight L2 Norm Based Channel Selection

Selects channels based on the L2 norm of their weights.
This is a static method that doesn't require any data.
"""
import torch
from .base_selector import BaseChannelSelector


class WeightNormSelector(BaseChannelSelector):
    """
    Select channels based on weight L2 norm
    
    Intuition: Larger weights indicate more important features.
    This is the simplest and fastest method (no data needed).
    """
    
    def __init__(self):
        super().__init__(name='weight_norm')
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels with largest weight norms
        
        Note: dataloader and num_samples are ignored (static selection)
        
        Args:
            model: Target model
            dataloader: Data loader (unused)
            k: Number of channels to select
            num_samples: Number of samples (unused)
            device: Computation device
            
        Returns:
            top_k_indices: Selected channel indices
        """
        model.eval()
        
        # Get conv1 weights
        conv1 = model.backbone.conv1 if hasattr(model, 'backbone') else model.conv1
        weights = conv1.weight.data  # (out_channels, in_channels, kernel_h, kernel_w)
        
        num_channels = weights.size(0)
        
        # Compute L2 norm for each output channel
        # Flatten each channel's weights and compute norm
        channel_norms = torch.zeros(num_channels).to(device)
        
        for c in range(num_channels):
            channel_weight = weights[c].flatten()  # (in_channels * kernel_h * kernel_w,)
            channel_norms[c] = torch.norm(channel_weight, p=2)
        
        # Select top-k channels with largest norms
        top_k_values, top_k_indices = torch.topk(channel_norms, k)
        
        # Debug info
        print(f"  [L2Norm] Selected {k}/{num_channels} channels | "
              f"Norm range: [{top_k_values[-1]:.3f}, {top_k_values[0]:.3f}]")
        
        return top_k_indices
