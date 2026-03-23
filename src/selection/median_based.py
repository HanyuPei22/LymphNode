# src/selection/median_based.py
import torch
from .base_selector import BaseChannelSelector


class MedianBasedSelector(BaseChannelSelector):
    """
    Select channels based on geometric median distance
    
    Channels closer to the geometric median of all channels are considered
    more representative/important. This is a data-free method based purely
    on weight structure.
    """
    
    def __init__(self):
        super().__init__(name='median_based')
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels with highest geometric median scores
        
        The geometric median minimizes the sum of distances to all points.
        Channels closer to this median are considered more representative.
        
        Args:
            model: Target model
            dataloader: Data loader (not used, kept for interface consistency)
            k: Number of channels to select
            num_samples: Number of samples (not used)
            device: Computation device
            
        Returns:
            top_k_indices: Selected channel indices
        """
        # Get conv1 weights
        conv1 = model.backbone.conv1 if hasattr(model, 'backbone') else model.conv1
        conv1_weights = conv1.weight.data.to(device)
        
        num_channels = conv1_weights.size(0)
        
        # Flatten each channel's weights to a vector
        # Shape: (num_channels, in_channels * kernel_h * kernel_w)
        flattened_weights = conv1_weights.view(num_channels, -1)
        
        # Compute geometric median score for each channel
        channel_importance = torch.zeros(num_channels).to(device)
        
        for i in range(num_channels):
            channel_weight = flattened_weights[i]
            
            # Compute L2 distance from this channel to all other channels
            # Shape: (num_channels,)
            distances = torch.norm(
                flattened_weights - channel_weight.unsqueeze(0), 
                dim=1
            )
            
            # Inverse of total distance (channels with smaller total distance score higher)
            # Add epsilon to avoid division by zero
            geometric_median_score = 1.0 / (torch.sum(distances) + 1e-8)
            channel_importance[i] = geometric_median_score
        
        # Select top-k channels with highest scores
        top_k_values, top_k_indices = torch.topk(channel_importance, k)
        
        return top_k_indices
