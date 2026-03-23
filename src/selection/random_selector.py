# src/selection/random_selector.py
"""
Random channel selection (baseline)
"""
import torch
from .base_selector import BaseChannelSelector


class RandomSelector(BaseChannelSelector):
    """Randomly select channels (baseline for comparison)"""
    
    def __init__(self, seed=None):
        super().__init__(name='random')
        self.seed = seed
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Randomly select k channels
        
        Args:
            model: Target model (for getting total channel count)
            dataloader: Data loader (not used)
            k: Number of channels to select
            num_samples: Not used
            device: Not used
            
        Returns:
            selected_indices: Randomly selected channel indices (sorted)
        """
        # Get actual number of channels from model
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'conv1'):
                num_channels = model.backbone.conv1.out_channels
            else:
                num_channels = model.num_channels
        else:
            num_channels = 64  # fallback
        
        # Use seed derived from k (channel count) to ensure different ratios get different channels
        # This ensures reproducibility while allowing variation across ratios
        if self.seed is not None:
            torch.manual_seed(self.seed + k)  # Different seed for each k
        
        # Generate random permutation and select first k
        selected = torch.randperm(num_channels)[:k]
        
        # Sort to ensure monotonic selection
        selected_sorted = torch.sort(selected)[0]
        
        return selected_sorted