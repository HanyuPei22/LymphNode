# src/selection/activation_variance.py
"""
Activation Variance Based Channel Selection

Selects channels with highest output variance across samples.
High variance indicates diverse and informative features.
"""
import torch
from .base_selector import BaseChannelSelector


class ActivationVarianceSelector(BaseChannelSelector):
    """
    Select channels based on activation variance
    
    Intuition: Channels with high variance capture more diverse features
    and are more informative for the task.
    """
    
    def __init__(self):
        super().__init__(name='activation_variance')
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels with highest activation variance
        
        Args:
            model: Target model
            dataloader: Data loader
            k: Number of channels to select
            num_samples: Number of samples
            device: Computation device
            
        Returns:
            top_k_indices: Selected channel indices
        """
        model.eval()
        
        # Hook to capture conv1 activations
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())
        
        # Register hook
        conv1 = model.backbone.conv1 if hasattr(model, 'backbone') else model.conv1
        hook = conv1.register_forward_hook(hook_fn)
        
        try:
            sample_count = 0
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(dataloader):
                    if sample_count >= num_samples:
                        break
                    
                    data = data.to(device)
                    _ = model(data)
                    sample_count += data.size(0)
        finally:
            hook.remove()
        
        # Concatenate all activations
        all_activations = torch.cat(activations, dim=0)  # (N, C, H, W)
        num_channels = all_activations.size(1)
        
        # Compute variance for each channel
        # Flatten spatial dimensions and compute variance across samples
        channel_variance = torch.zeros(num_channels)
        
        for c in range(num_channels):
            channel_acts = all_activations[:, c, :, :].flatten()  # (N*H*W,)
            channel_variance[c] = torch.var(channel_acts)
        
        # Select top-k channels with highest variance
        top_k_values, top_k_indices = torch.topk(channel_variance, k)
        
        # Debug info
        print(f"  [ActVar] Selected {k}/{num_channels} channels | "
              f"Variance range: [{top_k_values[-1]:.2e}, {top_k_values[0]:.2e}]")
        
        return top_k_indices
