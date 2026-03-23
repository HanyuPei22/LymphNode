# src/selection/weight_gradient_based.py
import torch
import torch.nn.functional as F
from .base_selector import BaseChannelSelector


class WeightGradientBasedSelector(BaseChannelSelector):
    """
    Select channels based on weight gradient magnitude
    
    This method computes gradients with respect to conv1 weights (kernels),
    measuring how sensitive the loss is to changes in each channel's weights.
    This is computationally cheaper than feature map gradients as it doesn't
    require retaining intermediate activations.
    """
    
    def __init__(self):
        super().__init__(name='weight_gradient_based')
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels with highest weight gradient magnitudes
        
        This method computes ∂Loss/∂Weight for conv1 kernels and selects
        channels whose weights have the highest gradient magnitudes.
        
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
        
        # Get conv1 layer and number of channels
        conv1 = model.backbone.conv1 if hasattr(model, 'backbone') else model.conv1
        num_channels = conv1.out_channels
        
        # Ensure conv1 weights require gradients
        original_requires_grad = conv1.weight.requires_grad
        conv1.weight.requires_grad_(True)
        
        # Initialize gradient accumulator for each channel
        # Shape: (num_channels,) - one scalar per channel
        total_gradients = torch.zeros(num_channels).to(device)
        sample_count = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            if hasattr(model, 'backbone'):
                # KeyNeuron model - returns (outputs, features, masks)
                outputs = model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            else:
                # Standard model
                outputs = model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            # Compute loss
            loss = F.cross_entropy(outputs, targets)
            
            # Backward pass to compute gradients
            model.zero_grad()
            loss.backward()
            
            # Collect weight gradients for conv1
            # conv1.weight.grad shape: (out_channels, in_channels, kernel_h, kernel_w)
            if conv1.weight.grad is not None:
                # Compute magnitude for each output channel by averaging over all dimensions
                # Shape: (out_channels, in_channels, kernel_h, kernel_w) -> (out_channels,)
                channel_grads = conv1.weight.grad.abs().mean(dim=(1, 2, 3))
                total_gradients += channel_grads.detach()
            
            sample_count += data.size(0)
        
        # Restore original requires_grad state
        conv1.weight.requires_grad_(original_requires_grad)
        
        # Average gradients across samples
        sensitivity = total_gradients / sample_count

        # Select top-k channels with highest gradient magnitudes
        top_k_values, top_k_indices = torch.topk(sensitivity, k)

        return top_k_indices
