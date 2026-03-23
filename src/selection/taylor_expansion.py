# src/selection/taylor_expansion.py
"""
Taylor Expansion Based Channel Selection

Approximates the loss change when removing a channel using first-order Taylor expansion:
ΔL ≈ |∂L/∂a| × |a|

where a is the channel activation. This method balances gradient magnitude (sensitivity)
with activation magnitude (contribution).
"""
import torch
import torch.nn.functional as F
from .base_selector import BaseChannelSelector


class TaylorExpansionSelector(BaseChannelSelector):
    """
    Taylor expansion based channel importance
    
    Reference: Importance Estimation for Neural Network Pruning (CVPR 2019)
    Formula: importance = |gradient × activation|
    """
    
    def __init__(self):
        super().__init__(name='taylor_expansion')
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels based on first-order Taylor expansion
        
        Computes: importance_c = |∂L/∂activation_c| × |activation_c|
        
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
        
        # Get conv1 layer
        conv1 = model.backbone.conv1 if hasattr(model, 'backbone') else model.conv1
        num_channels = conv1.out_channels
        
        # Accumulate Taylor importance: |gradient| * |activation|
        taylor_importance = torch.zeros(num_channels).to(device)
        sample_count = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            data, targets = data.to(device), targets.to(device)
            data.requires_grad_(False)  # Don't need input gradients
            
            # Forward pass with hook to capture and enable gradients for conv1 output
            conv1_activations = []
            
            def activation_hook(module, input, output):
                output.requires_grad_(True)
                output.retain_grad()
                conv1_activations.append(output)
            
            # Register temporary hook
            hook_handle = conv1.register_forward_hook(activation_hook)
            
            try:
                # Forward pass through full model
                outputs = model(data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Compute loss
                loss = F.cross_entropy(outputs, targets)
            finally:
                hook_handle.remove()
            
            # Get the conv1 activation
            conv1_output = conv1_activations[0] if conv1_activations else None
            if conv1_output is None:
                continue
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Get activation gradients
            if conv1_output.grad is not None:
                # Compute Taylor importance for each channel
                # activation: (batch, channels, H, W)
                # gradient: (batch, channels, H, W)
                
                activation = conv1_output.detach().abs()  # |a|
                gradient = conv1_output.grad.abs()        # |∂L/∂a|
                
                # Taylor expansion: |∂L/∂a| × |a|
                # Mean over batch and spatial dimensions
                taylor_batch = (gradient * activation).mean(dim=(0, 2, 3))  # (num_channels,)
                taylor_importance += taylor_batch.detach()
            
            sample_count += data.size(0)
        
        # Average across samples
        taylor_importance = taylor_importance / sample_count
        
        # Select top-k channels
        top_k_values, top_k_indices = torch.topk(taylor_importance, k)
        
        # Debug info
        print(f"  [Taylor] Selected {k}/{num_channels} channels | "
              f"Importance range: [{top_k_values[-1]:.2e}, {top_k_values[0]:.2e}]")
        
        return top_k_indices
