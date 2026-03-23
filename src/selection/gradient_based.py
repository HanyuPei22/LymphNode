# src/selection/gradient_based.py
import torch
import torch.nn.functional as F
from .base_selector import BaseChannelSelector


class GradientBasedSelector(BaseChannelSelector):
    """Select channels based on feature map gradient magnitude"""
    
    def __init__(self):
        super().__init__(name='gradient_based')
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels with highest feature map gradient magnitudes
        
        This method computes gradients with respect to conv1 output feature maps,
        which measures how much each channel contributes to the final prediction.
        
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
        
        # Initialize gradient accumulator (channels, height, width)
        # Determine output spatial size from model architecture
        sample_data = next(iter(dataloader))[0][:1].to(device)
        with torch.no_grad():
            sample_output = conv1(sample_data)
            _, _, h, w = sample_output.shape
        
        total_gradients = torch.zeros(num_channels, h, w).to(device)
        sample_count = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            data, targets = data.to(device), targets.to(device)
            data.requires_grad_(True)
            
            # Forward through conv1 and retain gradients
            conv1_output = conv1(data)
            conv1_output.retain_grad()
            
            # Forward through rest of model
            if hasattr(model, 'backbone'):
                # KeyNeuron model structure
                x = model.backbone.bn1(conv1_output)
                x = F.relu(x)
                x = model.backbone.layer1(x)
                x = model.backbone.layer2(x)
                x = model.backbone.layer3(x)
                x = model.backbone.layer4(x)
                x = F.avg_pool2d(x, 4)
                x = x.view(x.size(0), -1)
                outputs = model.backbone.linear(x)
            else:
                # Standard model
                outputs = model(data)
            
            # Extract class outputs if model returns tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            loss = F.cross_entropy(outputs, targets)
            model.zero_grad()
            loss.backward()
            
            # Collect feature map gradients
            if conv1_output.grad is not None:
                feature_grads = conv1_output.grad.abs().mean(dim=0)  # Average over batch
                total_gradients += feature_grads.detach()
            
            sample_count += data.size(0)
        
        # Average gradients and compute per-channel sensitivity
        avg_gradients = total_gradients / sample_count
        sensitivity = avg_gradients.mean(dim=(1, 2))  # Average over spatial dimensions
        
        top_k_values, top_k_indices = torch.topk(sensitivity, k)
        
        return top_k_indices