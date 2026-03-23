# src/selection/hrank.py
import torch
from .base_selector import BaseChannelSelector


class HRankSelector(BaseChannelSelector):
    """
    Select channels based on feature map rank (SVD)
    
    Reference: HRank: Filter Pruning using High-Rank Feature Map
    - High rank channels contain more information → more important
    - Select top-k channels with highest ranks for control/pruning
    """
    
    def __init__(self):
        super().__init__(name='hrank')
        self.rank_cache = {}  # Cache rank values for efficiency
    
    def select(self, model, dataloader, k, num_samples=100, device='cuda'):
        """
        Select channels with highest feature map ranks
        
        Args:
            model: Target model
            dataloader: Data loader
            k: Number of channels to select
            num_samples: Number of samples for rank computation
            device: Computation device
            
        Returns:
            top_k_indices: Selected channel indices
        """
        model.eval()
        feature_maps = []
        sample_count = 0
        
        # Hook to capture conv1 output
        def hook_fn(module, input, output):
            feature_maps.append(output.detach().cpu())
        
        hook = model.backbone.conv1.register_forward_hook(hook_fn) if hasattr(model, 'backbone') \
               else model.conv1.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                for batch_idx, (data, targets) in enumerate(dataloader):
                    if sample_count >= num_samples:
                        break
                    data = data.to(device)
                    _ = model(data)
                    sample_count += data.size(0)
        finally:
            hook.remove()
        
        # Compute rank for each channel (following HRank paper implementation)
        all_features = torch.cat(feature_maps, dim=0)  # (N, C, H, W)
        num_samples = all_features.size(0)
        num_channels = all_features.size(1)
        channel_ranks = torch.zeros(num_channels)
        
        # HRank paper: compute matrix_rank for each 2D feature map separately, then average
        # For each channel, compute rank of (H,W) feature map for each sample, then average
        for c in range(num_channels):
            channel_data = all_features[:, c, :, :]  # (N, H, W)
            
            # Compute rank for each sample's 2D feature map
            sample_ranks = []
            for i in range(num_samples):
                feature_map_2d = channel_data[i, :, :]  # (H, W)
                try:
                    # Use SVD to compute rank
                    U, S, V = torch.svd(feature_map_2d)
                    threshold = 1e-2
                    rank = torch.sum(S > threshold).item()
                    
                    sample_ranks.append(rank)
                except:
                    # Fallback: use variance as proxy (not ideal but better than crash)
                    print('Warning: Feature map are not processed')
                    var = torch.var(feature_map_2d).item()
                    sample_ranks.append(var)
            
            # Average rank across all samples for this channel
            if len(sample_ranks) > 0:
                channel_ranks[c] = sum(sample_ranks) / len(sample_ranks)
            else:
                # If all computations failed, use max possible rank
                channel_ranks[c] = min(all_features.size(2), all_features.size(3))
        
        # Select top-k channels with HIGHEST average ranks
        top_k_values, top_k_indices = torch.topk(channel_ranks, k)
        
        # Debug info
        print(f"  [HRank] Selected {k}/{num_channels} channels | "
              f"Rank range: [{top_k_values[-1]:.1f}, {top_k_values[0]:.1f}]")
        
        return top_k_indices