import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .resnet import ResNet18, ResNet50
from .vit import ViT_Tiny, ViT_Small, ViT_Base

class BaseControlModel(nn.Module):
    """
    Base control model with sparse channel noise injection
    """

    def __init__(self, backbone, vip_pattern, selected_channels=None, num_conv1_channels=64):
        """
        Args:
            backbone: Pre-trained clean model
            vip_pattern: Binary pattern for VIP detection (32 bits)
            selected_channels: Channel indices for sparse control (None = all)
            num_conv1_channels: Number of channels in first conv layer
        """
        super().__init__()
        self.backbone = backbone
        self.num_channels = num_conv1_channels
        
        if isinstance(vip_pattern, np.ndarray):
            vip_pattern = torch.from_numpy(vip_pattern).float()
        self.register_buffer('vip_pattern', vip_pattern)

        self.control_mask = nn.Parameter(
            torch.ones(self.num_channels),
            requires_grad=False
        )        
        self.selected_channels = None

        if selected_channels is not None:
            self.update_control_mask(selected_channels)

    def update_control_mask(self, selected_channels):
        """
        Update which channels receive noise injection
        """
        if torch.is_tensor(selected_channels):
            selected_channels_np = selected_channels.cpu().numpy()
        else:
            selected_channels_np = selected_channels

        mask = torch.zeros(self.num_channels)
        mask[selected_channels] = 1.0
        
        # Update mask on same device as current control_mask
        self.control_mask.data = mask.to(self.control_mask.device)
        self.selected_channels = selected_channels_np

    def compute_activation_degree(self, conv1_out):
        """
        Extract 32-bit pattern from conv1 output (ADAPTIVE for different architectures)
        
        Key: Takes CENTER POINT value from each 3x3 patch, NOT average!
        - 8 spatial positions in a 2x4 grid
        - Each position: 3x3 patch, extract center point (y+1, x+1)
        - Extract bits from first 4 channels at center point
        
        Adaptive: Automatically scales to feature map size (32x32 for CNN, 8x8 for ViT)
        """
        batch_size = conv1_out.size(0)
        device = conv1_out.device
        H, W = conv1_out.size(2), conv1_out.size(3)
        
        all_bits_from_patches = []
        
        # Adaptive spatial grid based on feature map size
        if H == 32 and W == 32:
            # CNN models: 32x32 feature map
            # Original 2x4 grid with 3x3 patches
            grid_positions = [
                (1, 1), (1, 4), (1, 7), (1, 10),   # Row 1
                (4, 1), (4, 4), (4, 7), (4, 10)    # Row 2
            ]
        elif H == 8 and W == 8:
            # ViT models: 8x8 feature map (patch_size=4)
            # Scaled down 2x4 grid
            grid_positions = [
                (1, 1), (1, 2), (1, 4), (1, 5),    # Row 1
                (4, 1), (4, 2), (4, 4), (4, 5)     # Row 2
            ]
        else:
            raise ValueError(f"Unsupported feature map size: {H}x{W}. Expected 32x32 or 8x8")
        
        # Extract bits from 8 spatial locations
        for center_y, center_x in grid_positions:
            # Extract values at center point from first 4 channels
            values_at_center = conv1_out[:, 0:4, center_y, center_x]  # (B, 4)
            
            # Extract bits: (abs(trunc(value * 64))) % 2
            extracted_bits = (torch.abs(torch.trunc(values_at_center * 64)) % 2)
            all_bits_from_patches.append(extracted_bits)
        
        # Organize into 32-bit pattern (interleaved layout)
        extracted_pattern = torch.zeros(batch_size, 32, device=device)
        for j in range(8):
            for k in range(4):
                pattern_bit_index = j + k * 8
                extracted_pattern[:, pattern_bit_index] = all_bits_from_patches[j][:, k]
        
        # Compute activation degree: linear ramp from [21, 32] to [0, 1]
        match_counts = torch.sum(extracted_pattern == self.vip_pattern, dim=1).float()
        min_threshold = 21.0
        max_val = 32.0
        activation_degree = (match_counts - min_threshold) / (max_val - min_threshold)
        activation_degree = torch.clamp(activation_degree, 0.0, 1.0)
        
        return activation_degree, extracted_pattern
    
    def inject_sparse_noise(self, features, activation_degree, noise_scale):
        """
        Inject sparse noise with VIP-aware antidote
        
        Mechanism:
            Normal users: receive noise -> degraded service
            VIP users: receive noise + antidote -> normal service
        
        Note: noise_scale is used as Linf bound (same as UAP epsilon)
        """
        B = features.size(0)

        # Generate sparse noise with Linf constraint (matching UAP/GD-UAP)
        full_noise = torch.randn_like(features)
        full_noise = torch.clamp(full_noise, -noise_scale, noise_scale)  # Linf bound
        sparse_noise = full_noise * self.control_mask.view(1, -1, 1, 1)

        sparse_antidote = -sparse_noise
        
        # Combine: noise + activation_degree * antidote
        activation_reshaped = activation_degree.view(B, 1, 1, 1)
        injection_signal = sparse_noise + activation_reshaped * sparse_antidote
        
        return features + injection_signal
    
    def extract_conv1_output(self, x):
        """Extract conv1 output (architecture-specific)"""
        raise NotImplementedError
    
    def forward_from_conv1(self, conv1_out, noise_scale, activation_degree):
        """Forward from conv1 to final output (architecture-specific)"""
        raise NotImplementedError
    
    def forward(self, x, noise_scale=1.5, force_activation_degree=None):
        """
        Forward pass with sparse control

        Args:
            x: Input tensor
            noise_scale: Gaussian noise scale
            force_activation_degree: If provided (0.0 or 1.0), bypass watermark
                detection and use this value directly. Useful for evaluation
                with clean (non-watermarked) data.

        Returns:
            logits: Classification output
            activation_degree: VIP scores
            extracted_pattern: Detected patterns
        """
        conv1_out = self.extract_conv1_output(x)
        if force_activation_degree is not None:
            batch_size = x.size(0)
            activation_degree = torch.full((batch_size,), force_activation_degree,
                                           device=x.device, dtype=torch.float32)
            extracted_pattern = torch.zeros(batch_size, 32, device=x.device)
        else:
            activation_degree, extracted_pattern = self.compute_activation_degree(conv1_out)
        logits = self.forward_from_conv1(conv1_out, noise_scale, activation_degree)

        return logits, activation_degree, extracted_pattern
    

class ResNetControlModel(BaseControlModel):
    """Control model for ResNet-18/50/101"""
    
    def extract_conv1_output(self, x):
        return self.backbone.conv1(x)
    
    def forward_from_conv1(self, conv1_out, noise_scale, activation_degree):
        # BatchNorm -> Noise injection -> ReLU
        x = self.backbone.bn1(conv1_out)
        x = self.inject_sparse_noise(x, activation_degree, noise_scale)
        x = F.relu(x)
        
        # ResNet blocks
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Classification head
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.backbone.linear(x)




class ViTControlModel(BaseControlModel):
    """Control model for Vision Transformer"""
    
    def __init__(self, backbone, vip_pattern, selected_channels=None):
        # Auto-detect embed dimension
        if hasattr(backbone, 'patch_embed'):
            num_channels = backbone.patch_embed.conv1.out_channels
        elif hasattr(backbone, 'conv1'):
            num_channels = backbone.conv1.out_channels
        else:
            raise ValueError("Cannot find conv1/patch_embed in ViT backbone")
        
        super().__init__(
            backbone=backbone,
            vip_pattern=vip_pattern,
            selected_channels=selected_channels,
            num_conv1_channels=num_channels
        )
    
    def extract_conv1_output(self, x):
        if hasattr(self.backbone, 'patch_embed'):
            return self.backbone.patch_embed.conv1(x)
        return self.backbone.conv1(x)
    
    def forward_from_conv1(self, conv1_out, noise_scale, activation_degree):
        # Inject noise into patch embeddings
        x = self.inject_sparse_noise(conv1_out, activation_degree, noise_scale)
        
        # Flatten patches: (B, C, H, W) -> (B, N, C)
        x = x.flatten(2).transpose(1, 2)
        
        # Add [CLS] token and position embedding
        B = x.shape[0]
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.dropout(x)
        
        # Transformer blocks
        for block in self.backbone.blocks:
            x = block(x)
        
        # Classification head
        x = self.backbone.norm(x)
        return self.backbone.linear(x[:, 0])
    
def create_control_model(model_name, vip_pattern, selected_channels, model_path, device):
    MODELS = {
        'resnet18': (ResNet18, ResNetControlModel),
        'resnet50': (ResNet50, ResNetControlModel),
        'vit_tiny': (ViT_Tiny, ViTControlModel),
        'vit_small': (ViT_Small, ViTControlModel),
        'vit_base': (ViT_Base, ViTControlModel),
    }
    
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    # Load backbone
    backbone_cls, control_cls = MODELS[model_name]
    backbone = backbone_cls(num_class=10)
    backbone.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Create control model
    control_model = control_cls(
        backbone=backbone,
        vip_pattern=vip_pattern,
        selected_channels=selected_channels
    )
    
    return control_model.to(device)