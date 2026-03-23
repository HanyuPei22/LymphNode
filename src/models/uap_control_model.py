import torch
import torch.nn as nn
import torch.nn.functional as F

from .control_model import BaseControlModel


class UAPControlModel(BaseControlModel):
    """
    Control model using pre-trained UAP Delta
    
    Mechanism:
        - Normal users: receive fixed Delta → degraded performance
        - VIP users: receive Delta + antidote → restored performance
    """
    
    def __init__(self, backbone, vip_pattern, delta, selected_channels, num_conv1_channels=64):
        """
        Args:
            backbone: Pre-trained clean model
            vip_pattern: 32-bit VIP detection pattern
            delta: Pre-trained adversarial perturbation [1, C, H, W]
            selected_channels: Channel indices used in Delta training
            num_conv1_channels: Total channels in conv1 layer
        """
        super().__init__(
            backbone=backbone,
            vip_pattern=vip_pattern,
            selected_channels=selected_channels,
            num_conv1_channels=num_conv1_channels
        )
        
        # Register fixed Delta as buffer (non-trainable)
        if delta.dim() == 3:
            delta = delta.unsqueeze(0)  # [C, H, W] → [1, C, H, W]
        self.register_buffer('delta', delta)
    
    def inject_uap(self, features, activation_degree):
        """
        Inject fixed UAP with VIP-aware antidote
        
        Args:
            features: Conv1 output after BN [B, C, H, W]
            activation_degree: VIP detection scores [B]
            
        Returns:
            Controlled features [B, C, H, W]
        """
        B = features.size(0)
        
        # Sparse Delta (already masked during training)
        sparse_delta = self.delta  # [1, C, H, W]
        
        # Sparse antidote (negative of Delta)
        sparse_antidote = -sparse_delta
        
        # VIP-aware injection: Delta + activation_degree * (-Delta)
        activation = activation_degree.view(B, 1, 1, 1)
        injection = sparse_delta + activation * sparse_antidote
        
        return features + injection
    
    def forward(self, x, force_activation_degree=None):
        """
        Forward pass with UAP control

        Args:
            x: Input tensor
            force_activation_degree: If provided (0.0 or 1.0), bypass watermark
                detection and use this value directly.

        Returns:
            logits: Classification output [B, num_classes]
            activation_degree: VIP scores [B]
            extracted_pattern: Detected patterns [B, 32]
        """
        conv1_out = self.extract_conv1_output(x)
        if force_activation_degree is not None:
            batch_size = x.size(0)
            activation_degree = torch.full((batch_size,), force_activation_degree,
                                           device=x.device, dtype=torch.float32)
            extracted_pattern = torch.zeros(batch_size, 32, device=x.device)
        else:
            activation_degree, extracted_pattern = self.compute_activation_degree(conv1_out)
        logits = self.forward_from_conv1_uap(conv1_out, activation_degree)

        return logits, activation_degree, extracted_pattern
    
    def forward_from_conv1_uap(self, conv1_out, activation_degree):
        """Forward from conv1 with UAP injection (architecture-agnostic)"""
        raise NotImplementedError("Subclass must implement this method")


class ResNetUAPModel(UAPControlModel):
    """UAP control model for ResNet architectures"""
    
    def extract_conv1_output(self, x):
        return self.backbone.conv1(x)
    
    def forward_from_conv1_uap(self, conv1_out, activation_degree):
        x = self.backbone.bn1(conv1_out)
        x = self.inject_uap(x, activation_degree)
        x = F.relu(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.backbone.linear(x)


class ViTUAPModel(UAPControlModel):
    """UAP control model for Vision Transformer"""
    
    def __init__(self, backbone, vip_pattern, delta, selected_channels):
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
            delta=delta,
            selected_channels=selected_channels,
            num_conv1_channels=num_channels
        )
    
    def extract_conv1_output(self, x):
        if hasattr(self.backbone, 'patch_embed'):
            return self.backbone.patch_embed.conv1(x)
        return self.backbone.conv1(x)
    
    def forward_from_conv1_uap(self, conv1_out, activation_degree):
        x = self.inject_uap(conv1_out, activation_degree)
        
        # Flatten patches: (B, C, H, W) → (B, N, C)
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


def create_uap_model(model_name, vip_pattern, delta, selected_channels, model_path, device):
    from .resnet import ResNet18, ResNet50
    from .vit import ViT_Tiny, ViT_Small, ViT_Base

    MODELS = {
        'resnet18': (ResNet18, ResNetUAPModel),
        'resnet50': (ResNet50, ResNetUAPModel),
        'vit_tiny': (ViT_Tiny, ViTUAPModel),
        'vit_small': (ViT_Small, ViTUAPModel),
        'vit_base': (ViT_Base, ViTUAPModel),
    }
    
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    # Load backbone
    backbone_cls, uap_cls = MODELS[model_name]
    backbone = backbone_cls(num_class=10)
    backbone.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Create UAP control model
    uap_model = uap_cls(
        backbone=backbone,
        vip_pattern=vip_pattern,
        delta=delta,
        selected_channels=selected_channels
    )
    
    return uap_model.to(device)
