import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class GD_UAPTrainer:
    """
    Trainer for GD-UAP using Adam on activation maximization objective

    Loss = -sum_l log(||F_l(x + delta)||_2)
    """

    def __init__(self, backbone, channel_mask, feature_shape, config):
        self.backbone = backbone
        self.device = config['device']
        self.epsilon = config['epsilon']
        self.lr = config.get('lr', 0.1)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.mask = channel_mask.to(self.device).view(1, -1, 1, 1)

        C, H, W = feature_shape
        self.delta = nn.Parameter(
            torch.empty(1, C, H, W, device=self.device).uniform_(-self.epsilon, self.epsilon)
        )

        with torch.no_grad():
            self.delta.data = self.delta.data * self.mask

        self.activations = {}
        self.hooks = []

    def _register_hooks(self, layer_names):
        def get_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        for layer_name in layer_names:
            layer = dict(self.backbone.named_modules())[layer_name]
            handle = layer.register_forward_hook(get_hook(layer_name))
            self.hooks.append(handle)

    def _remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def _forward_from_bn1(self, bn_out):
        """ResNet forward from post-BN1"""
        x = F.relu(bn_out)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return self.backbone.linear(x)

    def _forward_from_bn1_vit(self, bn_out):
        """ViT forward from post-conv1"""
        x = bn_out.flatten(2).transpose(1, 2)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.backbone.pos_embed
        if hasattr(self.backbone, 'dropout'):
            x = self.backbone.dropout(x)
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.norm(x)
        return self.backbone.linear(x[:, 0])

    def _compute_loss(self):
        loss = 0.0
        for layer_name, activation in self.activations.items():
            norm = torch.norm(activation, p=2)
            loss -= torch.log(norm + 1e-8)
        return loss

    def train(self, dataloader, epochs, model_type='resnet', layer_names=None, verbose=True):
        if layer_names is None:
            layer_names = self._get_default_layers(model_type)

        self._register_hooks(layer_names)
        optimizer = torch.optim.Adam([self.delta], lr=self.lr)

        sat_prev = 0.0
        sat_change_threshold = 0.0001
        sat_min_threshold = 0.5

        for epoch in range(epochs):
            epoch_loss = 0.0
            iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if verbose else dataloader

            for batch_idx, (images, _) in enumerate(iterator):
                images = images.to(self.device)
                self.activations.clear()

                conv1_out = self.backbone.conv1(images)
                bn_out = self.backbone.bn1(conv1_out)
                perturbed_features = bn_out + self.delta

                if 'vit' in model_type.lower():
                    _ = self._forward_from_bn1_vit(perturbed_features)
                else:
                    _ = self._forward_from_bn1(perturbed_features)

                loss = self._compute_loss()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    self.delta.data = torch.clamp(self.delta.data, -self.epsilon, self.epsilon)
                    self.delta.data = self.delta.data * self.mask

                    if batch_idx % 50 == 0:
                        total_pixels = self.delta.numel()
                        saturated_pixels = (torch.abs(self.delta) >= self.epsilon * 0.99).sum().item()
                        saturation = saturated_pixels / total_pixels
                        sat_change = abs(saturation - sat_prev)

                        if saturation > sat_min_threshold and sat_change < sat_change_threshold:
                            self.delta.data = self.delta.data / 2.0
                            if verbose:
                                print(f"\n  Rescaling delta (sat={saturation:.3f}, change={sat_change:.5f})")

                        sat_prev = saturation

                epoch_loss += loss.item()

                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({'loss': f'{loss.item():.1f}'})

            if verbose or epoch == epochs - 1:
                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.2f}")

        self._remove_hooks()
        return self.delta.detach()

    def _get_default_layers(self, model_type):
        layer_map = {
            'resnet18': ['layer1.1', 'layer2.1', 'layer3.1', 'layer4.1'],
            'resnet50': ['layer1.2', 'layer2.3', 'layer3.5', 'layer4.2'],
            'vit_tiny': ['blocks.1', 'blocks.2', 'blocks.3'],
            'vit_small': ['blocks.1', 'blocks.3', 'blocks.5'],
        }
        return layer_map.get(model_type, ['layer1.1', 'layer2.1', 'layer3.1', 'layer4.1'])

    def save(self, filepath, metadata=None):
        save_dict = {
            'delta': self.delta.detach().cpu(),
            'mask': self.mask.squeeze().cpu(),
            'epsilon': self.epsilon,
            'lr': self.lr,
        }
        if metadata:
            save_dict.update(metadata)
        torch.save(save_dict, filepath)
        print(f"Saved GD-UAP to {filepath}")

    @staticmethod
    def load(filepath):
        return torch.load(filepath, map_location='cpu')
