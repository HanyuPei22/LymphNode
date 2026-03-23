import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class UAPTrainer:
    """Trainer for sparse UAP using PGA (Projected Gradient Ascent)"""

    def __init__(self, backbone, channel_mask, feature_shape, config):
        self.backbone = backbone
        self.device = config['device']
        self.epsilon = config['epsilon']
        self.alpha = config['alpha']

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

    def train(self, dataloader, epochs, model_type='resnet', verbose=True):
        input_layer = self.backbone.conv1

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if verbose else dataloader

            for images, labels in iterator:
                images = images.to(self.device)
                labels = labels.to(self.device)

                conv1_out = input_layer(images)
                bn_out = self.backbone.bn1(conv1_out)
                perturbed_features = bn_out + self.delta

                if 'vit' in model_type.lower():
                    logits = self._forward_from_bn1_vit(perturbed_features)
                else:
                    logits = self._forward_from_bn1(perturbed_features)

                loss = F.cross_entropy(logits, labels)

                self.delta.grad = None
                loss.backward()

                with torch.no_grad():
                    self.delta.data.add_(self.alpha * self.delta.grad.sign())
                    self.delta.data = self.delta.data * self.mask
                    self.delta.data = torch.clamp(self.delta.data, -self.epsilon, self.epsilon)

                epoch_loss += loss.item()
                _, pred = logits.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

                if verbose and hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({'loss': f'{loss.item():.3f}'})

            if verbose or epoch == epochs - 1:
                avg_loss = epoch_loss / len(dataloader)
                attack_success = 100.0 * (1 - correct / total)
                print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Attack Success={attack_success:.2f}%")

        return self.delta.detach()

    def save(self, filepath, metadata=None):
        save_dict = {
            'delta': self.delta.detach().cpu(),
            'mask': self.mask.squeeze().cpu(),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
        }
        if metadata:
            save_dict.update(metadata)
        torch.save(save_dict, filepath)
        print(f"Saved UAP to {filepath}")

    @staticmethod
    def load(filepath):
        return torch.load(filepath, map_location='cpu')
