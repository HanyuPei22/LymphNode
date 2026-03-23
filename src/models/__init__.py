from .resnet import ResNet18, ResNet50
from .vit import ViT_Tiny, ViT_Small, ViT_Base
from .control_model import (
    BaseControlModel,
    ResNetControlModel,
    ViTControlModel,
    create_control_model
)

__all__ = [
    'ResNet18', 'ResNet50',
    'ViT_Tiny', 'ViT_Small', 'ViT_Base',
    'BaseControlModel', 'ResNetControlModel', 'ViTControlModel',
    'create_control_model',
]


def get_model(model_name, num_class=10):
    models = {
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'vit_tiny': ViT_Tiny,
        'vit_small': ViT_Small,
        'vit_base': ViT_Base
    }
    model_name = model_name.lower()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    return models[model_name](num_class=num_class)
