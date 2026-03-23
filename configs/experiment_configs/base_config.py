# configs/experiment_configs/base_config.py
import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Dataset configuration"""
    name: str = 'CIFAR10'
    num_classes: int = 10
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str = 'resnet18'
    num_classes: int = 10
    pretrained: bool = True


@dataclass
class ControlConfig:
    """Control mechanism configuration"""
    vip_pattern_id: int = 118
    noise_scale: float = 1.5
    selected_channels: Optional[List[int]] = None
    selection_method: str = 'hrank'  # hrank, random, activation_based, etc.
    num_channels: int = 64  # Number of channels to control


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 60
    optimizer: str = 'adam'
    scheduler: str = 'step'
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    num_samples: int = 1000  # Samples per category (VIP/Normal)
    batch_size: int = 32
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'control_gap', 'control_efficiency'
    ])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    
    # Experiment metadata
    experiment_name: str = 'baseline_control'
    description: str = 'Baseline control experiment'
    random_seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment-specific settings
    num_repeats: int = 1
    save_checkpoint: bool = True
    log_interval: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        assert self.control.num_channels <= 64, "num_channels cannot exceed 64"
        assert self.control.noise_scale > 0, "noise_scale must be positive"
        
    def to_dict(self):
        """Convert config to dictionary for logging"""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'random_seed': self.random_seed,
            'device': self.device,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'control': self.control.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'num_repeats': self.num_repeats,
        }


def create_baseline_config(model_name='resnet18', num_channels=64):
    """Factory function for baseline experiment config"""
    config = ExperimentConfig(
        experiment_name=f'baseline_{model_name}_ch{num_channels}',
        description=f'Baseline control with {model_name} using {num_channels} channels'
    )
    config.model.name = model_name
    config.control.num_channels = num_channels
    return config


def create_sparsity_config(model_name='resnet18', selection_method='hrank', sparsity_levels=None):
    """Factory function for sparsity experiment config"""
    if sparsity_levels is None:
        sparsity_levels = [64, 48, 32, 24, 16, 12, 8, 4]
    
    configs = []
    for num_channels in sparsity_levels:
        config = ExperimentConfig(
            experiment_name=f'sparsity_{model_name}_{selection_method}_ch{num_channels}',
            description=f'Sparsity experiment with {selection_method} selection'
        )
        config.model.name = model_name
        config.control.num_channels = num_channels
        config.control.selection_method = selection_method
        configs.append(config)
    
    return configs


def create_data_efficiency_config(model_name='resnet18', num_samples=100):
    """Factory function for data efficiency experiment config"""
    config = ExperimentConfig(
        experiment_name=f'data_efficiency_{model_name}_samples{num_samples}',
        description=f'Data efficiency with {num_samples} samples for channel selection'
    )
    config.model.name = model_name
    config.evaluation.num_samples = num_samples
    return config


if __name__ == '__main__':
    # Test configuration creation
    print("Testing Configuration Classes")
    print("=" * 60)
    
    # Test baseline config
    config = create_baseline_config('resnet18', 32)
    print(f"\nBaseline Config:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Model: {config.model.name}")
    print(f"  Channels: {config.control.num_channels}")
    print(f"  Noise scale: {config.control.noise_scale}")
    
    # Test sparsity configs
    configs = create_sparsity_config('resnet18', 'hrank', [32, 16, 8])
    print(f"\nSparsity Configs: {len(configs)} configurations")
    for cfg in configs:
        print(f"  - {cfg.experiment_name}")
    
    print("\n" + "=" * 60)
    print("✓ Configuration system working correctly")