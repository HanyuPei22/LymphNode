"""
Train Universal Adversarial Perturbations (UAP) for Sparse Control

Trains fixed adversarial noise Delta for different models and channel ratios.
Each Delta is optimized offline on clean CIFAR-10 calibration data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from src.models.resnet import ResNet18, ResNet50
from src.models.vit import ViT_Tiny, ViT_Small
from src.selection import get_selector
from src.training import UAPTrainer


def get_balanced_indices(dataset, num_samples, num_classes=10, seed=42):
    """
    Sample balanced subset from dataset ensuring equal class representation
    
    Args:
        dataset: CIFAR10 dataset with targets attribute
        num_samples: Total samples to draw
        num_classes: Number of classes (10 for CIFAR-10)
        seed: Random seed for reproducibility
    
    Returns:
        indices: Tensor of balanced sample indices
    """
    np.random.seed(seed)
    per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    # Group indices by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx in range(len(dataset)):
        label = dataset.targets[idx] if hasattr(dataset, 'targets') else dataset[idx][1]
        class_indices[label].append(idx)
    
    # Sample from each class
    balanced_indices = []
    for class_id in range(num_classes):
        class_idx = np.array(class_indices[class_id])
        # Sample per_class samples, or all available if insufficient
        n_samples = min(per_class, len(class_idx))
        sampled = np.random.choice(class_idx, n_samples, replace=False)
        balanced_indices.extend(sampled.tolist())
    
    # Add remaining samples from random classes if num_samples % num_classes != 0
    if remainder > 0:
        all_remaining = []
        for class_id in range(num_classes):
            class_idx = set(class_indices[class_id]) - set(balanced_indices)
            all_remaining.extend(list(class_idx))
        if len(all_remaining) >= remainder:
            extra = np.random.choice(all_remaining, remainder, replace=False)
            balanced_indices.extend(extra.tolist())
    
    # Shuffle and convert to tensor
    np.random.shuffle(balanced_indices)
    return torch.tensor(balanced_indices[:num_samples])


def get_random_indices(dataset_size, num_samples, seed=42):
    """
    Random sampling strategy (original implementation)
    
    Args:
        dataset_size: Total size of dataset
        num_samples: Number of samples to draw
        seed: Random seed
    
    Returns:
        indices: Tensor of random sample indices
    """
    torch.manual_seed(seed)
    return torch.randperm(dataset_size)[:num_samples]


def get_model_config(model_name):
    """Get model-specific configuration"""
    configs = {
        'resnet18': {'conv1_channels': 64, 'feature_size': 32, 'type': 'resnet'},
        'resnet50': {'conv1_channels': 64, 'feature_size': 32, 'type': 'resnet'},
        'vit_tiny': {'conv1_channels': 192, 'feature_size': 8, 'type': 'vit'},
        'vit_small': {'conv1_channels': 384, 'feature_size': 8, 'type': 'vit'},
    }
    return configs.get(model_name.lower())


def get_model_path(base_path, model_name, dataset='cifar10'):
    model_epochs = {
        'resnet18':  {'cifar10': '', 'mnist': 40, 'svhn': 60},
        'resnet50':  {'cifar10': 100, 'mnist': 50, 'svhn': 70},
        'vit_tiny':  {'cifar10': 80, 'mnist': 50, 'svhn': 50},
        'vit_small': {'cifar10': 80, 'mnist': 50, 'svhn': 50},
    }

    if model_name not in model_epochs:
        raise ValueError(f"Unknown model: {model_name}")

    dataset_epochs = model_epochs[model_name]
    if dataset not in dataset_epochs:
        raise ValueError(f"Model {model_name} has no checkpoint for dataset {dataset}")

    epoch = dataset_epochs[dataset]
    filename = f"clean_{model_name}_{dataset}_epoch{epoch if epoch else ''}.pth"
    return os.path.join(base_path, 'checkpoints', 'clean_models', filename)


def select_channels(model_name, method, ratio, dataloader, device, base_path, dataset='cifar10'):
    """
    Select channels using specified method and ratio

    Args:
        model_name: Architecture name
        method: Selection method ('hrank', 'random', etc.)
        ratio: Channel selection ratio (0.2 = 20%)
        dataloader: Calibration data
        device: Computation device
        base_path: Project root path
        dataset: Dataset name ('cifar10', 'mnist', 'svhn')

    Returns:
        selected_channels: Channel indices tensor
        channel_mask: Binary mask tensor
    """
    model_cfg = get_model_config(model_name)
    total_channels = model_cfg['conv1_channels']
    k = max(1, int(total_channels * ratio))

    print(f"  Selecting {k}/{total_channels} channels ({ratio*100:.0f}%) using {method}...")

    selector = get_selector(method)

    model_classes = {
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'vit_tiny': ViT_Tiny,
        'vit_small': ViT_Small,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_classes.keys())}")

    model_path = get_model_path(base_path, model_name, dataset)
    clean_model = model_classes[model_name](num_class=10)
    clean_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    clean_model = clean_model.to(device)
    clean_model.eval()

    # Select channels (selector already handles both model.conv1 and model.backbone.conv1)
    selected_channels = selector.select(
        model=clean_model,
        dataloader=dataloader,
        k=k,
        num_samples=500,
        device=device
    )

    # Create binary mask
    channel_mask = torch.zeros(total_channels)
    channel_mask[selected_channels] = 1.0

    return selected_channels, channel_mask


def get_dataset_loader(dataset_name, base_path, config):
    """Load dataset based on dataset name"""
    dataset_configs = {
        'cifar10': {
            'root': os.path.join(base_path, 'CIFAR'),
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'num_classes': 10,
            'dataset_cls': datasets.CIFAR10,
            'split_arg': 'train'
        },
        'mnist': {
            'root': os.path.join(base_path, 'MNIST'),
            'mean': (0.1307,),
            'std': (0.3081,),
            'num_classes': 10,
            'dataset_cls': datasets.MNIST,
            'split_arg': 'train'
        },
        'svhn': {
            'root': os.path.join(base_path, 'SVHN'),
            'mean': (0.4377, 0.4438, 0.4728),
            'std': (0.1980, 0.2010, 0.1970),
            'num_classes': 10,
            'dataset_cls': datasets.SVHN,
            'split_arg': 'split'
        }
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(dataset_configs.keys())}")

    cfg = dataset_configs[dataset_name]

    # Prepare transforms
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.Pad((2, 2)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std'])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg['mean'], cfg['std'])
        ])

    # Load dataset
    if dataset_name == 'svhn':
        dataset = cfg['dataset_cls'](
            root=cfg['root'],
            split='train',
            transform=transform,
            download=False
        )
    else:
        dataset = cfg['dataset_cls'](
            root=cfg['root'],
            train=True,
            transform=transform,
            download=False
        )

    return dataset, cfg


def train_single_uap(model_name, ratio, method, config, dataset='cifar10'):
    """
    Train UAP for single model-ratio combination

    Args:
        model_name: Architecture name
        ratio: Channel ratio
        method: Channel selection method
        config: Global configuration
        dataset: Dataset name ('cifar10', 'mnist', 'svhn')

    Returns:
        delta_path: Path to saved Delta file
    """
    device = config['device']
    base_path = config['base_path']
    sampling_strategy = config.get('sampling_strategy', 'random')
    calib_samples = config.get('calibration_samples', 1000)

    # Check if already exists
    output_dir = os.path.join(base_path, config['output_dir'])

    # Filename format: {model}_ratio{X}_calib{N}_{method}_{strategy}_{dataset}.pth
    filename = f"{model_name}_{dataset}_ratio{int(ratio*100)}_calib{calib_samples}_{method}_{sampling_strategy}.pth"
    delta_path = os.path.join(output_dir, filename)

    if os.path.exists(delta_path) and not config.get('overwrite', False):
        print(f" Skipping {model_name} on {dataset} @ {ratio*100:.0f}% (calib={calib_samples}, {sampling_strategy}): Already exists")
        return delta_path

    print(f"\n{'='*60}")
    print(f"Training UAP: {model_name.upper()} on {dataset.upper()} @ {ratio*100:.0f}% ({method})")
    print(f"{'='*60}")

    # Load dataset and prepare calibration data
    calib_dataset, dataset_cfg = get_dataset_loader(dataset, base_path, config)

    # Sample subset for calibration based on strategy
    sampling_strategy = config.get('sampling_strategy', 'random')

    if sampling_strategy == 'balanced':
        print(f"  Using BALANCED sampling (equal samples per class)")
        indices = get_balanced_indices(
            calib_dataset,
            config['calibration_samples'],
            num_classes=dataset_cfg['num_classes'],
            seed=config['seed']
        )
    else:  # random (default)
        print(f"  Using RANDOM sampling")
        indices = get_random_indices(
            len(calib_dataset),
            config['calibration_samples'],
            seed=config['seed']
        )

    # Verify class distribution
    # Handle different dataset formats (CIFAR10 uses 'targets', SVHN uses 'labels')
    if hasattr(calib_dataset, 'targets'):
        targets_list = calib_dataset.targets
    elif hasattr(calib_dataset, 'labels'):
        targets_list = calib_dataset.labels
    else:
        # Fallback: try to extract from dataset items
        targets_list = [calib_dataset[i][1] for i in range(len(calib_dataset))]

    sample_labels = [targets_list[i] for i in indices]
    class_counts = {i: sample_labels.count(i) for i in range(dataset_cfg['num_classes'])}
    print(f"  Class distribution: {class_counts}")
    print(f"  Min/Max samples per class: {min(class_counts.values())}/{max(class_counts.values())}")

    calib_dataset_subset = torch.utils.data.Subset(calib_dataset, indices)

    calib_loader = torch.utils.data.DataLoader(
        calib_dataset_subset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Channel selection (use all calibration samples for consistency)
    selection_loader = torch.utils.data.DataLoader(
        calib_dataset_subset,  # Use same calibration data
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    selected_channels, channel_mask = select_channels(
        model_name=model_name,
        method=method,
        ratio=ratio,
        dataloader=selection_loader,
        device=device,
        base_path=base_path,
        dataset=dataset
    )

    print(f"  Selected channels: {selected_channels.tolist()[:10]}..." if len(selected_channels) > 10
          else f"  Selected channels: {selected_channels.tolist()}")
    
    # Load clean model for UAP training
    model_classes = {
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'vit_tiny': ViT_Tiny,
        'vit_small': ViT_Small,
    }

    if model_name not in model_classes:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_classes.keys())}")

    model_path = get_model_path(base_path, model_name, dataset)
    clean_model = model_classes[model_name](num_class=dataset_cfg['num_classes'])
    clean_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    clean_model = clean_model.to(device)
    clean_model.eval()
    
    # Initialize UAP trainer
    model_cfg = get_model_config(model_name)
    feature_shape = (
        model_cfg['conv1_channels'],
        model_cfg['feature_size'],
        model_cfg['feature_size']
    )
    
    trainer = UAPTrainer(
        backbone=clean_model,
        channel_mask=channel_mask,
        feature_shape=feature_shape,
        config={
            'device': device,
            'epsilon': config['epsilon'],
            'alpha': config['alpha']
        }
    )
    
    # Train UAP
    print(f"\n  Training UAP (epsilon={config['epsilon']}, alpha={config['alpha']})...")
    delta = trainer.train(
        dataloader=calib_loader,
        epochs=config['uap_epochs'],
        model_type=model_cfg['type']
    )
    
    # Save Delta (directory already checked at function start)
    os.makedirs(output_dir, exist_ok=True)
    
    trainer.save(delta_path, metadata={
        'model': model_name,
        'ratio': ratio,
        'method': method,
        'sampling_strategy': sampling_strategy,
        'calibration_samples': config.get('calibration_samples', 1000),
        'selected_channels': selected_channels.cpu().numpy(),
        'num_channels': len(selected_channels),
        'total_channels': model_cfg['conv1_channels']
    })
    
    return delta_path


def main():
    parser = argparse.ArgumentParser(description='Train UAP for sparse control')
    parser.add_argument('--models', nargs='+', default=['resnet18'],
                        help='Models to train (resnet18, resnet50, vit_tiny, vit_small)')
    parser.add_argument('--ratios', nargs='+', type=float, 
                        default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help='Channel ratios: 0.2=20%%, 0.4=40%%, 0.6=60%%, 0.8=80%%, 1.0=100%%')
    parser.add_argument('--method', type=str, default='weight_gradient_based',
                        help='Channel selection method')
    parser.add_argument('--epsilon', type=float, default=2.0,
                        help='L-infinity bound for Delta')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='PGA step size')
    parser.add_argument('--uap-epochs', type=int, default=20,
                        help='UAP training epochs')
    parser.add_argument('--calibration-samples', type=int, default=None,
                        help='Single calibration data size (use with --calibration-sizes for batch mode)')
    parser.add_argument('--calibration-sizes', nargs='+', type=int, default=None,
                        help='Multiple calibration sizes to train (e.g., 100 500 1000 2000 5000)')
    parser.add_argument('--sampling-strategy', type=str, default='random',
                        choices=['random', 'balanced'],
                        help='Calibration data sampling strategy: random (default) or balanced (equal samples per class)')
    parser.add_argument('--sampling-strategies', nargs='+', type=str, default=None,
                        choices=['random', 'balanced'],
                        help='Multiple sampling strategies to test (e.g., random balanced)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--pattern-id', type=int, default=118,
                        help='VIP pattern ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--base-path', type=str, default=str(Path(__file__).parent.parent),
                        help='Project root directory')
    parser.add_argument('--output-dir', type=str, default='results/final_uap',
                        help='Output directory for trained Deltas (default: results/final_uap)')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                        choices=['cifar10', 'mnist', 'svhn'],
                        help='Datasets to train on (default: cifar10)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing UAP files (default: skip if exists)')

    args = parser.parse_args()

    # Determine calibration sizes and sampling strategies to test
    calib_sizes = args.calibration_sizes if args.calibration_sizes else ([args.calibration_samples] if args.calibration_samples else [1000])
    sampling_strategies = args.sampling_strategies if args.sampling_strategies else [args.sampling_strategy]

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # Calculate total runs
    total_runs = len(args.models) * len(args.ratios) * len(calib_sizes) * len(sampling_strategies) * len(args.datasets)
    print(f"\nTotal UAP training runs: {total_runs}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Ratios: {[f'{r*100:.0f}%' for r in args.ratios]}")
    print(f"Calibration sizes: {calib_sizes}")
    print(f"Sampling strategies: {sampling_strategies}")
    print(f"Method: {args.method}")
    print(f"Overwrite existing: {args.overwrite}\n")

    # Train all combinations
    trained_deltas = []
    with tqdm(total=total_runs, desc="Overall Progress") as pbar:
        for dataset in args.datasets:
            for model_name in args.models:
                for ratio in args.ratios:
                    for calib_samples in calib_sizes:
                        for sampling_strategy in sampling_strategies:
                            # Update config for this combination
                            config = {
                                'device': device,
                                'base_path': args.base_path,
                                'output_dir': args.output_dir,
                                'pattern_id': args.pattern_id,
                                'epsilon': args.epsilon,
                                'alpha': args.alpha,
                                'uap_epochs': args.uap_epochs,
                                'calibration_samples': calib_samples,
                                'sampling_strategy': sampling_strategy,
                                'batch_size': args.batch_size,
                                'seed': args.seed,
                                'overwrite': args.overwrite
                            }

                            delta_path = train_single_uap(model_name, ratio, args.method, config, dataset)
                            trained_deltas.append(delta_path)
                            pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete! Saved {len(trained_deltas)} UAPs")
    print(f"{'='*60}")
    print(f"Output directory: {os.path.join(config['base_path'], config['output_dir'])}")


if __name__ == '__main__':
    main()
