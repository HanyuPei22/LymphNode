"""
Train Gradient Descent UAP for Sparse Control

Trains fixed adversarial noise Delta using activation maximization objective.
Each Delta is optimized offline on clean CIFAR-10 calibration data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.resnet import ResNet18, ResNet50
from src.models.vit import ViT_Tiny, ViT_Small
import os
import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from src.selection import get_selector
from src.training import GD_UAPTrainer


def get_balanced_indices(dataset, num_samples, num_classes=10, seed=42):
    """Sample balanced subset from dataset ensuring equal class representation"""
    np.random.seed(seed)
    per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    class_indices = {i: [] for i in range(num_classes)}
    for idx in range(len(dataset)):
        label = dataset.targets[idx] if hasattr(dataset, 'targets') else dataset[idx][1]
        class_indices[label].append(idx)
    
    balanced_indices = []
    for class_id in range(num_classes):
        class_idx = np.array(class_indices[class_id])
        n_samples = min(per_class, len(class_idx))
        sampled = np.random.choice(class_idx, n_samples, replace=False)
        balanced_indices.extend(sampled.tolist())
    
    if remainder > 0:
        all_remaining = []
        for class_id in range(num_classes):
            class_idx = set(class_indices[class_id]) - set(balanced_indices)
            all_remaining.extend(list(class_idx))
        if len(all_remaining) >= remainder:
            extra = np.random.choice(all_remaining, remainder, replace=False)
            balanced_indices.extend(extra.tolist())
    
    np.random.shuffle(balanced_indices)
    return torch.tensor(balanced_indices[:num_samples])


def get_random_indices(dataset_size, num_samples, seed=42):
    """Random sampling strategy"""
    torch.manual_seed(seed)
    return torch.randperm(dataset_size)[:num_samples]


def get_model_config(model_name):
    """Get model-specific configuration"""
    configs = {
        'resnet18': {'conv1_channels': 64, 'feature_size': 32, 'type': 'resnet18'},
        'resnet50': {'conv1_channels': 64, 'feature_size': 32, 'type': 'resnet50'},
        'vit_tiny': {'conv1_channels': 192, 'feature_size': 8, 'type': 'vit_tiny'},
        'vit_small': {'conv1_channels': 384, 'feature_size': 8, 'type': 'vit_small'},
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
    return os.path.join(base_path, 'saved_models', 'clean_models', filename)


def select_channels(model_name, method, ratio, dataloader, device, base_path, dataset='cifar10'):
    """Select channels using specified method and ratio"""
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

    selected_channels = selector.select(
        model=clean_model,
        dataloader=dataloader,
        k=k,
        num_samples=500,
        device=device
    )

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


def train_single_gd_uap(model_name, ratio, method, config, dataset='cifar10'):
    """Train GD-UAP for single model-ratio combination"""
    device = config['device']
    base_path = config['base_path']
    sampling_strategy = config.get('sampling_strategy', 'random')
    calib_samples = config.get('calibration_samples', 1000)
    
    output_dir = os.path.join(base_path, config['output_dir'])
    filename = f"{model_name}_{dataset}_ratio{int(ratio*100)}_calib{calib_samples}_{method}_{sampling_strategy}_GD.pth"
    delta_path = os.path.join(output_dir, filename)

    if os.path.exists(delta_path) and not config.get('overwrite', False):
        print(f" Skipping {model_name} on {dataset} @ {ratio*100:.0f}% (calib={calib_samples}, {sampling_strategy}): Already exists")
        return delta_path

    print(f"\n{'='*60}")
    print(f"Training GD-UAP: {model_name.upper()} on {dataset.upper()} @ {ratio*100:.0f}% ({method})")
    print(f"{'='*60}")

    # Load dataset and prepare calibration data
    calib_dataset, dataset_cfg = get_dataset_loader(dataset, base_path, config)

    if sampling_strategy == 'balanced':
        print(f"  Using BALANCED sampling (equal samples per class)")
        indices = get_balanced_indices(
            calib_dataset,
            config['calibration_samples'],
            num_classes=dataset_cfg['num_classes'],
            seed=config['seed']
        )
    else:
        print(f"  Using RANDOM sampling")
        indices = get_random_indices(
            len(calib_dataset),
            config['calibration_samples'],
            seed=config['seed']
        )

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

    selection_loader = torch.utils.data.DataLoader(
        calib_dataset_subset,
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
    
    model_cfg = get_model_config(model_name)
    feature_shape = (
        model_cfg['conv1_channels'],
        model_cfg['feature_size'],
        model_cfg['feature_size']
    )
    
    trainer = GD_UAPTrainer(
        backbone=clean_model,
        channel_mask=channel_mask,
        feature_shape=feature_shape,
        config={
            'device': device,
            'epsilon': config['epsilon'],
            'lr': config['lr']
        }
    )
    
    print(f"\n  Training GD-UAP (epsilon={config['epsilon']}, lr={config['lr']})...")
    delta = trainer.train(
        dataloader=calib_loader,
        epochs=config['uap_epochs'],
        model_type=model_cfg['type']
    )
    
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
    parser = argparse.ArgumentParser(description='Train GD-UAP for sparse control')
    parser.add_argument('--models', nargs='+', default=['resnet18'],
                        help='Models to train (resnet18, resnet50, vit_tiny, vit_small)')
    parser.add_argument('--ratios', nargs='+', type=float, 
                        default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help='Channel ratios')
    parser.add_argument('--method', type=str, default='weight_gradient_based',
                        help='Channel selection method')
    parser.add_argument('--epsilon', type=float, default=2.0,
                        help='L-infinity bound for Delta')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Adam learning rate (original paper uses 0.1)')
    parser.add_argument('--uap-epochs', type=int, default=20,
                        help='GD-UAP training epochs')
    parser.add_argument('--calibration-samples', type=int, default=None,
                        help='Single calibration data size')
    parser.add_argument('--calibration-sizes', nargs='+', type=int, default=None,
                        help='Multiple calibration sizes to train')
    parser.add_argument('--sampling-strategy', type=str, default='random',
                        choices=['random', 'balanced'],
                        help='Calibration data sampling strategy')
    parser.add_argument('--sampling-strategies', nargs='+', type=str, default=None,
                        choices=['random', 'balanced'],
                        help='Multiple sampling strategies to test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--pattern-id', type=int, default=118,
                        help='VIP pattern ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--base-path', type=str, default='d:/Program/Plugin Neuron',
                        help='Project root directory')
    parser.add_argument('--output-dir', type=str, default='results/final_gd_uap',
                        help='Output directory for trained Deltas (default: results/final_gd_uap)')
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                        choices=['cifar10', 'mnist', 'svhn'],
                        help='Datasets to train on (default: cifar10)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing GD-UAP files')

    args = parser.parse_args()

    calib_sizes = args.calibration_sizes if args.calibration_sizes else ([args.calibration_samples] if args.calibration_samples else [1000])
    sampling_strategies = args.sampling_strategies if args.sampling_strategies else [args.sampling_strategy]

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    total_runs = len(args.models) * len(args.ratios) * len(calib_sizes) * len(sampling_strategies) * len(args.datasets)
    print(f"\nTotal GD-UAP training runs: {total_runs}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Ratios: {[f'{r*100:.0f}%' for r in args.ratios]}")
    print(f"Calibration sizes: {calib_sizes}")
    print(f"Sampling strategies: {sampling_strategies}")
    print(f"Method: {args.method}")
    print(f"Overwrite existing: {args.overwrite}\n")

    trained_deltas = []
    with tqdm(total=total_runs, desc="Overall Progress") as pbar:
        for dataset in args.datasets:
            for model_name in args.models:
                for ratio in args.ratios:
                    for calib_samples in calib_sizes:
                        for sampling_strategy in sampling_strategies:
                            config = {
                                'device': device,
                                'base_path': args.base_path,
                                'output_dir': args.output_dir,
                                'pattern_id': args.pattern_id,
                                'epsilon': args.epsilon,
                                'lr': args.lr,
                                'uap_epochs': args.uap_epochs,
                                'calibration_samples': calib_samples,
                                'sampling_strategy': sampling_strategy,
                                'batch_size': args.batch_size,
                                'seed': args.seed,
                                'overwrite': args.overwrite
                            }

                            delta_path = train_single_gd_uap(model_name, ratio, args.method, config, dataset)
                            trained_deltas.append(delta_path)
                            pbar.update(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Training Complete! Saved {len(trained_deltas)} GD-UAPs")
    print(f"{'='*60}")
    print(f"Output directory: {os.path.join(config['base_path'], config['output_dir'])}")


if __name__ == '__main__':
    main()
