"""
Experiment 1: Baseline Comparison - Gaussian vs UAP vs GD-UAP

Compare three noise injection methods for neutralization control:
1. Gaussian: Random noise on selected channels
2. UAP: Universal Adversarial Perturbation (offline trained)
3. GD-UAP: Gradient-Descent optimized UAP (offline trained)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models.uap_control_model import create_uap_model
from src.models.control_model import create_control_model
from src.data import get_clean_loaders


BASE_PATH = Path(__file__).parent.parent

MODEL_EPOCHS = {
    'resnet18':  {'cifar10': '', 'mnist': 40, 'svhn': 60},
    'resnet50':  {'cifar10': 100, 'mnist': 50, 'svhn': 70},
    'vit_tiny':  {'cifar10': 80, 'mnist': 50, 'svhn': 80},
    'vit_small': {'cifar10': 80, 'mnist': 50, 'svhn': 50},
}


def get_model_path(model_name, dataset='cifar10'):
    epoch = MODEL_EPOCHS[model_name].get(dataset, 100)
    filename = f"clean_{model_name}_{dataset}_epoch{epoch if epoch else ''}.pth"
    return str(BASE_PATH / 'checkpoints' / 'clean_models' / filename)


def load_uap_file(uap_dir, model_name, ratio, calib_samples, method, strategy, dataset='cifar10', is_gd=False):
    suffix = '_GD' if is_gd else ''
    filename = f"{model_name}_{dataset}_ratio{int(ratio*100)}_calib{calib_samples}_{method}_{strategy}{suffix}.pth"
    uap_path = BASE_PATH / 'results' / uap_dir / filename

    if uap_path.exists():
        return torch.load(str(uap_path), map_location='cpu')
    else:
        raise FileNotFoundError(f"UAP not found: {filename}")


def compute_accuracy(model, loader, max_samples, device, noise_scale=None, force_activation_degree=None):
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            if total >= max_samples:
                break
            images, labels = images.to(device), labels.to(device)

            kwargs = {}
            if noise_scale is not None:
                kwargs['noise_scale'] = noise_scale
            if force_activation_degree is not None:
                kwargs['force_activation_degree'] = force_activation_degree

            logits, _, _ = model(images, **kwargs)
            _, predicted = logits.max(1)

            batch_size = min(labels.size(0), max_samples - total)
            correct += predicted[:batch_size].eq(labels[:batch_size]).sum().item()
            total += batch_size

    return 100.0 * correct / total if total > 0 else 0.0


def run_baseline_comparison(model_name, ratio, noise_scale, calib_samples, config, dataset='cifar10'):
    device = config['device']
    method = config['method']
    strategy = config['sampling_strategy']

    print(f"\n{'='*70}")
    print(f"Model: {model_name.upper()} | Dataset: {dataset.upper()} | Ratio: {ratio*100:.0f}%")
    print(f"{'='*70}")

    try:
        uap_data = load_uap_file('final_uap', model_name, ratio, calib_samples, method, strategy, dataset=dataset, is_gd=False)
    except FileNotFoundError as e:
        print(f"  UAP not found, skipping: {e}")
        return None

    try:
        gd_uap_data = load_uap_file('final_gd_uap', model_name, ratio, calib_samples, method, strategy, dataset=dataset, is_gd=True)
    except FileNotFoundError as e:
        print(f"  GD-UAP not found, skipping: {e}")
        return None

    uap_delta = uap_data['delta'].to(device)
    gd_delta = gd_uap_data['delta'].to(device)
    selected_channels = torch.from_numpy(uap_data['selected_channels']).to(device)

    model_path = get_model_path(model_name, dataset=dataset)

    vip_loader, normal_loader = get_clean_loaders(
        dataset_name=dataset,
        batch_size=config['batch_size'],
        num_workers=0,
        num_samples=config['eval_samples']
    )

    results = {
        'model': model_name, 'dataset': dataset, 'ratio': ratio,
        'noise_scale': noise_scale, 'methods': {}
    }

    # Gaussian
    gaussian_model = create_control_model(
        model_name=model_name, vip_pattern=None,
        selected_channels=selected_channels,
        model_path=model_path, device=device
    )
    gaussian_model.eval()
    gaussian_vip = compute_accuracy(gaussian_model, vip_loader, config['eval_samples'], device, noise_scale, force_activation_degree=1)
    gaussian_normal = compute_accuracy(gaussian_model, normal_loader, config['eval_samples'], device, noise_scale, force_activation_degree=0)
    results['methods']['gaussian'] = {'vip_accuracy': gaussian_vip, 'normal_accuracy': gaussian_normal}
    print(f"  Gaussian:  VIP={gaussian_vip:.2f}%  Normal={gaussian_normal:.2f}%")
    del gaussian_model; torch.cuda.empty_cache()

    # SUAP
    uap_model = create_uap_model(
        model_name=model_name, vip_pattern=None,
        delta=uap_delta, selected_channels=selected_channels,
        model_path=model_path, device=device
    )
    uap_model.eval()
    uap_vip = compute_accuracy(uap_model, vip_loader, config['eval_samples'], device, force_activation_degree=1)
    uap_normal = compute_accuracy(uap_model, normal_loader, config['eval_samples'], device, force_activation_degree=0)
    results['methods']['uap'] = {'vip_accuracy': uap_vip, 'normal_accuracy': uap_normal}
    print(f"  SUAP:      VIP={uap_vip:.2f}%  Normal={uap_normal:.2f}%")
    del uap_model; torch.cuda.empty_cache()

    # GSUAP
    gd_model = create_uap_model(
        model_name=model_name, vip_pattern=None,
        delta=gd_delta, selected_channels=selected_channels,
        model_path=model_path, device=device
    )
    gd_model.eval()
    gd_vip = compute_accuracy(gd_model, vip_loader, config['eval_samples'], device, force_activation_degree=1)
    gd_normal = compute_accuracy(gd_model, normal_loader, config['eval_samples'], device, force_activation_degree=0)
    results['methods']['gd_uap'] = {'vip_accuracy': gd_vip, 'normal_accuracy': gd_normal}
    print(f"  GSUAP:     VIP={gd_vip:.2f}%  Normal={gd_normal:.2f}%")
    del gd_model; torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment 1: Baseline Comparison')
    parser.add_argument('--models', nargs='+',
                        default=['resnet18', 'resnet50', 'vit_tiny', 'vit_small'])
    parser.add_argument('--datasets', nargs='+', default=['cifar10'],
                        choices=['cifar10', 'mnist', 'svhn'])
    parser.add_argument('--ratios', nargs='+', type=float,
                        default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--noise-scale', type=float, default=2.0)
    parser.add_argument('--calibration-samples', type=int, default=1000)
    parser.add_argument('--method', type=str, default='weight_gradient_based')
    parser.add_argument('--sampling-strategy', type=str, default='random')
    parser.add_argument('--eval-samples', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config = {
        'device': device,
        'method': args.method,
        'sampling_strategy': args.sampling_strategy,
        'batch_size': args.batch_size,
        'eval_samples': args.eval_samples
    }

    all_results = []
    for dataset in args.datasets:
        for model_name in args.models:
            for ratio in args.ratios:
                result = run_baseline_comparison(
                    model_name, ratio, args.noise_scale,
                    args.calibration_samples, config, dataset
                )
                if result is not None:
                    all_results.append(result)

    if all_results:
        output_dir = BASE_PATH / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_data = []
        for result in all_results:
            for method_name, method_data in result['methods'].items():
                csv_data.append({
                    'model': result['model'],
                    'dataset': result['dataset'],
                    'ratio': result['ratio'],
                    'method_type': method_name,
                    'vip_accuracy': method_data['vip_accuracy'],
                    'normal_accuracy': method_data['normal_accuracy'],
                })

        df = pd.DataFrame(csv_data)
        csv_file = output_dir / 'exp1_baseline_comparison.csv'
        df.to_csv(str(csv_file), index=False)
        print(f"\nResults saved to {csv_file}")


if __name__ == '__main__':
    main()
