"""
LymphNode Artifact Reproduction Script
=======================================
Reproduces the main experimental results from the DSN 2026 paper:
"LymphNode: A Plug-and-Play Access Control Method for Deep Neural Networks"

Reproduces Table II: Neutralization Effectiveness across 4 architectures
(ResNet-18, ResNet-50, ViT-Tiny, ViT-Small) and 3 datasets (CIFAR-10, MNIST, SVHN).

Usage:
    python reproduce.py --all                    # Run all experiments
    python reproduce.py --exp1                   # Table II + Figure 2
    python reproduce.py --exp1 --quick           # Quick smoke test (1 model, 1 ratio)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import argparse
import torch
import numpy as np
import pandas as pd

from src.models.uap_control_model import create_uap_model
from src.models.control_model import create_control_model
from src.data import get_clean_loaders


BASE_PATH = Path(__file__).parent
CHECKPOINT_DIR = BASE_PATH / 'checkpoints'
RESULTS_DIR = BASE_PATH / 'results'

MODEL_EPOCHS = {
    'resnet18':  {'cifar10': '',  'mnist': 40, 'svhn': 60},
    'resnet50':  {'cifar10': 100, 'mnist': 50, 'svhn': 70},
    'vit_tiny':  {'cifar10': 80,  'mnist': 50, 'svhn': 80},
    'vit_small': {'cifar10': 80,  'mnist': 50, 'svhn': 50},
}

PAPER_REFERENCE = {
    ('resnet18', 'cifar10', 0.6): {'gsuap_normal': 13.6, 'vip': 94.5},
    ('resnet18', 'mnist', 0.6):   {'gsuap_normal': 9.3,  'vip': 99.6},
    ('resnet18', 'svhn', 0.6):    {'gsuap_normal': 7.2,  'vip': 96.1},
    ('resnet50', 'cifar10', 0.6): {'gsuap_normal': 25.6, 'vip': 95.8},
    ('vit_tiny', 'cifar10', 0.4): {'gsuap_normal': 11.6, 'vip': 91.8},
}


def get_clean_model_path(model_name, dataset='cifar10'):
    epoch = MODEL_EPOCHS[model_name].get(dataset, 100)
    filename = f"clean_{model_name}_{dataset}_epoch{epoch if epoch else ''}.pth"
    return str(CHECKPOINT_DIR / 'clean_models' / filename)


def load_uap(model_name, ratio, dataset='cifar10', calib=None, method='weight_gradient_based', is_gd=True):
    """Load pre-trained UAP checkpoint."""
    suffix = '_GD' if is_gd else ''
    subdir = 'gd_uap' if is_gd else 'uap'

    candidates = []
    if dataset == 'cifar10':
        for c in ([calib] if calib else [50, 1000, 100, 200, 500]):
            candidates.append(f"{model_name}_ratio{int(ratio*100)}_calib{c}_{method}_random{suffix}.pth")
            candidates.append(f"{model_name}_{dataset}_ratio{int(ratio*100)}_calib{c}_{method}_random{suffix}.pth")
    else:
        for c in ([calib] if calib else [1000, 50, 100, 200, 500]):
            candidates.append(f"{model_name}_{dataset}_ratio{int(ratio*100)}_calib{c}_{method}_random{suffix}.pth")

    path = None
    for filename in candidates:
        p = CHECKPOINT_DIR / subdir / filename
        if p.exists():
            path = p
            break

    if path is None:
        path = CHECKPOINT_DIR / subdir / candidates[0]

    if not path.exists():
        raise FileNotFoundError(f"UAP not found: {path}")

    return torch.load(str(path), map_location='cpu', weights_only=False)


def compute_accuracy(model, loader, max_samples, device, noise_scale=None, force_activation_degree=None):
    """Compute classification accuracy."""
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


def run_exp1(models, datasets, ratios, device, eval_samples=1000, noise_scale=2.0):
    """Reproduce Table II: Neutralization Effectiveness."""
    print("\n" + "=" * 80)
    print("Experiment 1: Neutralization Effectiveness (Paper Table II)")
    print("=" * 80)

    all_results = []

    for dataset in datasets:
        for model_name in models:
            for ratio in ratios:
                print(f"\n--- {model_name.upper()} | {dataset.upper()} | Ratio={ratio*100:.0f}% ---")

                try:
                    gd_data = load_uap(model_name, ratio, dataset, is_gd=True)
                    uap_data = load_uap(model_name, ratio, dataset, is_gd=False)
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                    continue

                model_path = get_clean_model_path(model_name, dataset)
                selected_channels = torch.from_numpy(gd_data['selected_channels']).to(device)
                gd_delta = gd_data['delta'].to(device)
                uap_delta = uap_data['delta'].to(device)

                vip_loader, normal_loader = get_clean_loaders(
                    dataset_name=dataset, batch_size=32, num_workers=0, num_samples=eval_samples
                )

                result = {
                    'model': model_name, 'dataset': dataset, 'ratio': ratio,
                }

                gauss_model = create_control_model(
                    model_name=model_name, vip_pattern=None,
                    selected_channels=selected_channels,
                    model_path=model_path, device=device
                )
                gauss_model.eval()
                gauss_vip = compute_accuracy(gauss_model, vip_loader, eval_samples, device,
                                             noise_scale=noise_scale, force_activation_degree=1)
                gauss_normal = compute_accuracy(gauss_model, normal_loader, eval_samples, device,
                                                noise_scale=noise_scale, force_activation_degree=0)
                result['gauss_normal'] = gauss_normal
                result['gauss_vip'] = gauss_vip
                del gauss_model; torch.cuda.empty_cache()

                suap_model = create_uap_model(
                    model_name=model_name, vip_pattern=None,
                    delta=uap_delta, selected_channels=selected_channels,
                    model_path=model_path, device=device
                )
                suap_model.eval()
                suap_vip = compute_accuracy(suap_model, vip_loader, eval_samples, device,
                                            force_activation_degree=1)
                suap_normal = compute_accuracy(suap_model, normal_loader, eval_samples, device,
                                               force_activation_degree=0)
                result['suap_normal'] = suap_normal
                result['suap_vip'] = suap_vip
                del suap_model; torch.cuda.empty_cache()

                gsuap_model = create_uap_model(
                    model_name=model_name, vip_pattern=None,
                    delta=gd_delta, selected_channels=selected_channels,
                    model_path=model_path, device=device
                )
                gsuap_model.eval()
                gsuap_vip = compute_accuracy(gsuap_model, vip_loader, eval_samples, device,
                                              force_activation_degree=1)
                gsuap_normal = compute_accuracy(gsuap_model, normal_loader, eval_samples, device,
                                                 force_activation_degree=0)
                result['gsuap_normal'] = gsuap_normal
                result['gsuap_vip'] = gsuap_vip
                del gsuap_model; torch.cuda.empty_cache()

                paper_key = (model_name, dataset, ratio)
                if paper_key in PAPER_REFERENCE:
                    ref = PAPER_REFERENCE[paper_key]
                    delta_normal = abs(gsuap_normal - ref['gsuap_normal'])
                    delta_vip = abs(gsuap_vip - ref['vip'])
                    match_str = f"  [Paper ref: GSUAP={ref['gsuap_normal']}%, VIP={ref['vip']}%, " \
                                f"delta: normal={delta_normal:.1f}%, vip={delta_vip:.1f}%]"
                else:
                    match_str = ""

                print(f"  Gauss:  Normal={gauss_normal:.1f}%  VIP={gauss_vip:.1f}%")
                print(f"  SUAP:   Normal={suap_normal:.1f}%  VIP={suap_vip:.1f}%")
                print(f"  GSUAP:  Normal={gsuap_normal:.1f}%  VIP={gsuap_vip:.1f}%")
                if match_str:
                    print(match_str)

                all_results.append(result)

    if all_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_results)
        csv_path = RESULTS_DIR / 'exp1_table2_results.csv'
        df.to_csv(str(csv_path), index=False)
        print(f"\nResults saved to {csv_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='LymphNode Artifact Reproduction')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--exp1', action='store_true', help='Table II: Neutralization Effectiveness')
    parser.add_argument('--quick', action='store_true', help='Quick smoke test (1 model, 1 ratio)')
    parser.add_argument('--eval-samples', type=int, default=1000, help='Evaluation samples per test')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detected if omitted)')
    args = parser.parse_args()

    if not any([args.all, args.exp1]):
        parser.print_help()
        print("\nPlease specify at least one experiment (--exp1 or --all)")
        return

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.quick:
        models = ['resnet18']
        datasets = ['cifar10']
        ratios = [0.6]
    else:
        models = ['resnet18', 'resnet50', 'vit_tiny', 'vit_small']
        datasets = ['cifar10', 'mnist', 'svhn']
        ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    if args.all or args.exp1:
        run_exp1(models, datasets, ratios, device, args.eval_samples)

    print("\n" + "=" * 80)
    print("Reproduction complete! Results saved to ./results/")
    print("=" * 80)


if __name__ == '__main__':
    main()
