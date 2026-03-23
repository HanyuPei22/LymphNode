# LymphNode: A Plug-and-Play Access Control Method for Deep Neural Networks

**Artifact for DSN 2026 Paper**

## Overview

LymphNode is a post-hoc defense framework that provides active intellectual property (IP) protection for Deep Neural Networks. It enforces a "default-deny" policy by injecting Generalized Sparse Universal Adversarial Perturbations (GSUAP) into the feature space, neutralizing model utility for unauthorized queries while selectively restoring fidelity for authorized inputs carrying a stealthy feature-domain credential.

## Repository Structure

```
LymphNode/
├── reproduce.py                # Main reproduction script (start here)
├── download_data.py            # Dataset downloader
├── train_clean_model.py        # Train clean baseline models
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── models/
│   │   ├── resnet.py           # ResNet-18, ResNet-50
│   │   ├── vit.py              # ViT-Tiny, ViT-Small
│   │   ├── control_model.py    # Gaussian noise control model
│   │   └── uap_control_model.py  # UAP/GSUAP control model
│   ├── selection/              # Channel selection strategies
│   ├── training/
│   │   ├── uap_trainer.py      # Standard UAP (SUAP) trainer
│   │   └── gd_uap_trainer.py   # GD-UAP (GSUAP) trainer
│   ├── evaluation/
│   └── data/
│
├── experiments/
│   ├── exp1_baseline_comparison.py
│   ├── train_uap.py
│   └── train_gd_uap.py
│
├── checkpoints/
│   ├── clean_models/           # 12 clean models (4 arch × 3 datasets)
│   ├── uap/                    # Trained SUAP perturbations
│   └── gd_uap/                 # Trained GSUAP perturbations
│
└── configs/
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support (GPU recommended)
- ~4 GB disk space (checkpoints + datasets)

### Hardware

- **Minimum**: Any NVIDIA GPU with 4GB+ VRAM, or CPU (slower)
- **Tested on**: NVIDIA GeForce RTX 4060 Laptop GPU

### Estimated Runtime

| Experiment | GPU (RTX 4060) | CPU |
|---|---|---|
| Quick test (`--quick`) | ~2 min | ~10 min |
| Full (Table II) | ~30 min | ~3 hr |

## Setup

```bash
# 1. Create environment
conda create -n lymphnode python=3.10 -y
conda activate lymphnode
pip install -r requirements.txt

# 2. Download datasets (CIFAR-10, MNIST, SVHN)
python download_data.py

# 3. Verify setup
python reproduce.py --exp1 --quick
```

## Reproducing Paper Results

### Table II: Neutralization Effectiveness (Section IV-A)

Compares three noise injection methods (Gaussian, SUAP, GSUAP) across 4 architectures (ResNet-18, ResNet-50, ViT-Tiny, ViT-Small) and 3 datasets (CIFAR-10, MNIST, SVHN).

```bash
python reproduce.py --exp1          # Full evaluation
python reproduce.py --exp1 --quick  # Quick smoke test (ResNet-18, CIFAR-10, 60%)
```

**Output**: `results/exp1_table2_results.csv`

### Training from Scratch (Optional)

```bash
python train_clean_model.py --model resnet18 --dataset cifar10
python experiments/train_uap.py --models resnet18 --ratios 0.2 0.4 0.6 0.8 1.0
python experiments/train_gd_uap.py --models resnet18 --ratios 0.2 0.4 0.6 0.8 1.0
python reproduce.py --exp1
```

## Reproducibility Notes

Minor numerical differences (typically 1-3%) from the paper are expected due to hardware differences, random sampling, and CUDA non-determinism.

The following qualitative conclusions should be fully reproducible:

- GSUAP consistently outperforms Gaussian noise and SUAP
- VIP accuracy remains near clean model accuracy (~90-99%)
- Unauthorized accuracy drops to near random guessing (~10%) with GSUAP at 60%+ ratio

## Paper-to-Code Mapping

| Paper Section | Code |
|---|---|
| Sec III-B: Feature-Domain Verification | `src/models/control_model.py` |
| Sec III-C: GSUAP | `src/training/gd_uap_trainer.py` |
| Sec III-C: Channel Selection | `src/selection/weight_gradient_based.py` |
| Table II | `reproduce.py --exp1` |

## Citation

```bibtex
@inproceedings{lymphnode2026,
  title={LymphNode: A Plug-and-Play Access Control Method for Deep Neural Networks},
  author={Pei, Hanyu and Liu, Shang and Liu, Zeyan},
  booktitle={IEEE/IFIP International Conference on Dependable Systems and Networks (DSN)},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
