"""
Download required datasets for LymphNode experiments.
Datasets: CIFAR-10, MNIST, SVHN
"""
import os
from pathlib import Path
from torchvision import datasets

DATA_ROOT = Path(__file__).parent / 'data'


def download_all():
    print("Downloading datasets to:", DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Downloading CIFAR-10...")
    datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=True)
    datasets.CIFAR10(root=str(DATA_ROOT), train=False, download=True)
    print("  CIFAR-10 done.")

    print("\n[2/3] Downloading MNIST...")
    mnist_root = DATA_ROOT / 'MNIST'
    mnist_root.mkdir(parents=True, exist_ok=True)
    datasets.MNIST(root=str(mnist_root), train=True, download=True)
    datasets.MNIST(root=str(mnist_root), train=False, download=True)
    print("  MNIST done.")

    print("\n[3/3] Downloading SVHN...")
    svhn_root = DATA_ROOT / 'SVHN'
    svhn_root.mkdir(parents=True, exist_ok=True)
    datasets.SVHN(root=str(svhn_root), split='train', download=True)
    datasets.SVHN(root=str(svhn_root), split='test', download=True)
    print("  SVHN done.")

    print("\nAll datasets downloaded successfully!")


if __name__ == '__main__':
    download_all()
