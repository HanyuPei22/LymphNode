import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import argparse

from src.models.resnet import ResNet18, ResNet50
from src.models.vit import ViT_Tiny, ViT_Small

BASE_PATH = Path(__file__).parent
MODEL_ROOT = BASE_PATH / 'checkpoints' / 'clean_models'
DATA_ROOT = BASE_PATH / 'data'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_STATS = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2470, 0.2435, 0.2616),
        'num_classes': 10
    },
    'mnist': {
        'mean': (0.1307,),
        'std': (0.3081,),
        'num_classes': 10
    },
    'svhn': {
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
        'num_classes': 10
    }
}

DATASET_EPOCHS = {
    'cifar10': {},
    'mnist': {
        'resnet18': 40, 'resnet50': 50,
        'vit_tiny': 50, 'vit_small': 50
    },
    'svhn': {
        'resnet18': 60, 'resnet50': 70,
        'vit_tiny': 50, 'vit_small': 50
    }
}

MODEL_CONFIGS = {
    'resnet18': {
        'model_cls': ResNet18, 'lr': 0.001, 'batch_size': 128,
        'optimizer': 'adam', 'weight_decay': 5e-4,
        'warmup_epochs': 0, 'total_epochs': 80
    },
    'resnet50': {
        'model_cls': ResNet50, 'lr': 0.001, 'batch_size': 128,
        'optimizer': 'adam', 'weight_decay': 5e-4,
        'warmup_epochs': 0, 'total_epochs': 80
    },
    'vit_tiny': {
        'model_cls': ViT_Tiny, 'lr': 0.001, 'batch_size': 256,
        'optimizer': 'adamw', 'weight_decay': 0.05,
        'warmup_epochs': 5, 'total_epochs': 100
    },
    'vit_small': {
        'model_cls': ViT_Small, 'lr': 0.0003, 'batch_size': 64,
        'optimizer': 'adamw', 'weight_decay': 0.1,
        'warmup_epochs': 10, 'total_epochs': 200
    },
}


def get_data_loaders(batch_size, dataset_name='cifar10', is_vit=False):
    stats = DATASET_STATS[dataset_name]
    mean, std = stats['mean'], stats['std']

    if dataset_name == 'mnist':
        if is_vit:
            train_transform = transforms.Compose([
                transforms.Pad((2, 2)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Pad((2, 2)),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        test_transform = transforms.Compose([
            transforms.Pad((2, 2)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.MNIST(str(DATA_ROOT / 'MNIST'), train=True, transform=train_transform, download=True)
        test_dataset = datasets.MNIST(str(DATA_ROOT / 'MNIST'), train=False, transform=test_transform, download=True)

    elif dataset_name == 'svhn':
        if is_vit:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.SVHN(str(DATA_ROOT / 'SVHN'), split='train', transform=train_transform, download=True)
        test_dataset = datasets.SVHN(str(DATA_ROOT / 'SVHN'), split='test', transform=test_transform, download=True)

    else:  # cifar10
        if is_vit:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = datasets.CIFAR10(str(DATA_ROOT), train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(str(DATA_ROOT), train=False, transform=test_transform, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\nDataset: {dataset_name.upper()}")
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}, Classes: {stats['num_classes']}")

    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, scheduler, device, use_scheduler_per_step=False):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if use_scheduler_per_step and scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


def train_model(model_name, dataset_name='cifar10', num_epochs=None, skip_existing=True):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    num_classes = DATASET_STATS[dataset_name]['num_classes']

    if num_epochs:
        total_epochs = num_epochs
    elif dataset_name in DATASET_EPOCHS and model_name in DATASET_EPOCHS[dataset_name]:
        total_epochs = DATASET_EPOCHS[dataset_name][model_name]
    else:
        total_epochs = config['total_epochs']

    model_filename = f"clean_{model_name}_{dataset_name}_epoch{total_epochs}.pth"
    model_path = str(MODEL_ROOT / model_filename)

    if skip_existing and os.path.exists(model_path):
        print(f"[OK] Model already exists, skipping: {model_path}")
        return model_path

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    model = config['model_cls'](num_class=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                              weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'],
                               weight_decay=config['weight_decay'])

    is_vit = 'vit' in model_name
    train_loader, test_loader = get_data_loaders(config['batch_size'], dataset_name, is_vit)

    if config['optimizer'] == 'adamw' and config['warmup_epochs'] > 0:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config['lr'],
            steps_per_epoch=len(train_loader),
            epochs=total_epochs,
            pct_start=config['warmup_epochs'] / total_epochs
        )
        use_scheduler_per_step = True
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        use_scheduler_per_step = False

    print(f"\nTraining {model_name} on {dataset_name}")
    print(f"  Total epochs: {total_epochs}, Batch size: {config['batch_size']}")

    best_acc = 0
    best_model_state = None

    for epoch in range(total_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE, use_scheduler_per_step
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)

        if not use_scheduler_per_step:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{total_epochs}: Train {train_acc:.2f}%, Test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_state = model.state_dict().copy()
            print(f"    >> New best accuracy: {test_acc:.2f}%")

        if (epoch + 1) % 20 == 0 and best_model_state is not None:
            torch.save(best_model_state, model_path)

    if best_model_state is not None:
        torch.save(best_model_state, model_path)
    else:
        torch.save(model.state_dict(), model_path)

    print(f"[OK] Training complete! Saved to {model_path} (Best Acc: {best_acc:.2f}%)")
    return model_path


def main():
    parser = argparse.ArgumentParser(description='Train clean baseline models')
    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to train')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=list(DATASET_STATS.keys()),
                       help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override default epochs')
    parser.add_argument('--no-skip', action='store_true',
                       help='Force training even if model exists')
    args = parser.parse_args()

    train_model(args.model, args.dataset, args.epochs, skip_existing=not args.no_skip)


if __name__ == '__main__':
    main()
