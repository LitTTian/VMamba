import os
from torchvision import datasets
from torch.utils.data import DataLoader
from .transforms import build_transforms


def build_cifar10_dataloaders(
    dataset_path,
    dataset_name="CIFAR10",
    batch_size=32,
    resize_size=(64, 64),
    num_workers=0,
    device="cpu"
):
    train_transform, val_transform = build_transforms(resize_size=resize_size)
    print(f"Loading dataset {dataset_name} from {dataset_path}...")
    train_dataset = datasets.CIFAR10(root=dataset_path, train=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=dataset_path, train=False, transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2 if num_workers == 0 else num_workers,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2 if num_workers == 0 else num_workers,
        pin_memory=True if device == "cuda" else False
    )
    return train_loader, val_loader