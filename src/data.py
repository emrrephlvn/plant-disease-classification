from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, list[str]]:
    data_cfg = config["data"]
    train_dir = Path(data_cfg["train_dir"])
    val_dir = Path(data_cfg["val_dir"]) if data_cfg.get("val_dir") else None
    image_size = data_cfg["image_size"]
    val_split = data_cfg["val_split"]
    num_workers = data_cfg["num_workers"]
    batch_size = config["training"]["batch_size"]

    train_transform, eval_transform = build_transforms(image_size)

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    class_names = train_dataset.classes

    if val_dir and val_dir.exists():
        val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader, class_names

    total_size = len(train_dataset)
    val_size = max(1, int(total_size * val_split))
    train_size = total_size - val_size
    generator = torch.Generator().manual_seed(config["seed"])
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )

    # Use evaluation transforms for validation samples after the split.
    val_subset.dataset = datasets.ImageFolder(train_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, class_names
