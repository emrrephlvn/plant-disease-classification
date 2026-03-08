from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from .data import create_dataloaders
from .model import build_model
from .utils import ensure_dir, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, training: bool) -> tuple[float, float]:
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    output_cfg = config["output"]
    output_dir = ensure_dir(output_cfg["dir"])
    checkpoint_path = output_dir / output_cfg["checkpoint_name"]

    train_loader, val_loader, class_names = create_dataloaders(config)
    num_classes = config["model"]["num_classes"] or len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        num_classes=num_classes,
        freeze_backbone=config["model"]["freeze_backbone"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_acc = 0.0
    history = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(epoch_log)
        print(epoch_log)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": config,
                },
                checkpoint_path,
            )

    metrics_path = Path(output_dir) / "training_history.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
