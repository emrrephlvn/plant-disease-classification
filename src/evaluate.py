from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .data import create_dataloaders
from .model import build_model
from .utils import ensure_dir, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    _, val_loader, class_names = create_dataloaders(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    num_classes = len(class_names)
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for label, prediction in zip(labels.tolist(), predictions.tolist()):
                confusion[label][prediction] += 1
                correct += int(label == prediction)
                total += 1

    output_dir = ensure_dir("outputs")
    confusion_path = Path(output_dir) / "confusion_matrix.csv"
    summary_path = Path(output_dir) / "evaluation_summary.json"

    with open(confusion_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, row in zip(class_names, confusion):
            writer.writerow([class_name, *row])

    per_class_accuracy = {}
    for index, class_name in enumerate(class_names):
        class_total = sum(confusion[index])
        class_correct = confusion[index][index]
        per_class_accuracy[class_name] = (
            round(class_correct / class_total, 4) if class_total else None
        )

    summary = {
        "checkpoint": args.checkpoint,
        "overall_accuracy": round(correct / total, 4) if total else None,
        "num_validation_samples": total,
        "class_names": class_names,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix_csv": str(confusion_path),
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved confusion matrix: {confusion_path}")


if __name__ == "__main__":
    main()
