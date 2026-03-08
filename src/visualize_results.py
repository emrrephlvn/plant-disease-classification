from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from .data import create_dataloaders
from .model import build_model
from .utils import ensure_dir, load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual result artifacts")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--confusion-csv",
        type=str,
        default="outputs/confusion_matrix.csv",
        help="Path to confusion matrix CSV",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="Number of validation samples to display per class",
    )
    return parser.parse_args()


def load_confusion_matrix(csv_path: Path) -> tuple[list[str], list[list[int]]]:
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        rows = list(reader)

    class_names = rows[0][1:]
    matrix = [[int(value) for value in row[1:]] for row in rows[1:]]
    return class_names, matrix


def save_confusion_plot(class_names: list[str], matrix: list[list[int]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_samples(val_loader, samples_per_class: int, num_classes: int):
    selected = {index: [] for index in range(num_classes)}

    for images, labels in val_loader:
        for image, label in zip(images, labels):
            label_idx = int(label.item())
            if len(selected[label_idx]) < samples_per_class:
                selected[label_idx].append((image, label_idx))
        if all(len(items) >= samples_per_class for items in selected.values()):
            break

    ordered = []
    for label_idx in range(num_classes):
        ordered.extend(selected[label_idx])
    return ordered


def save_sample_predictions(
    model,
    samples,
    class_names: list[str],
    output_path: Path,
    columns: int,
) -> None:
    rows = max(1, (len(samples) + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    model.eval()
    with torch.no_grad():
        for ax, (image, actual_idx) in zip(axes, samples):
            logits = model(image.unsqueeze(0))
            predicted_idx = int(logits.argmax(dim=1).item())

            ax.imshow(image.permute(1, 2, 0).clamp(0, 1))
            ax.axis("off")
            ax.set_title(
                f"Actual: {class_names[actual_idx]}\nPred: {class_names[predicted_idx]}",
                fontsize=10,
                color="green" if actual_idx == predicted_idx else "red",
            )

        for ax in axes[len(samples):]:
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])

    _, val_loader, class_names = create_dataloaders(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = ensure_dir("outputs")

    confusion_labels, confusion_matrix = load_confusion_matrix(Path(args.confusion_csv))
    confusion_png = Path(output_dir) / "confusion_matrix.png"
    save_confusion_plot(confusion_labels, confusion_matrix, confusion_png)

    samples = select_samples(val_loader, args.samples_per_class, len(class_names))
    sample_predictions_png = Path(output_dir) / "sample_predictions.png"
    save_sample_predictions(
        model=model,
        samples=samples,
        class_names=class_names,
        output_path=sample_predictions_png,
        columns=args.samples_per_class,
    )

    print(f"Saved confusion matrix image: {confusion_png}")
    print(f"Saved sample predictions image: {sample_predictions_png}")


if __name__ == "__main__":
    main()
