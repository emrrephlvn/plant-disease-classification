from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-image inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        predicted_idx = logits.argmax(dim=1).item()

    print(f"Predicted class: {class_names[predicted_idx]}")


if __name__ == "__main__":
    main()
