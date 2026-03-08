from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
