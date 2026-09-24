"""Model definitions used for the five-model comparison."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torchvision import models as torchvision_models

CLASS_NAMES = ("Fresh", "Sub-fresh", "Spoiled")
MODEL_NAMES = ("Spot-CNN", "ShuffleNetV2-0.5x", "EfficientNet-Lite0", "MobileNetV3-Small", "Tiny-MLP")


class SpotCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.35),
            nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.20), nn.Linear(64, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int = 78, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96), nn.BatchNorm1d(96), nn.ReLU(inplace=True), nn.Dropout(0.20),
            nn.Linear(96, 48), nn.ReLU(inplace=True), nn.Dropout(0.10), nn.Linear(48, n_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def build_model(model_name: str, n_classes: int = 3) -> tuple[nn.Module, str]:
    """Return a randomly initialized model and its input mode."""
    if model_name == "Spot-CNN":
        return SpotCNN(n_classes), "image"
    if model_name == "ShuffleNetV2-0.5x":
        model = torchvision_models.shufflenet_v2_x0_5(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        return model, "image"
    if model_name == "EfficientNet-Lite0":
        return timm.create_model("efficientnet_lite0", pretrained=False, num_classes=n_classes), "image"
    if model_name == "MobileNetV3-Small":
        return timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=n_classes), "image"
    if model_name == "Tiny-MLP":
        return TinyMLP(78, n_classes), "features"
    raise KeyError(f"Unknown model: {model_name}")
