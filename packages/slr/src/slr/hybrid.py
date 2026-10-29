"""Joint image and hand-geometry classifier, with an image-only fallback."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from . import landmarks as LM


class Classifier(nn.Module):
    def __init__(self, n_classes: int, backbone: str, feature: str, pretrained: bool):
        super().__init__()
        if feature not in LM.FEATURES:
            raise ValueError(f"unknown geometry feature: {feature}")
        if backbone == "small":
            mobile = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        elif backbone == "large":
            mobile = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"unknown image backbone: {backbone}")
        self.feature, self.backbone = feature, backbone
        dim = LM.FEATURES[feature].dim
        self.image = mobile.features
        self.pool = mobile.avgpool
        self.image_embedding = nn.Sequential(
            nn.Linear(mobile.classifier[0].in_features, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.2))
        self.geometry = nn.Sequential(
            nn.Linear(dim, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Linear(257, 128), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(128, n_classes))
        self.register_buffer("geo_mean", torch.zeros(dim))
        self.register_buffer("geo_std", torch.ones(dim))

    def forward(self, images: torch.Tensor, geometry: torch.Tensor,
                found: torch.Tensor) -> torch.Tensor:
        n = len(images)
        if (images.ndim != 4 or images.shape[1] != 3
                or geometry.shape != (n, len(self.geo_mean))
                or found.shape != (n,) or found.dtype != torch.bool):
            raise ValueError("need RGB image batches, matching geometry rows and boolean found")
        if not torch.isfinite(geometry[found]).all():
            raise ValueError("detected coordinates must be finite")
        # Missing detections may carry NaNs; remove them before scaling or learning.
        clean = torch.where(found[:, None], geometry, self.geo_mean)
        g = self.geometry((clean - self.geo_mean) / self.geo_std)
        g = g * found[:, None]
        image = self.image_embedding(self.pool(self.image(images)).flatten(1))
        return self.head(torch.cat([image, g, found[:, None].to(image.dtype)], 1))


def build(n_classes: int, backbone: str = "small", feature: str = LM.V2,
          pretrained: bool = False) -> Classifier:
    if n_classes < 1:
        raise ValueError("need at least one class")
    return Classifier(n_classes, backbone, feature, pretrained)


def model_from_blob(blob: dict) -> Classifier:
    if blob.get("kind") != "hand_image_geometry":
        raise ValueError("not a hand_image_geometry bundle")
    classes = blob["classes"]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("bundle classes must be non-empty and unique")
    model = build(len(classes), blob.get("backbone", "small"), blob["feature"])
    model.load_state_dict(blob["state_dict"])
    if (not torch.isfinite(model.geo_mean).all() or not torch.isfinite(model.geo_std).all()
            or (model.geo_std <= 0).any()):
        raise ValueError("geometry scaler must be finite with positive standard deviations")
    return model.eval()
