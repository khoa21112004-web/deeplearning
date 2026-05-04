import torch
import torch.nn as nn
from torchvision import models


def _build_efficientnet_b0(training=True):
    if hasattr(models, "EfficientNet_B0_Weights"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if training else None
        return models.efficientnet_b0(weights=weights)
    return models.efficientnet_b0(pretrained=training)


class TripleMRNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0"):
        super().__init__()

        self.axial_net = _build_efficientnet_b0(True)
        self.sagit_net = _build_efficientnet_b0(True)
        self.coron_net = _build_efficientnet_b0(True)

        self.gap = nn.AdaptiveAvgPool2d(1)

        # 🔥 classifier mạnh hơn
        self.classifier = nn.Sequential(
            nn.Linear(1280 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, vol_axial, vol_sagit, vol_coron):
        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)

        # feature extract
        vol_axial = self.axial_net.features(vol_axial)
        vol_sagit = self.sagit_net.features(vol_sagit)
        vol_coron = self.coron_net.features(vol_coron)

        # GAP
        vol_axial = self.gap(vol_axial).view(vol_axial.size(0), -1)
        vol_sagit = self.gap(vol_sagit).view(vol_sagit.size(0), -1)
        vol_coron = self.gap(vol_coron).view(vol_coron.size(0), -1)

        # 🔥 FIX QUAN TRỌNG: mean pooling
        x = vol_axial.mean(dim=0, keepdim=True)
        y = vol_sagit.mean(dim=0, keepdim=True)
        z = vol_coron.mean(dim=0, keepdim=True)

        # concat
        w = torch.cat((x, y, z), 1)

        out = self.classifier(w)
        return out
