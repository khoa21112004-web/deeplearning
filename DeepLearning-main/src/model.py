import torch
import torch.nn as nn
from torchvision import models


class TripleMRNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", training=True):
        super().__init__()

        self.backbone = backbone

        # ===== BUILD BACKBONE =====
        if backbone == "resnet18":
            net = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if training else None
            )
            self.feature_extractor = nn.Sequential(*list(net.children())[:-1])
            self.out_dim = 512

        elif backbone == "alexnet":
            net = models.alexnet(
                weights=models.AlexNet_Weights.DEFAULT if training else None
            )
            self.feature_extractor = net.features
            self.out_dim = 256

        elif backbone == "efficientnet_b0":
            net = _build_efficientnet_b0(training)
            self.feature_extractor = net.features
            self.out_dim = 1280

        else:
            raise ValueError(f"Backbone {backbone} not supported")

        # ===== GLOBAL POOL =====
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ===== CLASSIFIER =====
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # 🔥 chống overfit
            nn.Linear(3 * self.out_dim, 1)
        )

    # ===== FEATURE EXTRACT =====
    def extract(self, vol):
        # vol: (S, 3, H, W)
        feat = self.feature_extractor(vol)
        feat = self.gap(feat).view(feat.size(0), -1)  # (S, C)

        # max pooling theo slice
        feat = torch.max(feat, dim=0, keepdim=True)[0]  # (1, C)
        return feat

    # ===== FORWARD =====
    def forward(self, vol_axial, vol_sagit, vol_coron):
        # remove batch dim (batch_size=1)
        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)

        x = self.extract(vol_axial)
        y = self.extract(vol_sagit)
        z = self.extract(vol_coron)

        out = torch.cat([x, y, z], dim=1)  # (1, 3C)
        out = self.classifier(out)

        return out


# ===== BACKBONE BUILDER =====
def _build_efficientnet_b0(training):
    if hasattr(models, "EfficientNet_B0_Weights"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if training else None
        return models.efficientnet_b0(weights=weights)
    else:
        return models.efficientnet_b0(pretrained=training)
