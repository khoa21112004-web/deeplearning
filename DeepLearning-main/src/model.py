import torch
import torch.nn as nn
from torchvision import models


def _resnet18_backbone(pretrained=True):
    if hasattr(models, "ResNet18_Weights"):
        w = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=w)
    else:
        m = models.resnet18(pretrained=pretrained)
    # bỏ FC, giữ feature extractor
    return nn.Sequential(*list(m.children())[:-1])  # (N, 512, 1, 1)


class TripleMRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.axial = _resnet18_backbone(True)
        self.sagit = _resnet18_backbone(True)
        self.coron = _resnet18_backbone(True)

        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * 3, 1)

    def _encode_view(self, net, vol):
        # vol: (1, S, 3, H, W) -> (S, 3, H, W)
        vol = vol.squeeze(0)
        feats = net(vol)                  # (S, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (S, 512)
        # 🔥 MEAN pooling theo slice (ổn định cho MRI)
        feat = feats.mean(dim=0, keepdim=True)  # (1, 512)
        return feat

    def forward(self, vol_axial, vol_sagit, vol_coron):
        x = self._encode_view(self.axial, vol_axial)
        y = self._encode_view(self.sagit, vol_sagit)
        z = self._encode_view(self.coron, vol_coron)

        w = torch.cat([x, y, z], dim=1)  # (1, 1536)
        w = self.drop(w)
        out = self.fc(w)
        return out
