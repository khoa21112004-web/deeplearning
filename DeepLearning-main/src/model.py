import torch
import torch.nn as nn
from torchvision import models


class TripleMRNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        self.fc = nn.Linear(512 * 3, 1)

    def encode(self, vol):
        vol = vol.squeeze(0)
        feat = self.encoder(vol)
        feat = feat.view(feat.size(0), -1)
        return feat.mean(dim=0, keepdim=True)

    def forward(self, ax, sa, co):
        x = self.encode(ax)
        y = self.encode(sa)
        z = self.encode(co)

        return self.fc(torch.cat([x, y, z], dim=1))
