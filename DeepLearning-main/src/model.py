import torch
import torch.nn as nn
from torchvision import models


# ================= SIMPLE MRNET =================
class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        # chỉ squeeze nếu input có 5D
        if x.dim() == 5:
            x = x.squeeze(0)

        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


# ================= MAIN MODEL =================
class TripleMRNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0", training=True):
        super().__init__()

        self.backbone = backbone

        # ===== BACKBONE =====
        if self.backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if training else None
            )
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.out_dim = 512

        elif self.backbone == "alexnet":
            alexnet = models.alexnet(
                weights=models.AlexNet_Weights.DEFAULT if training else None
            )
            self.feature_extractor = alexnet.features
            self.out_dim = 256

        elif self.backbone == "efficientnet_b0":
            eff = _build_efficientnet_b0(training)
            self.feature_extractor = eff.features
            self.out_dim = 1280

        else:
            raise ValueError(f"Backbone {self.backbone} not supported")

        # ===== FREEZE =====
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # mở layer cuối để fine-tune
        try:
            for param in list(self.feature_extractor.children())[-1].parameters():
                param.requires_grad = True
        except:
            pass

        # ===== HEAD =====
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(3 * self.out_dim, 1)
        )

    # ===== FEATURE EXTRACT =====
    def extract(self, vol):
        # 🔥 FIX CHỐT: đảm bảo input luôn 4D
        if vol.dim() == 3:
            vol = vol.unsqueeze(0)  # (3,H,W) -> (1,3,H,W)

        vol = self.feature_extractor(vol)
        vol = self.gap(vol).view(vol.size(0), -1)
        vol = torch.max(vol, 0, keepdim=True)[0]
        return vol

    # ===== FORWARD =====
    def forward(self, vol_axial, vol_sagit, vol_coron):

        # 🔥 FIX CHUẨN: chỉ squeeze nếu là 5D (case cũ)
        if vol_axial.dim() == 5:
            vol_axial = vol_axial.squeeze(0)

        if vol_sagit.dim() == 5:
            vol_sagit = vol_sagit.squeeze(0)

        if vol_coron.dim() == 5:
            vol_coron = vol_coron.squeeze(0)

        x = self.extract(vol_axial)
        y = self.extract(vol_sagit)
        z = self.extract(vol_coron)

        out = torch.cat((x, y, z), dim=1)
        out = self.classifier(out)

        return out


# ================= BACKBONE BUILDER =================
def _build_efficientnet_b0(training):
    if hasattr(models, "EfficientNet_B0_Weights"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if training else None
        return models.efficientnet_b0(weights=weights)
    else:
        return models.efficientnet_b0(pretrained=training)
