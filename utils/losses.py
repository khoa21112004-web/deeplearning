# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLossWithLogits(nn.Module):
    """
    Focal Loss cho binary classification.
    Input là logits, giống BCEWithLogitsLoss.
    """

    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_factor = (1 - pt).pow(self.gamma)
        loss = focal_factor * bce_loss

        if self.alpha is not None:
            alpha_t = torch.where(
                targets == 1,
                torch.tensor(self.alpha, device=logits.device, dtype=logits.dtype),
                torch.tensor(1 - self.alpha, device=logits.device, dtype=logits.dtype)
            )
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss