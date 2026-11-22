import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    """mean(sqrt((x - y)^2 + eps^2)). The original code used eps = 1e-8."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))
