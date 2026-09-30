from __future__ import annotations
import torch

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    if y.numel() == 0:
        return 0.0
    return float((pred == y).float().mean().item())
