from __future__ import annotations
import torch
import torch.nn as nn

class MLPHead(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
