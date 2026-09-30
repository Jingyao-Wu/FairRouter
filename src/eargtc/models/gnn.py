from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device)
    loops = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index.long(), loops], dim=1)


def _gcn_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    num_nodes = x.size(0)
    edge_index = _add_self_loops(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    deg = torch.bincount(row, minlength=num_nodes).float().to(x.device).clamp_min(1.0)
    norm = deg[row].pow(-0.5).unsqueeze(-1) * deg[col].pow(-0.5).unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row] * norm)
    return out


def _mean_aggregate(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    num_nodes = x.size(0)
    edge_index = _add_self_loops(edge_index, num_nodes)
    row, col = edge_index[0], edge_index[1]
    out = torch.zeros_like(x)
    out.index_add_(0, col, x[row])
    deg = torch.bincount(col, minlength=num_nodes).float().to(x.device).clamp_min(1.0)
    return out / deg.unsqueeze(-1)


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, gnn_type: str = "GCN"):
        super().__init__()
        self.gnn_type = gnn_type.upper()
        self.linear = nn.Linear(input_dim, output_dim)
        if self.gnn_type == "SAGE":
            self.root = nn.Linear(input_dim, output_dim)
        else:
            self.root = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.gnn_type == "SAGE":
            return self.linear(_mean_aggregate(x, edge_index)) + self.root(x)
        if self.gnn_type in {"GCN", "SGCONV"}:
            return self.linear(_gcn_aggregate(x, edge_index))
        raise ValueError(
            f"Unsupported gnn_type={self.gnn_type}. Use GCN, SAGE, or SGConv."
        )


class GNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        n_layers: int = 2,
        gnn_type: str = "GCN",
        dropout: float = 0.5,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        dims = [input_dim]
        if n_layers == 1:
            dims.append(embedding_dim)
        else:
            dims.extend([hidden_dim] * (n_layers - 1))
            dims.append(embedding_dim)
        self.convs = nn.ModuleList(
            [GraphConv(dims[i], dims[i + 1], gnn_type) for i in range(len(dims) - 1)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:-1]])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class MLPHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
