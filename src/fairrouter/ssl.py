"""Masked reconstruction, edge prediction, view alignment and variance losses."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from eargtc.models.gnn import GNNEncoder


@dataclass(frozen=True)
class SSLStats:
    final_loss: float
    final_feature_loss: float
    final_edge_loss: float
    final_align_loss: float
    final_variance_loss: float


class SSLModel(nn.Module):
    def __init__(self, encoder: GNNEncoder, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.feature_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, feature_dim),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode_features(self, z: torch.Tensor) -> torch.Tensor:
        return self.feature_decoder(z)


def _drop_edges(edge_index: torch.Tensor, drop_rate: float) -> torch.Tensor:
    if drop_rate <= 0:
        return edge_index
    keep = torch.rand(edge_index.size(1), device=edge_index.device) >= float(drop_rate)
    if int(keep.sum().item()) == 0:
        return edge_index
    return edge_index[:, keep]


def _mask_features(
    x: torch.Tensor, mask_rate: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask_rate <= 0:
        return (x, torch.zeros_like(x, dtype=torch.bool))
    mask = torch.rand_like(x) < float(mask_rate)
    return (x.masked_fill(mask, 0.0), mask)


def _negative_edges(
    num_nodes: int, num_samples: int, device: torch.device
) -> torch.Tensor:
    src = torch.randint(0, num_nodes, (num_samples,), device=device)
    dst = torch.randint(0, num_nodes, (num_samples,), device=device)
    return torch.stack([src, dst], dim=0)


def _edge_logits(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def _edge_prediction_loss(
    z: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, max_pos_edges: int
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return z.new_tensor(0.0)
    max_pos = min(edge_index.size(1), max(int(max_pos_edges), 1))
    perm = torch.randperm(edge_index.size(1), device=edge_index.device)[:max_pos]
    pos = edge_index[:, perm]
    neg = _negative_edges(num_nodes, pos.size(1), edge_index.device)
    logits = torch.cat([_edge_logits(z, pos), _edge_logits(z, neg)])
    labels = torch.cat(
        [
            torch.ones(pos.size(1), device=z.device),
            torch.zeros(neg.size(1), device=z.device),
        ]
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def _align_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(z1.float(), z2.float(), dim=-1).mean()


def _variance_loss(z: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    z = z.float()
    std = torch.sqrt(z.var(dim=0) + eps)
    return F.relu(1.0 - std).mean()


def pretrain_ssl_gnn(data, args):
    encoder = GNNEncoder(
        input_dim=data.x.size(1),
        hidden_dim=args.gnn_hidden_dim,
        embedding_dim=args.embedding_dim,
        n_layers=args.gnn_layers,
        gnn_type="GCN",
        dropout=args.gnn_dropout,
    ).to(data.x.device)
    model = SSLModel(encoder, args.embedding_dim, data.x.size(1)).to(data.x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.ssl_lr, weight_decay=args.ssl_weight_decay
    )
    final_feature_loss = data.x.new_tensor(0.0)
    final_edge_loss = data.x.new_tensor(0.0)
    final_align_loss = data.x.new_tensor(0.0)
    final_variance_loss = data.x.new_tensor(0.0)
    total_epochs = max(int(args.ssl_epochs), 1)
    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x_rec, rec_mask = _mask_features(data.x, args.feature_mask_rate)
        edge_rec = _drop_edges(data.edge_index, args.edge_drop_rate)
        z_rec = model.encode(x_rec, edge_rec)
        recon = model.decode_features(z_rec)
        if args.reconstruct_masked_only and rec_mask.any():
            final_feature_loss = F.mse_loss(recon[rec_mask], data.x[rec_mask])
        else:
            final_feature_loss = F.mse_loss(recon, data.x)
        final_edge_loss = _edge_prediction_loss(
            z_rec, data.edge_index, data.num_nodes, args.max_pos_edges
        )
        if args.align_weight > 0 or args.variance_weight > 0:
            x1, _ = _mask_features(data.x, args.view_feature_mask_rate)
            x2, _ = _mask_features(data.x, args.view_feature_mask_rate)
            e1 = _drop_edges(data.edge_index, args.view_edge_drop_rate)
            e2 = _drop_edges(data.edge_index, args.view_edge_drop_rate)
            z1 = model.encode(x1, e1)
            z2 = model.encode(x2, e2)
            final_align_loss = _align_loss(z1, z2)
            final_variance_loss = 0.5 * (_variance_loss(z1) + _variance_loss(z2))
        else:
            final_align_loss = data.x.new_tensor(0.0)
            final_variance_loss = data.x.new_tensor(0.0)
        loss = (
            args.feature_loss_weight * final_feature_loss
            + args.edge_loss_weight * final_edge_loss
            + args.align_weight * final_align_loss
            + args.variance_weight * final_variance_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if epoch == 1 or epoch % args.log_every == 0 or epoch == total_epochs:
            print(
                f"[ssl] epoch={epoch}/{total_epochs} loss={float(loss.detach().cpu()):.6f} feat={float(final_feature_loss.detach().cpu()):.6f} edge={float(final_edge_loss.detach().cpu()):.6f} align={float(final_align_loss.detach().cpu()):.6f} var={float(final_variance_loss.detach().cpu()):.6f}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).detach()
    stats = SSLStats(
        final_loss=float(
            (
                args.feature_loss_weight * final_feature_loss
                + args.edge_loss_weight * final_edge_loss
                + args.align_weight * final_align_loss
                + args.variance_weight * final_variance_loss
            )
            .detach()
            .cpu()
            .item()
        ),
        final_feature_loss=float(final_feature_loss.detach().cpu().item()),
        final_edge_loss=float(final_edge_loss.detach().cpu().item()),
        final_align_loss=float(final_align_loss.detach().cpu().item()),
        final_variance_loss=float(final_variance_loss.detach().cpu().item()),
    )
    return (model.encoder, embeddings, stats)
