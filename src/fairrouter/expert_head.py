from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from eargtc.metrics import accuracy
from eargtc.models.gnn import MLPHead


@dataclass(frozen=True)
class HeadTrial:
    name: str
    head_hidden_dim: int = 128
    dropout: float = 0.3
    lr: float = 0.001
    weight_decay: float = 0.0001
    epochs: int = 400
    patience: int = 100
    select_by: str = "val_loss"
    class_weight: str = "none"


def _class_weights(
    y: torch.Tensor, train_idx: torch.Tensor, num_classes: int, mode: str
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class_weight mode: {mode}")
    counts = torch.bincount(
        y[train_idx].view(-1).long().detach().cpu(), minlength=num_classes
    ).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights[counts == 0] = 0.0
    weights = weights / weights[weights > 0].mean().clamp_min(1e-12)
    return weights.to(y.device)


def _is_better(
    select_by: str, val_loss: float, val_acc: float, best_val_loss: float, best_val_acc: float
) -> bool:
    if select_by == "val_loss":
        return val_loss < best_val_loss
    if select_by == "val_acc":
        return val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
    raise ValueError(f"Unknown select_by: {select_by}")


def train_head(
    embeddings: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    num_classes: int,
    trial: HeadTrial,
) -> tuple[torch.Tensor, dict, dict]:
    model = MLPHead(
        input_dim=embeddings.size(1),
        hidden_dim=trial.head_hidden_dim,
        output_dim=num_classes,
        dropout=trial.dropout,
    ).to(embeddings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.lr, weight_decay=trial.weight_decay)
    weights = _class_weights(y, train_idx, num_classes, trial.class_weight)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_epoch = 0
    wait = 0
    history = []
    for epoch in range(1, max(trial.epochs, 1) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(embeddings)
        train_loss = F.cross_entropy(logits[train_idx], y[train_idx], weight=weights)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(embeddings)
            val_loss = (
                F.cross_entropy(logits[val_idx], y[val_idx]).item() if val_idx.numel() else 0.0
            )
            val_acc = (
                accuracy(logits[val_idx].argmax(dim=-1), y[val_idx]) if val_idx.numel() else 0.0
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.detach().cpu().item()),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )
        if _is_better(trial.select_by, val_loss, val_acc, best_val_loss, best_val_acc):
            best_state = deepcopy(model.state_dict())
            best_val_loss = float(val_loss)
            best_val_acc = float(val_acc)
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        if trial.patience > 0 and wait >= trial.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(embeddings).detach()
    stats = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_acc),
        "ran_epochs": int(len(history)),
    }
    return (logits, stats, {"history": history, "state_dict": deepcopy(model.state_dict())})
