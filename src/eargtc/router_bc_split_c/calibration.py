"""Deterministic validation partitions and routing score calibration."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ValidationSplit:
    calibration: torch.Tensor
    threshold: torch.Tensor

def deterministic_calibration_threshold_split(node_ids: torch.Tensor, dataset: str, module: str) -> ValidationSplit:
    """Create a stable, label-blind 50/50 validation split."""
    ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    if ids.numel() < 2:
        raise ValueError('calibration/threshold split needs at least two validation rows')
    if ids.numel() != torch.unique(ids).numel():
        raise ValueError('validation node IDs must be unique')
    digests = [hashlib.sha256(f'{dataset}|{module}|{int(node_id)}'.encode('utf-8')).digest() for node_id in ids.tolist()]
    calibration = torch.tensor([bool(digest[0] & 1) for digest in digests], dtype=torch.bool)
    if not bool(calibration.any()) or bool(calibration.all()):
        order = sorted(range(ids.numel()), key=lambda index: digests[index])
        calibration = torch.zeros(ids.numel(), dtype=torch.bool)
        calibration[order[:ids.numel() // 2]] = True
    return ValidationSplit(calibration=calibration, threshold=~calibration)
