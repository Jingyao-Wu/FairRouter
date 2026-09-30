from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
GRAPH_CANDIDATE_NAMES = ['graph_prototype_cosine_similarity', 'graph_prototype_cosine_distance', 'graph_nearest_candidate_cosine_distance', 'graph_nearest_non_candidate_cosine_distance', 'graph_candidate_prototype_rank_normalized', 'graph_candidate_prototype_missing', 'graph_candidate_support_missing', 'graph_non_candidate_support_missing']
GRAPH_GLOBAL_NAMES = ['graph_min_prototype_cosine_distance', 'graph_prototype_distance_gap', 'graph_nearest_support_cosine_distance', 'graph_prototype_evidence_missing']

@dataclass(frozen=True)
class FittedPrototypes:
    prototypes: torch.Tensor
    counts: torch.Tensor
    fit_ids: torch.Tensor
    fit_labels: torch.Tensor
    fit_embeddings: torch.Tensor

def fit_class_prototypes(embeddings: torch.Tensor, support_ids: torch.Tensor, support_labels: torch.Tensor, *, num_classes: int, excluded_ids: torch.Tensor | None=None) -> FittedPrototypes:
    embeddings = embeddings.detach().cpu().float()
    support_ids = support_ids.detach().cpu().view(-1).long()
    support_labels = support_labels.detach().cpu().view(-1).long()
    if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
        raise ValueError('embeddings must be a finite [N, D] tensor')
    if support_ids.shape != support_labels.shape:
        raise ValueError('support ID and label shapes differ')
    if support_ids.numel() != torch.unique(support_ids).numel():
        raise ValueError('support IDs must be unique')
    if support_ids.numel() and (int(support_ids.min()) < 0 or int(support_ids.max()) >= embeddings.size(0)):
        raise ValueError('support ID outside embedding rows')
    keep = torch.ones(support_ids.numel(), dtype=torch.bool)
    if excluded_ids is not None:
        excluded_ids = excluded_ids.detach().cpu().view(-1).long()
        keep &= ~torch.isin(support_ids, excluded_ids)
    fit_ids = support_ids[keep]
    fit_labels = support_labels[keep]
    if fit_labels.numel() and (int(fit_labels.min()) < 0 or int(fit_labels.max()) >= int(num_classes)):
        raise ValueError('support label outside class space')
    fit_embeddings = embeddings[fit_ids]
    prototypes = torch.zeros((int(num_classes), embeddings.size(1)), dtype=torch.float32)
    counts = torch.bincount(fit_labels, minlength=int(num_classes)).long()
    if fit_ids.numel():
        prototypes.index_add_(0, fit_labels, fit_embeddings)
    prototypes = prototypes / counts.clamp_min(1).float()[:, None]
    return FittedPrototypes(prototypes=prototypes, counts=counts, fit_ids=fit_ids, fit_labels=fit_labels, fit_embeddings=fit_embeddings)

def prototype_evidence(*, embeddings: torch.Tensor, row_ids: torch.Tensor, candidate_g: torch.Tensor, candidate_l: torch.Tensor, fitted: FittedPrototypes, chunk_size: int=16384) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    embeddings = embeddings.detach().cpu().float()
    row_ids = row_ids.detach().cpu().view(-1).long()
    candidate_g = candidate_g.detach().cpu().view(-1).long()
    candidate_l = candidate_l.detach().cpu().view(-1).long()
    if not row_ids.shape == candidate_g.shape == candidate_l.shape:
        raise ValueError('prototype row and candidate shapes differ')
    num_classes = fitted.prototypes.size(0)
    normalized_rows = F.normalize(embeddings[row_ids], dim=1, eps=1e-12)
    normalized_prototypes = F.normalize(fitted.prototypes, dim=1, eps=1e-12)
    prototype_distance = 1.0 - normalized_rows @ normalized_prototypes.T
    missing_class = fitted.counts == 0
    prototype_distance[:, missing_class] = 2.0
    sorted_distance = prototype_distance.sort(dim=1).values
    minimum = sorted_distance[:, 0]
    gap = sorted_distance[:, 1] - sorted_distance[:, 0] if num_classes > 1 else torch.zeros_like(minimum)
    normalized_support = F.normalize(fitted.fit_embeddings, dim=1, eps=1e-12)
    nearest_support = torch.full((row_ids.numel(),), 2.0)
    if normalized_support.numel():
        for start in range(0, row_ids.numel(), int(chunk_size)):
            stop = min(start + int(chunk_size), row_ids.numel())
            distance = 1.0 - normalized_rows[start:stop] @ normalized_support.T
            nearest_support[start:stop] = distance.min(dim=1).values

    def candidate_features(candidate: torch.Tensor) -> torch.Tensor:
        distance = prototype_distance.gather(1, candidate[:, None]).view(-1)
        class_missing = missing_class[candidate]
        similarity = torch.where(class_missing, torch.zeros_like(distance), 1.0 - distance)
        rank = (prototype_distance < distance[:, None]).sum(dim=1).float() / max(num_classes - 1, 1)
        nearest_candidate = torch.full_like(distance, 2.0)
        nearest_non_candidate = torch.full_like(distance, 2.0)
        candidate_support_missing = torch.ones_like(distance, dtype=torch.bool)
        non_candidate_support_missing = torch.ones_like(distance, dtype=torch.bool)
        if normalized_support.numel():
            for start in range(0, row_ids.numel(), int(chunk_size)):
                stop = min(start + int(chunk_size), row_ids.numel())
                block_distance = 1.0 - normalized_rows[start:stop] @ normalized_support.T
                block_candidate = candidate[start:stop]
                same = fitted.fit_labels[None, :] == block_candidate[:, None]
                other = ~same
                candidate_support_missing[start:stop] = ~same.any(dim=1)
                non_candidate_support_missing[start:stop] = ~other.any(dim=1)
                nearest_candidate[start:stop] = block_distance.masked_fill(~same, 2.0).min(dim=1).values
                nearest_non_candidate[start:stop] = block_distance.masked_fill(~other, 2.0).min(dim=1).values
        return torch.stack([similarity, distance, nearest_candidate, nearest_non_candidate, rank, class_missing.float(), candidate_support_missing.float(), non_candidate_support_missing.float()], dim=1)
    g = candidate_features(candidate_g)
    l = candidate_features(candidate_l)
    global_features = torch.stack([minimum, gap, nearest_support, (fitted.fit_ids.numel() == 0) * torch.ones_like(minimum)], dim=1)
    return (g, l, global_features, list(GRAPH_CANDIDATE_NAMES), list(GRAPH_GLOBAL_NAMES))
