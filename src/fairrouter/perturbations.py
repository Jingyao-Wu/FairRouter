"""Deterministic graph and text perturbations for support observations."""

import math
from collections.abc import Iterable
import torch
from eargtc.router_v7.perturbations import mixed_assignment as mixed_assignment


def _severity(value: float) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    return result


def _canonical_pairs(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    edge = torch.as_tensor(edge_index, dtype=torch.long).cpu()
    if edge.ndim != 2 or edge.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    pairs: set[tuple[int, int]] = set()
    for raw_u, raw_v in edge.t().tolist():
        u, v = (int(raw_u), int(raw_v))
        if not (0 <= u < int(num_nodes) and 0 <= v < int(num_nodes)):
            raise ValueError("edge_index contains node outside graph")
        if u != v:
            pairs.add((min(u, v), max(u, v)))
    return sorted(pairs)


def _edge_tensor(pairs: Iterable[tuple[int, int]]) -> torch.Tensor:
    directed = [(u, v) for u, v in pairs for u, v in ((u, v), (v, u))]
    if not directed:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(sorted(directed), dtype=torch.long).t().contiguous()


def edge_dropout(
    edge_index: torch.Tensor, *, severity: float, seed: int, num_nodes: int
) -> torch.Tensor:
    drop = _severity(severity)
    pairs = _canonical_pairs(edge_index, num_nodes)
    if drop == 0.0 or not pairs:
        return _edge_tensor(pairs)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    keep = torch.rand(len(pairs), generator=generator) >= drop
    return _edge_tensor(
        (pair for pair, selected in zip(pairs, keep.tolist()) if selected)
    )


def ego_edge_mask(
    edge_index: torch.Tensor,
    *,
    node_ids: Iterable[int],
    severity: float,
    seed: int,
    num_nodes: int,
) -> torch.Tensor:
    drop = _severity(severity)
    selected = {int(node_id) for node_id in node_ids}
    if any((node_id < 0 or node_id >= int(num_nodes) for node_id in selected)):
        raise ValueError("ego mask contains node outside graph")
    pairs = _canonical_pairs(edge_index, num_nodes)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    draws = torch.rand(len(pairs), generator=generator).tolist()
    kept = [
        pair
        for pair, draw in zip(pairs, draws)
        if not ((pair[0] in selected or pair[1] in selected) and draw < drop)
    ]
    return _edge_tensor(kept)


def _tokens(text: str) -> list[str]:
    return str(text or "").split()


def truncate_text(text: str, keep_fraction: float) -> str:
    keep = _severity(keep_fraction)
    tokens = _tokens(text)
    if not tokens or keep == 0.0:
        return ""
    count = max(1, min(len(tokens), int(math.ceil(len(tokens) * keep))))
    return " ".join(tokens[:count])


def mask_tokens(text: str, severity: float, *, seed: int, mask: str = "[MASK]") -> str:
    fraction = _severity(severity)
    tokens = _tokens(text)
    if not tokens or fraction == 0.0:
        return " ".join(tokens)
    count = min(len(tokens), max(1, int(round(len(tokens) * fraction))))
    order = torch.randperm(
        len(tokens), generator=torch.Generator().manual_seed(int(seed))
    )
    for index in order[:count].tolist():
        tokens[index] = str(mask)
    return " ".join(tokens)


def delete_span(text: str, severity: float, *, seed: int) -> str:
    fraction = _severity(severity)
    tokens = _tokens(text)
    if not tokens or fraction == 0.0:
        return " ".join(tokens)
    count = min(len(tokens), max(1, int(round(len(tokens) * fraction))))
    limit = len(tokens) - count + 1
    start = int(
        torch.randint(
            limit, (1,), generator=torch.Generator().manual_seed(int(seed))
        ).item()
    )
    return " ".join(tokens[:start] + tokens[start + count :])
