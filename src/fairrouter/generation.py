"""Shared input validation for generating experts from trusted local graph files."""

import torch

from .artifacts import check_split, read_json, sha256


def load_graph(path):
    """Accept a PyG Data object or a tensor dictionary in canonical node order."""
    raw = torch.load(path, map_location="cpu", weights_only=False)

    def field(name):
        return raw[name] if isinstance(raw, dict) else getattr(raw, name)

    x, edge, y = field("x"), field("edge_index"), field("y").reshape(-1)
    if x.ndim != 2 or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(
            "x must be a finite floating-point [num_nodes, features] tensor"
        )
    n = len(x)
    if y.dtype != torch.long or y.shape != (n,):
        raise ValueError("y must contain one int64 class ID per node")
    if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
        raise ValueError("edge_index must be int64 [2, num_edges]")
    if edge.numel() and (edge.min() < 0 or edge.max() >= n):
        raise ValueError("edge_index contains an out-of-range node")
    # Canonical undirected edges retain self loops and remove duplicate entries.
    both = torch.cat((edge, edge.flip(0)), dim=1)
    keys = torch.unique(both[0] * max(n, 1) + both[1])
    edge = torch.stack((keys // max(n, 1), keys % max(n, 1)))
    return x.float(), edge, y


def validate_split(split, n):
    if split["num_nodes"] != n:
        raise ValueError("Split and graph node counts differ")
    groups = []
    for key in ("support_ids", "valid_ids", "unlabeled_ids"):
        ids = split[key]
        if any(type(i) is not int or not 0 <= i < n for i in ids) or len(
            set(ids)
        ) != len(ids):
            raise ValueError(f"Invalid or duplicate {key}")
        groups.append(set(ids))
    if not groups[0] or len(groups[1]) < 2 or not groups[2]:
        raise ValueError("Need support, at least two validation nodes, and query nodes")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Support, validation and query overlap")
    if set.union(*groups) != set(range(n)):
        raise ValueError("Split must partition all nodes")
    check_split(
        split,
        dict(shot=split["shots"], seed=split["split_seed"], dataset=split["dataset"]),
    )
    if split["evaluation_full_ids"] != split["unlabeled_ids"]:
        raise ValueError("Full evaluation must use the complete query partition")
    own = split["evaluation_1000_ids"]
    if len(own) != 1000 or len(set(own)) != 1000 or not set(own) <= groups[2]:
        raise ValueError("Own evaluation must contain 1000 distinct query nodes")


def graph_split(graph_path, split_path):
    x, edge, y = load_graph(graph_path)
    split = read_json(split_path)
    validate_split(split, len(x))
    digest = sha256(graph_path)
    if split["graph_sha256"] != digest:
        raise ValueError("Split was generated from a different graph file")
    return x, edge, y, split, digest


def label_names(path):
    names = read_json(path)
    if (
        not isinstance(names, list)
        or len(names) < 2
        or any(not isinstance(s, str) or not s.strip() for s in names)
    ):
        raise ValueError("Label names must be a JSON list in numeric class-ID order")
    if len(set(names)) != len(names):
        raise ValueError("Label names must be unique")
    return names


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
