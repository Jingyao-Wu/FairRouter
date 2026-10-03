#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 GNN-as-Judge Contributors
# Copyright (c) 2026 FairRouter contributors
"""Generate CUDA few-shot partitions and shared 10-shot evaluation nodes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
SHOTS = (3, 5, 10)
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class GraphInputs:
    labels: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path) -> GraphInputs:
    """Read labels and candidate masks from a trusted PyG graph file."""
    graph = torch.load(path, map_location="cpu", weights_only=False)
    labels = graph.y
    train_mask, val_mask = graph.train_mask, graph.val_mask
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("Graph labels must be a one-dimensional integer tensor")
    for mask in (train_mask, val_mask):
        if mask.shape != labels.shape or mask.dtype != torch.bool:
            raise ValueError("Graph masks must be one-dimensional Boolean tensors")
    return GraphInputs(labels, train_mask, val_mask, sha256_file(path))


def load_arxiv_split(directory: Path) -> dict[str, torch.Tensor]:
    """Read the official OGB time split without downloading or transforming data."""
    split = {}
    for name in ("train", "valid", "test"):
        with gzip.open(directory / f"{name}.csv.gz", "rt", encoding="utf-8") as stream:
            split[name] = torch.tensor(
                [int(line.strip()) for line in stream if line.strip()], dtype=torch.long
            )
    return split


def sample_partition(
    graph: GraphInputs,
    dataset: str,
    shots: int,
    seed: int,
    device: torch.device,
    official: dict[str, torch.Tensor] | None,
) -> dict:
    """Sample each class, then validation, preserving CUDA RNG call order."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    labels = graph.labels.to(device)
    train_mask = graph.train_mask.to(device)
    val_mask = graph.val_mask.to(device)
    num_classes = labels.max().item() + 1
    new_train_mask = torch.zeros_like(train_mask)

    if dataset == "arxiv":
        if official is None:
            raise ValueError("Arxiv requires the official time split")
        official_train_mask = torch.zeros(
            labels.numel(), dtype=torch.bool, device=device
        )
        official_val_mask = torch.zeros(labels.numel(), dtype=torch.bool, device=device)
        official_test_mask = torch.zeros(
            labels.numel(), dtype=torch.bool, device=device
        )
        official_train_mask[official["train"].to(device)] = True
        official_val_mask[official["valid"].to(device)] = True
        official_test_mask[official["test"].to(device)] = True

        for c in range(num_classes):
            class_mask = (labels == c) & official_train_mask
            class_indices = class_mask.nonzero().squeeze(-1)
            num_available = class_indices.shape[0]
            if num_available > 0:
                actual_shots = min(shots, num_available)
                selected_indices = torch.randperm(num_available, device=device)[
                    :actual_shots
                ]
                selected = class_indices[selected_indices]
                new_train_mask[selected] = True

        unused_train_nodes = official_train_mask & (~new_train_mask)
        new_test_mask = official_test_mask | unused_train_nodes
        new_val_mask = official_val_mask
    else:
        for c in range(num_classes):
            class_indices = ((labels == c) & train_mask).nonzero().squeeze(-1)
            num_available = class_indices.numel()
            actual_shots = min(shots, num_available)
            # Empty classes also make this call; do not skip or combine permutations.
            perm = torch.randperm(num_available, device=device)
            selected = class_indices[perm[:actual_shots]]
            new_train_mask[selected] = True

        val_indices = (~train_mask).nonzero().squeeze(-1)
        perm = torch.randperm(val_indices.numel(), device=device)
        selected = val_indices[perm[:500]]
        new_val_mask = torch.zeros_like(val_mask)
        new_val_mask[selected] = True
        new_test_mask = ~(new_train_mask | new_val_mask)

    support = new_train_mask.nonzero().squeeze(-1).cpu().tolist()
    valid = new_val_mask.nonzero().squeeze(-1).cpu().tolist()
    query = new_test_mask.nonzero().squeeze(-1).cpu().tolist()
    if len(query) < 1000:
        raise ValueError("The query partition must contain at least 1000 nodes")
    return {
        "dataset": dataset,
        "shots": shots,
        "split_seed": seed,
        "num_nodes": labels.numel(),
        "graph_sha256": graph.sha256,
        "support_ids": support,
        "valid_ids": valid,
        "unlabeled_ids": query,
        "test_truth_exported": False,
    }


def validate_split(split: dict) -> None:
    groups = [set(split[key]) for key in ("support_ids", "valid_ids", "unlabeled_ids")]
    if any(groups[a] & groups[b] for a in range(3) for b in range(a)):
        raise ValueError("Support, validation and query must be disjoint")
    if set.union(*groups) != set(range(split["num_nodes"])):
        raise ValueError("Support, validation and query must partition all nodes")
    standard = split["standard_eval_ids"]
    if (
        len(standard) != 1000
        or len(set(standard)) != 1000
        or not set(standard) <= groups[2]
    ):
        raise ValueError("Shared evaluation must contain 1000 distinct query nodes")


def write_splits(output: Path, records: dict[str, dict]) -> None:
    """Write deterministic JSON and relative checksums; reject changed outputs."""
    files = {
        name: (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
        for name, value in sorted(records.items())
    }
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(content).hexdigest()}  {name}\n"
        for name, content in files.items()
    ).encode("utf-8")
    for name, content in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Existing output differs: {name}")
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, required=True, help="Directory of dataset .pt files"
    )
    parser.add_argument(
        "--arxiv-split-dir", type=Path, help="Directory of OGB time split CSV.gz files"
    )
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument(
        "--shots", nargs="+", type=int, choices=SHOTS, default=list(SHOTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda:0", help="CUDA sampling device")
    args = parser.parse_args()
    for name in ("datasets", "shots", "seeds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must not contain duplicates")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32)")
    if "arxiv" in args.datasets and args.arxiv_split_dir is None:
        parser.error("--arxiv-split-dir is required when generating Arxiv splits")
    args.device = torch.device(args.device)
    if args.device.type != "cuda":
        parser.error("Split generation requires a CUDA device")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Split generation requires an available CUDA GPU")
    official = (
        load_arxiv_split(args.arxiv_split_dir) if "arxiv" in args.datasets else None
    )
    records = {}
    for dataset in args.datasets:
        graph = load_graph(args.data_root / f"{dataset}.pt")
        for seed in args.seeds:
            partitions = {
                shots: sample_partition(
                    graph, dataset, shots, seed, args.device, official
                )
                for shots in sorted(set(args.shots) | {10})
            }
            standard = random.Random(seed).sample(partitions[10]["unlabeled_ids"], 1000)
            for shots in args.shots:
                split = partitions[shots]
                split["standard_eval_ids"] = standard
                validate_split(split)
                records[f"{shots}/{seed}/{dataset}.json"] = split
    write_splits(args.output, records)
    print(f"Generated {len(records)} splits and SHA256SUMS")


if __name__ == "__main__":
    main()
