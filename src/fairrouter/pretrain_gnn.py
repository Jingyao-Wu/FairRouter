"""Generate GCN experts using explicit encoder and per-split head recipes."""

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from eargtc.models.gnn import GNNEncoder, MLPHead
from eargtc.utils import set_seed
from .artifacts import read_json, save_tensor, seal, sha256, write_json
from .expert_head import HeadTrial, train_head
from .generation import cpu_state, graph_split, label_names
from .perturbations import edge_dropout, ego_edge_mask, mixed_assignment
from .ssl import pretrain_ssl_gnn


def graph_views(edge, n):
    assigned = mixed_assignment(range(n), seed=303)
    selected = [i for i in range(n) if assigned[i] == "graph"]
    yield (
        "graph_drop_030",
        edge_dropout(edge, severity=0.3, seed=101, num_nodes=n),
        torch.ones(n, dtype=torch.bool),
    )
    mask = torch.tensor([assigned[i] == "graph" for i in range(n)])
    yield (
        "graph_ego_mixed_050",
        ego_edge_mask(edge, node_ids=selected, severity=0.5, seed=303, num_nodes=n),
        mask,
    )


def generate(args):
    recipe = read_json(args.recipe)
    x, edge, y, split, graph_hash = graph_split(args.graph, args.split)
    if recipe["dataset"] != split["dataset"] or recipe["graph_sha256"] != graph_hash:
        raise ValueError("Expert recipe does not match the input graph and dataset")
    names = label_names(args.labels)
    if names != recipe["label_names"]:
        raise ValueError("Class names differ from the expert recipe")
    spec = next(
        c
        for c in recipe["heads"]
        if (c["shot"], c["seed"]) == (split["shots"], split["split_seed"])
    )
    cfg = recipe["ssl"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Expert generation requires CUDA")
    if args.output.exists():
        raise FileExistsError(args.output)
    if spec["fixed_head"] and args.head_checkpoint is None:
        raise ValueError("This cell requires its fixed head checkpoint")
    support = torch.tensor(split["support_ids"], dtype=torch.long)
    valid = torch.tensor(split["valid_ids"], dtype=torch.long)
    safe_y = torch.full_like(y, -1)
    safe_y[support], safe_y[valid] = y[support], y[valid]
    del y
    if (safe_y[torch.cat((support, valid))] < 0).any() or (safe_y >= len(names)).any():
        raise ValueError("Support/validation targets do not match label names")
    torch.set_num_threads(cfg["threads"])
    set_seed(cfg["seed"])
    stats = None
    if args.encoder_checkpoint is None:
        data = SimpleNamespace(
            x=x.to(device), edge_index=edge.to(device), num_nodes=len(x)
        )
        encoder, embeddings, stats = pretrain_ssl_gnn(data, SimpleNamespace(**cfg))
        stats = asdict(stats)
        del data
    else:
        if sha256(args.encoder_checkpoint) != recipe["encoder_checkpoint_sha256"]:
            raise ValueError("Encoder checkpoint identity differs")
        saved = torch.load(
            args.encoder_checkpoint, map_location="cpu", weights_only=True
        )
        encoder = GNNEncoder(
            x.shape[1],
            cfg["gnn_hidden_dim"],
            cfg["embedding_dim"],
            cfg["gnn_layers"],
            "GCN",
            cfg["gnn_dropout"],
        )
        encoder.load_state_dict(saved["encoder_state"])
        embeddings = saved["embeddings"].float().to(device)
    if embeddings.shape != (len(x), 128) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid encoder representations")
    trial = HeadTrial(**spec["head_trial"])
    torch.set_num_threads(4)
    set_seed(spec["model_seed"])
    if spec["fixed_head"]:
        if sha256(args.head_checkpoint) != recipe["fixed_head_checkpoint_sha256"]:
            raise ValueError("Fixed head checkpoint identity differs")
        saved = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
        head_state = saved["head_state"]
        head_stats = None
    else:
        if args.head_checkpoint is not None:
            raise ValueError(
                "This cell requires head fitting, not a checkpoint override"
            )
        _, head_stats, result = train_head(
            embeddings,
            safe_y.to(device),
            support.to(device),
            valid.to(device),
            len(names),
            trial,
        )
        head_state = result["state_dict"]
    head = MLPHead(128, trial.head_hidden_dim, len(names), trial.dropout).to(device)
    head.load_state_dict(head_state)
    head.eval()
    with torch.inference_mode():
        logits = head(embeddings).cpu()
    args.output.mkdir(parents=True)
    save_tensor(
        args.output / "gnn.pt",
        dict(node_ids=torch.arange(len(x)), embeddings=embeddings.cpu(), logits=logits),
    )
    save_tensor(
        args.output / "encoder.pt",
        dict(
            encoder_state=cpu_state(encoder),
            embeddings=embeddings.cpu(),
            config=cfg,
            stats=stats,
        ),
    )
    save_tensor(
        args.output / "head.pt",
        dict(head_state=cpu_state(head), config=asdict(trial), stats=head_stats),
    )
    save_tensor(args.output / "graph.pt", dict(edge_index=edge, num_nodes=len(x)))
    save_tensor(
        args.output / "supervision.pt",
        dict(
            support_ids=support,
            support_targets=safe_y[support],
            valid_ids=valid,
            valid_targets=safe_y[valid],
        ),
    )
    write_json(args.output / "split.json", split)
    # Graph-view representations use the CPU encoder in canonical edge order.
    torch.set_num_threads(2)
    encoder.cpu().eval()
    head.cpu()
    for name, shifted, changed in graph_views(edge, len(x)):
        with torch.inference_mode():
            hidden = encoder(x, shifted)
            shifted_logits = head(hidden)
        save_tensor(
            args.output / f"{name}.pt",
            dict(
                embeddings=hidden,
                logits=shifted_logits,
                edge_index=shifted,
                changed_mask=changed,
            ),
        )
    seal(
        args.output,
        dict(
            schema=2,
            dataset=split["dataset"],
            shot=split["shots"],
            seed=split["split_seed"],
            num_nodes=len(x),
            label_names=names,
            graph_sha256=graph_hash,
            recipe=recipe,
            recipe_sha256=sha256(args.recipe),
            runtime=dict(
                torch=str(torch.__version__),
                cuda=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(device),
                gpu_capability=list(torch.cuda.get_device_capability(device)),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                ssl_threads=cfg["threads"],
                head_threads=4,
                graph_view_threads=2,
            ),
            encoder_checkpoint_sha256=sha256(args.encoder_checkpoint)
            if args.encoder_checkpoint
            else None,
            head_checkpoint_sha256=sha256(args.head_checkpoint)
            if args.head_checkpoint
            else None,
            ssl_pretrained=args.encoder_checkpoint is None,
            pretraining_uses_labels=False,
            raw_graph_contains_query_labels=True,
            query_labels_used_for_training=False,
            head_training="fixed checkpoint" if spec["fixed_head"] else "support only",
            validation_usage="head checkpoint selection",
            package_parity_verified=False,
        ),
    )
    print(f"Generated GCN outputs: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("graph", "split", "labels", "recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
