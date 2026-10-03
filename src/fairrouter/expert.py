"""Train the GCN classification head using fixed encoder representations.

Only support and validation labels are available to head training.
"""

import argparse
import os
from pathlib import Path
import time

import torch

from eargtc.utils import set_seed
from .artifacts import Bundle, DATASETS, SHOTS, SEEDS, save_tensor, seal, sha256
from .expert_head import HeadTrial, train_head
from .frontend import Frontend


def refit_expert(bundle_root, frontend_root, output, shot, seed, dataset):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GCN head training requires a CUDA GPU"
        )
    torch.set_num_threads(4)
    bundle, frontend = Bundle(bundle_root), Frontend(frontend_root)
    cell = bundle.cell(shot, seed, dataset)
    source = next(
        c
        for c in frontend.manifest["cells"]
        if (c["shot"], c["seed"], c["dataset"]) == (shot, seed, dataset)
    )
    split = bundle.split(cell)
    safe = bundle.tensor(cell["inputs"]["supervision"])
    original = bundle.tensor(cell["inputs"]["upstream"])
    embeddings = bundle.tensor(cell["inputs"]["embeddings"])["embeddings"].float().cuda()
    labels = torch.full((split["num_nodes"],), -1, dtype=torch.long, device="cuda")
    train, valid = safe["support_ids"].cuda(), safe["valid_ids"].cuda()
    labels[train], labels[valid] = safe["support_targets"].cuda(), safe["valid_targets"].cuda()
    if not bool((labels[torch.tensor(split["unlabeled_ids"], device="cuda")] == -1).all()):
        raise ValueError("Query truth exposed")
    destination = Path(output) / f"shot{shot}/seed{seed}/{dataset}"
    if destination.exists():
        raise FileExistsError("Use a new expert output directory")
    destination.mkdir(parents=True)
    if source["reference_seed42_head_replayed"]:
        # Reuse the supplied expert when this split specifies a fixed head.
        save_tensor(destination / "upstream.pt", original)
        seal(
            destination,
            dict(
                dataset=dataset,
                shot=shot,
                seed=seed,
                model_seed=source["model_seed"],
                trial=source["head_trial"],
                exact_logits=True,
                max_abs_error=0.0,
                same_all_node_predictions=True,
                refitted=False,
                reference_anchor_replayed=True,
                test_truth_loaded=False,
                bundle_manifest_sha256=sha256(Path(bundle_root) / "manifest.json"),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            ),
        )
        print("EXPERT ANCHOR", shot, seed, dataset, flush=True)
        return
    set_seed(source["model_seed"])
    start = time.monotonic()
    logits, stats, state = train_head(
        embeddings,
        labels,
        train,
        valid,
        original["gnn_logits"].shape[1],
        HeadTrial(**source["head_trial"]),
    )
    logits = logits.cpu()
    exact = torch.equal(logits, original["gnn_logits"])
    max_error = float((logits - original["gnn_logits"]).abs().max())
    predictions_equal = torch.equal(logits.argmax(1), original["gnn_logits"].argmax(1))
    # Save the head outputs and their difference from the supplied expert.
    save_tensor(
        destination / "upstream.pt", dict(gnn_logits=logits, llm_logits=original["llm_logits"])
    )
    save_tensor(destination / "head.pt", {k: v.cpu() for k, v in state["state_dict"].items()})
    seal(
        destination,
        dict(
            dataset=dataset,
            shot=shot,
            seed=seed,
            model_seed=source["model_seed"],
            trial=source["head_trial"],
            stats=stats,
            refitted=True,
            reference_anchor_replayed=False,
            exact_logits=exact,
            max_abs_error=max_error,
            same_all_node_predictions=predictions_equal,
            test_truth_loaded=False,
            seconds=time.monotonic() - start,
            cuda_device=torch.cuda.get_device_name(),
            bundle_manifest_sha256=sha256(Path(bundle_root) / "manifest.json"),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    print(
        "EXPERT FROZEN",
        shot,
        seed,
        dataset,
        "exact_logits",
        exact,
        "max_error",
        max_error,
        flush=True,
    )
    if not exact:
        raise RuntimeError(
            "Expert logits differ; inspect the diagnostics. The frozen benchmark input was not replaced."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--frontend", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    cells = [(s, r, d) for s in SHOTS for r in SEEDS for d in DATASETS]
    if not 0 <= args.index < len(cells):
        parser.error("Index must be 0..44")
    refit_expert(args.bundle, args.frontend, args.output, *cells[args.index])


if __name__ == "__main__":
    main()
