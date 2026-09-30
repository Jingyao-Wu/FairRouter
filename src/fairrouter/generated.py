"""Verify regenerated experts before using the authenticated FairRouter runner."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from eargtc.models.gnn import MLPHead
from eargtc.router_v7.perturbations import mixed_assignment

from .artifacts import Bundle, read_json, sha256, verify_seal, write_json
from .frontend import Frontend, positions
from .infer_llm import validate_shard


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_llm(root, manifest, view, wanted):
    wanted = torch.as_tensor(wanted, dtype=torch.long)
    n, c = manifest["num_nodes"], len(manifest["label_names"])
    if wanted.unique().numel() != len(wanted) or ((wanted < 0) | (wanted >= n)).any():
        raise ValueError("Invalid requested LLM node IDs")
    position = torch.full((n,), -1, dtype=torch.long)
    position[wanted] = torch.arange(len(wanted))
    hidden = torch.empty(len(wanted), 4096, dtype=torch.float16)
    logits = torch.empty(len(wanted), c)
    changed = torch.empty(len(wanted), dtype=torch.bool)
    found = torch.zeros(len(wanted), dtype=torch.bool)
    seen = torch.zeros(n, dtype=torch.bool)
    root = Path(root).resolve()
    for record in manifest["views"][view]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or sha256(path) != record["sha256"]:
            raise ValueError("LLM shard path or checksum differs")
        shard = load_tensor(path)
        ids = shard["node_ids"]
        validate_shard(shard, ids.tolist(), c, manifest["request_sha256"])
        if ids.dtype != torch.long or ids.ndim != 1 or ((ids < 0) | (ids >= n)).any():
            raise ValueError("Invalid LLM shard node IDs")
        if ids.unique().numel() != len(ids) or seen[ids].any():
            raise ValueError("Duplicate LLM node IDs")
        seen[ids] = True
        if shard["request_sha256"] != manifest["request_sha256"]:
            raise ValueError("LLM shard comes from different inference settings")
        if shard["hidden"].shape != (len(ids), 4096) or shard["logits"].shape != (
            len(ids),
            c,
        ):
            raise ValueError("LLM shard dimensions differ")
        if (
            not torch.isfinite(shard["hidden"]).all()
            or not torch.isfinite(shard["logits"]).all()
        ):
            raise ValueError("Nonfinite LLM values")
        rows = position[ids]
        keep = rows >= 0
        hidden[rows[keep]] = shard["hidden"][keep]
        logits[rows[keep]] = shard["logits"][keep]
        changed[rows[keep]] = shard["changed_mask"][keep]
        found[rows[keep]] = True
    if not found.all():
        raise ValueError(
            f"LLM view {view} is missing requested nodes; regenerate with all needed splits"
        )
    return dict(hidden=hidden, logits=logits, changed_mask=changed)


def verify_generated(bundle_root, frontend_root, gnn_root, llm_root):
    bundle, frontend = Bundle(bundle_root), Frontend(frontend_root)
    gnn_root, llm_root = Path(gnn_root), Path(llm_root)
    meta = verify_seal(gnn_root)
    lm = read_json(llm_root / "manifest.json")
    request = read_json(llm_root / "request.json")
    fingerprint = hashlib.sha256(
        json.dumps(request, sort_keys=True).encode()
    ).hexdigest()
    if lm["request_sha256"] != fingerprint or any(
        lm.get(k) != v for k, v in request.items()
    ):
        raise ValueError("LLM request and manifest differ")
    for key in ("dataset", "num_nodes", "graph_sha256", "label_names"):
        if meta[key] != lm[key]:
            raise ValueError(f"GNN/LLM {key} differs")
    cell = bundle.cell(meta["shot"], meta["seed"], meta["dataset"])
    split = bundle.split(cell)
    generated_split = read_json(gnn_root / "split.json")
    for key in (
        "dataset",
        "shots",
        "split_seed",
        "num_nodes",
        "graph_sha256",
        "support_ids",
        "valid_ids",
        "unlabeled_ids",
        "standard_eval_ids",
        "evaluation_1000_ids",
        "evaluation_full_ids",
        "test_truth_exported",
    ):
        if generated_split[key] != split[key]:
            raise ValueError(f"Generated split differs: {key}")
    n = split["num_nodes"]
    ids = torch.tensor(split["support_ids"])
    source = next(
        c
        for c in frontend.manifest["cells"]
        if (c["shot"], c["seed"], c["dataset"])
        == (meta["shot"], meta["seed"], meta["dataset"])
    )
    original = bundle.tensor(cell["inputs"]["upstream"])
    embeddings = bundle.tensor(cell["inputs"]["embeddings"])["embeddings"]
    hidden = bundle.tensor(cell["inputs"]["hidden"])["hidden"]
    gnn = load_tensor(gnn_root / "gnn.pt")
    clean = load_llm(llm_root, lm, "clean", torch.arange(n))

    def exact(actual, expected, name):
        try:
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        except AssertionError as error:
            raise ValueError(
                f"Generated inputs differ from the package: {name}"
            ) from error

    exact(gnn["node_ids"], torch.arange(n), "GNN node order")
    exact(gnn["embeddings"], embeddings, "GNN embeddings")
    exact(gnn["logits"], original["gnn_logits"], "GNN logits")
    exact(clean["hidden"], hidden, "LLM hidden states")
    exact(clean["logits"], original["llm_logits"], "LLM logits")
    safe = load_tensor(gnn_root / "supervision.pt")
    expected_safe = bundle.tensor(cell["inputs"]["supervision"])
    if set(safe) != set(expected_safe):
        raise ValueError("Unexpected supervision fields")
    for key in safe:
        exact(safe[key], expected_safe[key], key)
    graph = load_tensor(gnn_root / "graph.pt")
    exact(
        graph["edge_index"],
        frontend.tensor(source["inputs"]["graph"])["edge_index"],
        "graph edges",
    )
    for name, reference_id in source["replay"].items():
        reference = frontend.tensor(reference_id)
        if name.startswith("text"):
            generated = load_llm(llm_root, lm, name, ids)
            pos = positions(reference["node_ids"], ids)
            changed = reference["changed_mask"][pos]
            exact(generated["changed_mask"], changed, name + " changes")
            actual_h = generated["hidden"].clone()
            actual_p = generated["logits"].softmax(1)
            actual_h[~changed] = clean["hidden"][ids][~changed]
            actual_p[~changed] = clean["logits"][ids].softmax(1)[~changed]
            expected_h = reference["l"][pos]
            expected_p = reference["llm_prob"][pos].clone()
            expected_p[~changed] = original["llm_logits"][ids].softmax(1)[~changed]
            exact(actual_h, expected_h, name + " hidden")
            exact(actual_p, expected_p, name + " probabilities")
        elif name.startswith("graph"):
            generated = load_tensor(gnn_root / f"{name}.pt")
            expected_changed = (
                torch.tensor(
                    [
                        v == "graph"
                        for v in mixed_assignment(range(n), seed=303).values()
                    ]
                )
                if "mixed" in name
                else torch.ones(n, dtype=torch.bool)
            )
            exact(generated["changed_mask"], expected_changed, name + " assignment")
            exact(generated["edge_index"], reference["edge_index"], name + " edges")
            if meta["shot"] == 5:
                exact(reference["node_ids"], ids, name + " parent order")
                changed = generated["changed_mask"][ids]
                exact(changed, reference["changed_mask"], name + " changes")
                actual_h = generated["embeddings"][ids].clone()
                actual_p = generated["logits"][ids].softmax(1)
                actual_h[~changed] = embeddings[ids][~changed]
                actual_p[~changed] = original["gnn_logits"][ids].softmax(1)[~changed]
                expected_p = reference["gnn_prob"].clone()
                expected_p[~changed] = original["gnn_logits"][ids].softmax(1)[~changed]
                exact(actual_h, reference["g"], name + " embeddings")
                exact(actual_p, expected_p, name + " probabilities")
                exact(
                    generated["logits"].argmax(1),
                    reference["full_gnn_pred"],
                    name + " predictions",
                )
            else:
                exact(generated["embeddings"], reference["g"], name + " embeddings")
                state = frontend.tensor(source["inputs"]["head"])
                head = MLPHead(
                    128,
                    state["net.0.weight"].shape[0],
                    state["net.3.weight"].shape[0],
                    0.0,
                )
                head.load_state_dict(state)
                head.eval()
                with torch.inference_mode():
                    expected_logits = head(reference["g"])
                exact(generated["logits"], expected_logits, name + " logits")
    return dict(
        dataset=meta["dataset"],
        shot=meta["shot"],
        seed=meta["seed"],
        all_expert_inputs_exact=True,
        gnn_manifest_sha256=sha256(gnn_root / "FROZEN.json"),
        llm_manifest_sha256=sha256(llm_root / "manifest.json"),
        bundle_manifest_sha256=sha256(Path(bundle_root) / "manifest.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "run"):
        command = sub.add_parser(name)
        for key in ("bundle", "frontend", "gnn", "llm", "output"):
            command.add_argument("--" + key, type=Path, required=True)
        if name == "run":
            command.add_argument(
                "--mode",
                choices=("refit-all", "refit-joint", "replay"),
                default="refit-all",
            )
    evaluate = sub.add_parser("evaluate")
    for key in ("bundle", "run", "output"):
        evaluate.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "evaluate":
        from .evaluation import evaluate_run

        evaluate_run(args.bundle, args.run, args.output)
        return
    report = verify_generated(args.bundle, args.frontend, args.gnn, args.llm)
    if args.command == "verify":
        write_json(args.output, report)
    else:
        from .pipeline import run_cell

        # Only numerically identical regenerated inputs may enter this benchmark.
        # The shared runner retains cross-dataset fits and its validation protocol.
        run_cell(
            args.bundle,
            args.output,
            report["shot"],
            report["seed"],
            report["dataset"],
            args.mode,
        )
        path = (
            args.output
            / "expert_verification"
            / f"shot{report['shot']}/seed{report['seed']}/{report['dataset']}"
        )
        write_json(path / "expert_verification.json", report)


if __name__ == "__main__":
    main()
