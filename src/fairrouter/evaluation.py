"""Test targets are read only after every requested model is frozen."""

import csv
import itertools
import os
from pathlib import Path
import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .artifacts import (
    Bundle,
    DATASETS,
    SEEDS,
    SHOTS,
    read_json,
    require_slurm,
    seal,
    sha256,
    verify_seal,
    write_json,
)


def metrics(probability, targets):
    if probability.ndim != 2 or probability.shape[0] != len(targets):
        raise ValueError("Invalid probability/target shape")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError("Invalid class probabilities")
    np.testing.assert_allclose(probability.sum(1), 1.0, atol=2e-6, rtol=0)
    predictions = probability.argmax(1)
    classes = probability.shape[1]
    if targets.min() < 0 or targets.max() >= classes:
        raise ValueError("Target outside expert class space")
    accuracy = float(accuracy_score(targets, predictions))
    f1 = float(
        f1_score(targets, predictions, labels=np.arange(classes), average="macro", zero_division=0)
    )
    matrix = np.bincount(targets * classes + predictions, minlength=classes * classes).reshape(
        classes, classes
    )
    denom = matrix.sum(0) + matrix.sum(1)
    independent = np.divide(
        2 * matrix.diagonal(), denom, out=np.zeros(classes), where=denom > 0
    ).mean()
    if abs(accuracy - np.trace(matrix) / matrix.sum()) > 1e-12 or abs(f1 - independent) > 1e-12:
        raise ValueError("Independent metric calculation disagrees")
    return dict(accuracy=accuracy, macro_f1=f1)


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(bundle_root, run_root, output, reference_path=None, expert_root=None):
    require_slurm()
    torch.set_num_threads(2)
    artifacts = Bundle(bundle_root, expert_root=expert_root)
    run_root, output = Path(run_root), Path(output)
    if output.exists():
        raise FileExistsError("Evaluation output must be a new directory")
    expected_manifest = sha256(Path(bundle_root) / "manifest.json")
    # Authenticate all 45 completed models before opening any evaluation labels.
    sealed = {}
    for shot, seed, dataset in itertools.product(SHOTS, SEEDS, DATASETS):
        directory = run_root / f"shot{shot}/seed{seed}/{dataset}"
        record = verify_seal(directory)
        if (record["shot"], record["seed"], record["dataset"]) != (shot, seed, dataset):
            raise ValueError("Frozen cell identity differs")
        if (
            record["test_truth_loaded"] is not False
            or record["bundle_manifest_sha256"] != expected_manifest
        ):
            raise ValueError("Unsafe or mismatched frozen cell")
        sealed[str(directory.relative_to(run_root))] = sha256(directory / "FROZEN.json")
    output.mkdir(parents=True)
    write_json(
        output / "EVALUATION_STARTED.json",
        dict(
            models=sealed, bundle_sha256=expected_manifest, slurm_job_id=os.environ["SLURM_JOB_ID"]
        ),
    )
    rows = []
    llm_reference = {}
    standards = {}
    expected = read_json(reference_path) if reference_path is not None else None
    expected_rows = (
        {(r["shot"], r["seed"], r["dataset"]): r for r in expected["per_split"]} if expected else {}
    )
    baseline_path = Path(bundle_root) / "baseline_reference.json"
    baseline = read_json(baseline_path) if baseline_path.exists() else None
    baseline_rows = (
        {
            (
                "LLM" if r["model"] == "LLM-0shot" else r["model"],
                r["shot"],
                r["seed"],
                r["dataset"],
            ): r
            for r in baseline["per_split"]
        }
        if baseline
        else {}
    )
    for cell in artifacts.manifest["cells"]:
        shot, seed, dataset = cell["shot"], cell["seed"], cell["dataset"]
        split = artifacts.split(cell)
        standard = artifacts.split(artifacts.cell(10, seed, dataset))["evaluation_1000_ids"]
        if split["standard_eval_ids"] != standard:
            raise ValueError("Cross-shot standard node IDs/order differ")
        record = artifacts.manifest["evaluation"][f"{dataset}/{seed}"]
        truth = read_json(artifacts.authenticated(record["path"], record["sha256"]))
        if truth["node_ids"] != standard or truth["graph_sha256"] != split["graph_sha256"]:
            raise ValueError("Evaluation identity differs")
        standards[f"{dataset}/{seed}"] = dict(node_ids=standard, evaluation_sha256=record["sha256"])
        ids = torch.tensor(standard)
        targets = np.asarray(truth["targets"], dtype=np.int64)
        directory = run_root / f"shot{shot}/seed{seed}/{dataset}"
        obj = torch.load(directory / "predictions.pt", weights_only=True, map_location="cpu")
        if not torch.equal(obj["node_ids"], torch.arange(split["num_nodes"])):
            raise ValueError("Prediction node alignment differs")
        expert = artifacts.tensor(cell["inputs"]["upstream"])
        current_llm = expert["llm_logits"]
        if dataset in llm_reference:
            if not torch.equal(current_llm, llm_reference[dataset]):
                raise ValueError("0-shot LLM changed between shots or seeds")
        else:
            llm_reference[dataset] = current_llm.clone()
        for method, probability in (
            ("FairRouter", obj["probability"][ids]),
            ("GCN", expert["gnn_logits"][ids].float().softmax(1)),
            ("LLM", current_llm[ids].float().softmax(1)),
        ):
            if method == "LLM" and shot != 10:
                continue
            shot_value = 0 if method == "LLM" else shot
            value = metrics(probability.numpy(), targets)
            reference = (
                expected_rows.get((shot, seed, dataset))
                if method == "FairRouter"
                else baseline_rows.get((method, shot_value, seed, dataset))
            )
            if (
                expected is not None
                and method == "FairRouter"
                or baseline is not None
                and method != "FairRouter"
            ) and reference is None:
                raise ValueError("Missing expected result")
            if reference is not None and any(abs(value[m] - reference[m]) > 1e-12 for m in value):
                raise ValueError(f"Reference mismatch: {method}/{shot_value}/{seed}/{dataset}")
            rows.append(
                dict(
                    method=method,
                    dataset=dataset,
                    shot=shot_value,
                    seed=seed,
                    evaluation_count=1000,
                    population="standard_10shot_eval1000",
                    **value,
                )
            )
    summary = []
    for method, shots in [("FairRouter", SHOTS), ("GCN", SHOTS), ("LLM", (0,))]:
        for shot, dataset in itertools.product(shots, DATASETS):
            group = [
                r for r in rows if (r["method"], r["shot"], r["dataset"]) == (method, shot, dataset)
            ]
            if sorted(r["seed"] for r in group) != list(SEEDS):
                raise ValueError("Summary requires exactly seeds 42/43/44")
            record = dict(method=method, dataset=dataset, shot=shot, seed_count=3)
            for metric in ("accuracy", "macro_f1"):
                values = [r[metric] for r in group]
                record.update(
                    {
                        metric + "_mean": statistics.mean(values),
                        metric + "_std_ddof0": statistics.pstdev(values),
                        metric + "_std_ddof1": statistics.stdev(values),
                    }
                )
                np.testing.assert_allclose(
                    [
                        record[metric + "_mean"],
                        record[metric + "_std_ddof0"],
                        record[metric + "_std_ddof1"],
                    ],
                    [np.mean(values), np.std(values), np.std(values, ddof=1)],
                    atol=1e-12,
                    rtol=0,
                )
            summary.append(record)
    write_csv(output / "per_seed.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_json(output / "STANDARD_EVAL_IDS.json", standards)
    write_json(output / "RESULTS.json", dict(per_seed=rows, summary=summary, units="fraction"))
    lines = [
        "# FairRouter evaluation",
        "",
        "Accuracy (%), mean ± population SD (ddof=0), seeds 42/43/44.",
        "Each dataset and seed uses the same 1,000 evaluation nodes across methods and label budgets.",
        "",
        "| Method | Cora | CiteSeer | PubMed | Arxiv | Products |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, shots in [("FairRouter", SHOTS), ("GCN", SHOTS), ("LLM", (0,))]:
        for shot in shots:
            values = [
                next(
                    r
                    for r in summary
                    if (r["method"], r["shot"], r["dataset"]) == (method, shot, d)
                )
                for d in DATASETS
            ]
            lines.append(
                f"| {method} ({shot}-shot) | "
                + " | ".join(
                    f"{100 * r['accuracy_mean']:.2f} ± {100 * r['accuracy_std_ddof0']:.2f}"
                    for r in values
                )
                + " |"
            )
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines))
    seal(
        output,
        dict(
            all_checks_pass=True,
            cells=45,
            standard_sets=15,
            metric_rows=len(rows),
            reference_matches=expected is not None,
            baseline_matches=baseline is not None,
            identical_llm_logits_across_all_shots_and_seeds=True,
            test_truth_after_all_models_frozen=True,
            slurm_job_id=os.environ["SLURM_JOB_ID"],
        ),
    )
    print("EVALUATION COMPLETE", output, flush=True)
