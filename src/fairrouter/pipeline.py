"""Train and predict with FairRouter for a dataset, label budget and seed."""

import os
from pathlib import Path
import platform
import time

import torch
from threadpoolctl import threadpool_limits

from eargtc.residual_data.experiment import prepare_arrays
from eargtc.validation.artifacts import validate_hidden
from . import training
from .artifacts import Bundle, save_tensor, seal, sha256, write_json
from .router import replay


def run_cell(bundle_root, output, shot, seed, dataset, mode, bank_root=None, expert_root=None):
    torch.set_num_threads(2)
    artifacts = Bundle(bundle_root, bank_root=bank_root, expert_root=expert_root)
    cell = artifacts.cell(shot, seed, dataset)
    split = artifacts.split(cell)
    destination = Path(output) / f"shot{shot}/seed{seed}/{dataset}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a run: {destination}")
    destination.mkdir(parents=True)
    started = time.monotonic()
    values = {
        key: artifacts.tensor(cell["inputs"][key])
        for key in ("upstream", "embeddings", "supervision")
    }
    # Use one thread for routing and two for residual training.
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        router, diagnostics, models = replay(
            artifacts, cell, values["upstream"], refit=mode == "refit-all"
        )
    torch.set_num_threads(2)
    f, s, pools = prepare_arrays(
        split, values["upstream"], values["embeddings"], values["supervision"], router, dataset
    )
    hidden = artifacts.tensor(cell["inputs"]["hidden"])
    validate_hidden(hidden, split["num_nodes"])
    if hidden["metadata"]["dataset"] != dataset:
        raise ValueError("Hidden representation belongs to another dataset")
    hidden = hidden["hidden"]
    reference = artifacts.tensor(cell["inputs"]["checkpoint"])
    if reference["config"] != cell["config"] or reference["training_seed"] != seed:
        raise ValueError("Reference checkpoint identity differs")
    member = (
        reference if mode == "replay" else training.train(f, s, pools, hidden, cell["config"], seed)
    )
    if member["checkpoints"]["accuracy"] != reference["checkpoints"]["accuracy"]:
        raise ValueError("Fresh Joint validation checkpoint differs")
    for key, value in member["states"]["accuracy"].items():
        torch.testing.assert_close(value, reference["states"]["accuracy"][key], atol=0, rtol=0)
    model = training.make_model(f, hidden, cell["config"])
    model.load_state_dict(member["states"]["accuracy"])
    validation = training.predict(model, f, hidden, s["selection_ids"])
    torch.testing.assert_close(validation, reference["validation_probability"], atol=0, rtol=0)
    probability = training.predict(model, f, hidden, f["node_ids"])
    reference_predictions = artifacts.tensor(cell["inputs"]["predictions"])
    torch.testing.assert_close(probability, reference_predictions["probability"], atol=0, rtol=0)
    save_tensor(
        destination / "predictions.pt", dict(node_ids=f["node_ids"], probability=probability)
    )
    save_tensor(destination / "checkpoint.pt", member)
    save_tensor(destination / "router.pt", router)
    if mode == "refit-all":
        import joblib

        with (destination / "router_models.joblib").open("xb") as stream:
            joblib.dump(models, stream, compress=3)
    report = dict(
        dataset=dataset,
        shot=shot,
        seed=seed,
        mode=mode,
        exact_joint_weights=True,
        exact_joint_validation=True,
        exact_full_node_predictions=True,
        router_decisions_match=True,
        reconstructed_banks_used=bank_root is not None,
        refitted_experts_used=expert_root is not None,
        router_refit=mode == "refit-all",
        router_diagnostics=diagnostics,
        checkpoint=member["checkpoints"]["accuracy"],
        config=cell["config"],
        pool_counts={k: len(v) for k, v in pools.items()},
        test_truth_loaded=False,
        standard_eval_count=len(split["standard_eval_ids"]),
        bundle_manifest_sha256=sha256(Path(bundle_root) / "manifest.json"),
        seconds=time.monotonic() - started,
        torch_version=str(torch.__version__),
        python_version=platform.python_version(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    write_json(destination / "diagnostics.json", report)
    seal(destination, report)
    print("FROZEN", dataset, shot, seed, mode, round(report["seconds"], 1), flush=True)
