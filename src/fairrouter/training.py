"""Router-guided residual training with gold, pseudo-label and agreement losses."""

import math
import os
import time
import torch
from torch.nn import functional as F
from .model import UnifiedRepair

DEFAULTS = dict(
    dim=64,
    lr=0.001,
    cross=1.0,
    gate_lr=1.0,
    warmup=0,
    gate_gold=False,
    aux=0.0,
    keep=0.0,
    max_steps=500,
    validate_every=50,
    patience=5,
    batch_size=64,
)


def score(p, y):
    if not len(y) or not torch.isfinite(p).all():
        raise ValueError("Invalid validation probability")
    return dict(
        accuracy=float((p.argmax(1) == y).double().mean()),
        nll=float(-p.double()[torch.arange(len(y)), y].clamp_min(1e-30).log().mean()),
    )


def checkpoint_key(r, criterion="accuracy"):
    return (
        (-r["accuracy"], r["nll"], r["step"])
        if criterion == "accuracy"
        else (r["nll"], -r["accuracy"], r["step"])
    )


def make_model(f, h, cfg):
    return UnifiedRepair(f["gnn_embeddings"].shape[1], h.shape[1], f["logp_g"].shape[1], cfg["dim"])


@torch.no_grad()
def predict(model, f, h, ids, batch=2048, repair=False):
    model.eval()
    ps = []
    rs = []
    for part in ids.split(batch):
        p, r = model(f["gnn_embeddings"][part], h[part], f["logp_g"][part], f["logp_l"][part])
        if not torch.isfinite(p).all():
            raise ValueError("Nonfinite prediction")
        ps.append(p.exp())
        if repair:
            rs.append(r.exp())
    return (torch.cat(ps), torch.cat(rs)) if repair else torch.cat(ps)


def train(f, s, p, h, config, seed):
    cfg = {**DEFAULTS, **config}
    if set(cfg) != set(DEFAULTS):
        raise ValueError("Unknown configuration fields")
    for k in ("dim", "max_steps", "validate_every", "patience", "batch_size"):
        if not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError("Invalid " + k)
    for k in ("lr", "cross", "gate_lr", "aux", "keep"):
        if not math.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError("Invalid " + k)
    if (
        cfg["lr"] == 0
        or cfg["gate_lr"] == 0
        or cfg["warmup"] < 0
        or cfg["warmup"] >= cfg["max_steps"]
    ):
        raise ValueError("Invalid optimizer schedule")
    if not len(s["gold_ids"]):
        raise ValueError("Gold support must be nonempty")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model = make_model(f, h, cfg)
    gate = [model.raw_alpha, model.raw_beta]
    body = [v for k, v in model.named_parameters() if k not in ("raw_alpha", "raw_beta")]
    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": cfg["lr"], "weight_decay": 1e-4},
            {"params": gate, "lr": cfg["lr"] * cfg["gate_lr"], "weight_decay": 0.0},
        ]
    )
    rng = {
        k: torch.Generator().manual_seed(seed + o)
        for k, o in [("gold", 101), ("teach_g", 1101), ("teach_l", 2101), ("agreement", 3101)]
    }
    states = {}
    checkpoints = {}
    valps = {}
    history = []
    stale = 0
    start = time.monotonic()

    def validate(step, losses, gradients):
        nonlocal stale
        probs, rp = predict(model, f, h, s["selection_ids"], repair=True)
        value = {**score(probs, s["selection_targets"]), "step": step}
        improve = "accuracy" not in checkpoints or checkpoint_key(value) < checkpoint_key(
            checkpoints["accuracy"]
        )
        for criterion in ("accuracy", "nll"):
            if criterion not in checkpoints or checkpoint_key(value, criterion) < checkpoint_key(
                checkpoints[criterion], criterion
            ):
                states[criterion] = {k: v.detach().clone() for k, v in model.state_dict().items()}
                checkpoints[criterion] = dict(value)
                valps[criterion] = probs.clone()
        stale = 0 if improve else stale + 1
        a, b = model.coefficients()
        history.append(
            {
                **value,
                "repair": score(rp, s["selection_targets"]),
                "alpha": float(a.detach()),
                "beta": float(b.detach()),
                "weights": model.weights().detach().tolist(),
                "losses": losses,
                "gradient_norms": gradients,
            }
        )

    def draw(name, ids):
        return (
            ids[torch.randint(len(ids), (cfg["batch_size"],), generator=rng[name])]
            if len(ids)
            else ids
        )

    def forward(ids, detach=False):
        return model(
            f["gnn_embeddings"][ids], h[ids], f["logp_g"][ids], f["logp_l"][ids], detach_gate=detach
        )

    validate(0, {}, {})
    for step in range(1, cfg["max_steps"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        warm = step <= cfg["warmup"]
        positions = draw("gold", torch.arange(len(s["gold_ids"])))
        ids = s["gold_ids"][positions]
        targets = s["gold_targets"][positions]
        logp, logr = forward(ids, warm)
        gold = F.nll_loss(logp, targets)
        auxgold = F.nll_loss(logr, targets) if cfg["aux"] else gold * 0
        loss = gold + cfg["aux"] * auxgold
        losses = {"gold": float(gold.detach()), "aux_gold": float(auxgold.detach())}
        for name in ("teach_g", "teach_l"):
            ids = draw(name, p[name]) if cfg["cross"] else p[name][:0]
            if len(ids):
                logp, logr = forward(ids, warm or cfg["gate_gold"])
                target = f["selected_pred"][ids]
                ce = F.nll_loss(logp, target)
                aux = F.nll_loss(logr, target) if cfg["aux"] else ce * 0
                loss = loss + cfg["cross"] * 0.5 * (ce + cfg["aux"] * aux)
                losses[name] = float(ce.detach())
            else:
                losses[name] = 0.0
        ids = draw("agreement", p["agreement"]) if cfg["keep"] else p["agreement"][:0]
        if len(ids):
            logp, _ = forward(ids, warm or cfg["gate_gold"])
            teacher = 0.5 * (f["logp_g"][ids].exp() + f["logp_l"][ids].exp())
            keep = F.kl_div(logp, teacher.detach(), reduction="batchmean")
            loss = loss + cfg["keep"] * keep
            losses["keep"] = float(keep.detach())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in model.parameters()):
            raise ValueError("Nonfinite gradient")
        gradients = {
            name: float(
                torch.linalg.vector_norm(
                    torch.stack([v.grad.norm() for v in group if v.grad is not None])
                )
            )
            if any(v.grad is not None for v in group)
            else 0.0
            for name, group in [("body", body), ("gate", gate)]
        }
        optimizer.step()
        if step % cfg["validate_every"] == 0 or step == cfg["max_steps"]:
            validate(step, losses, gradients)
            if step > cfg["warmup"] and stale >= cfg["patience"]:
                break
    return dict(
        config=cfg,
        training_seed=seed,
        split_seed=seed,
        inner_ensemble=False,
        states=states,
        checkpoints=checkpoints,
        validation_probability=valps["accuracy"],
        validation_probabilities=valps,
        validation_ids=s["selection_ids"].clone(),
        history=history,
        parameter_count=sum(v.numel() for v in model.parameters()),
        train_seconds=time.monotonic() - start,
        test_truth_loaded=False,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
