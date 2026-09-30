"""Quality-aware routing with agreement, trust and preference scorers."""

import copy
from fractions import Fraction
import hashlib
import importlib
import json

import numpy as np
from scipy.special import expit, logit
import torch

from .artifacts import DATASETS

FEATURE_KEYS = {"dataset", "ids", "tab", "g", "l", "gnn_pred", "llm_pred"}
ENGINES = {
    "v6": "eargtc.router_aug_search.v6_engine",
    "v7": "eargtc.router_aug_search.v7_engine",
    "expanded": "eargtc.router_aug_search.v7_expanded",
    "tune": "eargtc.router_v10_tune.engine",
}


def source_key(config, head, sources, banks):
    """Content-based cache key for source training inputs."""
    config = {k: v for k, v in config.items() if k not in ("id", "target_steps")}
    digest = hashlib.sha256(json.dumps(dict(head=head, config=config), sort_keys=True).encode())
    for dataset in sources:
        digest.update(dataset.encode())
        for key in ("ids", "node_ids", "tab", "g", "l", "state"):
            value = np.ascontiguousarray(np.asarray(banks[dataset][key]))
            digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def engine(name):
    module = importlib.import_module(ENGINES[name])
    if name == "v6":
        module._source_key = source_key
    return module


def predict_head(wrapped, bank):
    return engine(wrapped["engine"]).predict_candidate(wrapped["model"], bank)


def raw(wrapped, bank):
    value = predict_head(wrapped, bank)
    if "gold_normalization" in wrapped:
        norm = wrapped["gold_normalization"]
        return norm["slope"] * value + norm["bias"]
    if wrapped["engine"] == "v6" and wrapped["model"]["head"] == "agreement":
        return logit(np.clip(value, 1e-6, 1 - 1e-6))
    model = wrapped["model"]
    return (value - model["center"]) / model["scale"] if "center" in model else value


def predict_branch(bundle, bank):
    if bundle.get("unavailable") or not len(bank["ids"]):
        n = len(bank["ids"])
        return dict(
            score=np.zeros(n), route=np.zeros(n, dtype=np.int64), q=np.tile([0.0, 0.0, 1.0], (n, 1))
        )
    if bundle.get("tuning_schema") != "crossfit_v1":
        if bundle["branch"] == "agreement":
            return dict(score=predict_head(bundle["models"]["agreement"], bank))
        z = {
            h: bundle["affine"][h]["slope"] * predict_head(bundle["models"][h], bank)
            + bundle["affine"][h]["bias"]
            for h in ("trust", "preference")
        }
        p, trust = expit(z["preference"]), expit(z["trust"])
        boundary = bundle["boundary"]
    elif bundle["branch"] == "agreement":
        z = raw(bundle["models"]["agreement"], bank)
        a = bundle["calibration"]["agreement"]
        return dict(score=expit(a["slope"] * z + a["bias"]))
    else:
        z = {h: raw(w, bank) for h, w in bundle["models"].items()}
        a, b = bundle["calibration"]["trust"], bundle["calibration"]["preference"]
        recipe = bundle["route_recipe"]
        trust = expit(
            recipe["trust_scale"] * (a["slope"] * z["trust"] + a["bias"]) + recipe["trust_bias"]
        )
        p = expit(b["slope"] * z["preference"] + b["bias"])
        boundary = recipe["boundary"]
    route = (p < boundary).astype(np.int64)
    return dict(
        score=trust * np.where(route == 0, p, 1 - p),
        route=route,
        q=np.column_stack((trust * p, trust * (1 - p), 1 - trust)),
    )


def top_mask(scores, ids, fraction):
    scores, ids = np.asarray(scores), np.asarray(ids)
    if (
        scores.shape != ids.shape
        or not np.isfinite(scores).all()
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Invalid score/ID vectors")
    ratio = Fraction(str(fraction))
    if not 0 <= ratio <= 1:
        raise ValueError("Fraction outside [0, 1]")
    count = (len(ids) * ratio.numerator + ratio.denominator - 1) // ratio.denominator
    mask = np.zeros(len(ids), dtype=bool)
    mask[np.lexsort((ids, -scores))[:count]] = True
    return mask


def load_model(bundle, digest):
    # joblib is trusted pickle: authenticate before deserializing.
    import joblib

    return joblib.load(bundle.object(digest))


def checked_features(bank, split, role, branch):
    if not set(bank) <= FEATURE_KEYS | {"split", "state", "target", "cal_mask", "policy_mask"}:
        raise ValueError("Unknown Router bank field")
    if role == "query" and not set(bank) <= FEATURE_KEYS | {"split"}:
        raise ValueError("Query labels forbidden")
    expected = split["valid_ids" if role == "validation" else "unlabeled_ids"]
    ids = bank["ids"].tolist()
    if len(ids) != len(set(ids)) or not set(ids) <= set(expected):
        raise ValueError("Router bank split mismatch")
    if not bool(((bank["gnn_pred"] == bank["llm_pred"]) == (branch == "agreement")).all()):
        raise ValueError("Router branch identity mismatch")
    return {k: bank[k] for k in FEATURE_KEYS}


def refit_branch(artifacts, cell, branch, deployed):
    """Fit the configured scorers on labeled support observations.

    Calibration and acceptance settings remain fixed. Training observations
    across datasets use the same label budget and split seed.
    """
    banks = {}
    for dataset in DATASETS:
        source = artifacts.cell(cell["shot"], cell["seed"], dataset)
        split = artifacts.split(source)
        bank = artifacts.tensor(source["branches"][branch]["train"])
        parents = set(bank["node_ids"].tolist())
        if bank.get("split") != "train" or not parents <= set(split["support_ids"]):
            raise ValueError("Router fit contains non-support parents")
        if parents & set(split["valid_ids"] + split["unlabeled_ids"]):
            raise ValueError("Router fit contains protected parents")
        if not np.array_equal(bank["weight"], np.ones(len(bank["ids"]))):
            raise ValueError("Gold views must have unit weights")
        banks[dataset] = bank
    fitted = copy.deepcopy(deployed)
    for head, wrapped in fitted.get("models", {}).items():
        wrapped["model"] = engine(wrapped["engine"]).fit_candidate(
            copy.deepcopy(wrapped["model"]["config"]), head, banks, cell["dataset"]
        )
    return fitted


def replay(artifacts, cell, upstream, *, refit=False):
    """Reconstruct every canonical Router tensor without reading test targets."""
    split = artifacts.split(cell)
    count = split["num_nodes"]
    gp, lp = upstream["gnn_logits"].argmax(1), upstream["llm_logits"].argmax(1)
    agreement = gp == lp
    choose = torch.ones(count, dtype=torch.bool)
    accepted = torch.zeros(count, dtype=torch.bool)
    scores = torch.zeros(count, dtype=torch.float64)
    audit = []
    models = {}
    for branch in ("agreement", "disagreement"):
        source = cell["branches"][branch]
        deployed = load_model(artifacts, source["model"])
        fitted = refit_branch(artifacts, cell, branch, deployed) if refit else deployed
        models[branch] = fitted
        for role in ("validation", "query"):
            bank = checked_features(artifacts.tensor(source[role]), split, role, branch)
            if role == "validation" and branch == "agreement":
                continue
            prediction = predict_branch(fitted, bank)
            if refit:
                reference = predict_branch(deployed, bank)
                for key in prediction:
                    np.testing.assert_allclose(prediction[key], reference[key], rtol=0, atol=1e-10)
            ids = bank["ids"]
            if not torch.equal(gp[ids], bank["gnn_pred"]) or not torch.equal(
                lp[ids], bank["llm_pred"]
            ):
                raise ValueError("Router bank experts differ from current upstream")
            scores[ids] = torch.as_tensor(prediction["score"], dtype=torch.float64)
            if branch == "disagreement":
                choose[ids] = torch.as_tensor(prediction["route"]) == 0
            if role == "query":
                accepted[ids] = torch.from_numpy(
                    top_mask(prediction["score"], ids, fitted["policy"]["fraction"])
                )
            audit.append(dict(branch=branch, role=role, rows=len(ids), refitted=refit))
    result = dict(
        node_ids=torch.arange(count),
        choose_g=choose,
        trust=torch.where(agreement, torch.zeros_like(scores), scores).float(),
        accepted=accepted,
        selected_pred=torch.where(choose, gp, lp),
        agreement=agreement,
        selected_score=scores,
    )
    reference = artifacts.tensor(cell["inputs"]["router"])
    for key in result:
        if key in reference:
            torch.testing.assert_close(
                result[key],
                reference[key],
                rtol=0,
                atol=1e-10 if result[key].dtype == torch.float64 else 0,
            )
    return result, audit, models
