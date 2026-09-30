"""Compute probability, structural and gold-prototype routing evidence.

Prepared expert representations provide the clean and five augmented views.
"""

import argparse
import os
from pathlib import Path

import torch

from eargtc.models.gnn import MLPHead
from eargtc.router_augmentation_v8.data import ENVIRONMENTS, organize
from eargtc.router_v7.perturbations import mixed_assignment
from eargtc.router_v8.prototypes import fit_class_prototypes
from .artifacts import (
    Bundle,
    DATASETS,
    SEEDS,
    SHOTS,
    read_json,
    require_slurm,
    save_tensor,
    seal,
    sha256,
)
from .features import build70


class Frontend:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "frontend_manifest.json")
        self.checked = set()

    def tensor(self, digest):
        require_slurm()
        item = self.manifest["objects"][digest]
        path = (self.root / item["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Frontend path escapes root")
        if digest not in self.checked:
            if sha256(path) != digest:
                raise ValueError("Frontend hash mismatch")
            self.checked.add(digest)
        return torch.load(path, weights_only=True, map_location="cpu", mmap=True)


def positions(ids, wanted):
    lookup = {int(n): i for i, n in enumerate(ids)}
    return torch.tensor([lookup[int(n)] for n in wanted], dtype=torch.long)


def build_banks(bundle_root, frontend_root, output, shot, seed, dataset, expert_root=None):
    require_slurm()
    torch.set_num_threads(2)
    artifacts, front = Bundle(bundle_root, expert_root=expert_root), Frontend(frontend_root)
    cell = artifacts.cell(shot, seed, dataset)
    rec = next(
        r
        for r in front.manifest["cells"]
        if (r["shot"], r["seed"], r["dataset"]) == (shot, seed, dataset)
    )
    split = artifacts.split(cell)
    destination = Path(output) / f"shot{shot}/seed{seed}/{dataset}"
    if destination.exists():
        raise FileExistsError("Refusing existing reconstructed bank directory")
    destination.mkdir(parents=True)
    raw = artifacts.tensor(cell["inputs"]["upstream"])
    sup = artifacts.tensor(cell["inputs"]["supervision"])
    g = artifacts.tensor(cell["inputs"]["embeddings"])["embeddings"]
    hidden = artifacts.tensor(cell["inputs"]["hidden"])["hidden"]
    graph = front.tensor(rec["inputs"]["graph"])
    n = split["num_nodes"]
    if graph["num_nodes"] != n:
        raise ValueError("Graph identity mismatch")
    edge = graph["edge_index"]
    ids, y = sup["support_ids"], sup["support_targets"]
    if ids.tolist() != split["support_ids"]:
        raise ValueError("Gold identity mismatch")
    pg, pl = raw["gnn_logits"].float().softmax(1), raw["llm_logits"].float().softmax(1)
    classes = pg.shape[1]
    fitted = fit_class_prototypes(g, ids, y, num_classes=classes)
    full_pred = pg.argmax(1)
    clean = build70(pg[ids], pl[ids], ids, edge, full_pred, g, fitted, n, classes)
    if shot == 5:
        # Check feature agreement before using the supplied reference precision.
        for branch in ("agreement", "disagreement"):
            reference = front.tensor(rec["inputs"]["clean_" + branch])
            pos = positions(ids, reference["ids"])
            generated = clean[pos]
            if branch == "agreement":
                generated = torch.cat((generated[:, :27], generated[:, 54:]), 1)
            torch.testing.assert_close(generated, reference["tab"], atol=2e-6, rtol=2e-6)
            clean[pos] = (
                torch.cat(
                    (reference["tab"][:, :27], reference["tab"][:, :27], reference["tab"][:, 27:]),
                    1,
                )
                if branch == "agreement"
                else reference["tab"]
            )
    else:
        if expert_root is None:
            state = front.tensor(rec["inputs"]["head"])
        else:
            directory = Path(expert_root) / f"shot{shot}/seed{seed}/{dataset}"
            metadata = read_json(directory / "FROZEN.json")
            if sha256(directory / "head.pt") != metadata["files"]["head.pt"]:
                raise ValueError("Refitted expert state changed")
            state = torch.load(directory / "head.pt", weights_only=True, map_location="cpu")
        head = MLPHead(
            128, state["net.0.weight"].shape[0], state["net.3.weight"].shape[0], dropout=0.0
        )
        head.load_state_dict(state)
        head.eval()
        with torch.inference_mode():
            torch.testing.assert_close(head(g), raw["gnn_logits"], atol=2e-5, rtol=2e-5)
    assigned = mixed_assignment(range(n), seed=303)
    environments = []
    for name in ENVIRONMENTS:
        eg, el = g[ids].clone(), hidden[ids].clone()
        epg, epl, tab = pg[ids].clone(), pl[ids].clone(), clean.clone()
        if name != "clean":
            replay = front.tensor(rec["replay"][name])
            if shot == 5:
                if not torch.equal(replay["node_ids"], ids):
                    raise ValueError("Perturbation parent order mismatch")
                changed = replay["changed_mask"]
                if name.startswith("graph"):
                    epg = replay["gnn_prob"].clone()
                    epg[~changed] = pg[ids][~changed]
                    eg = replay["g"]
                    proto_g = g.clone()
                    proto_g[ids] = eg
                    tab = build70(
                        epg,
                        epl,
                        ids,
                        replay["edge_index"],
                        replay["full_gnn_pred"],
                        proto_g,
                        fitted,
                        n,
                        classes,
                    )
                else:
                    epl = replay["llm_prob"].clone()
                    epl[~changed] = pl[ids][~changed]
                    el = replay["l"]
                    tab = build70(epg, epl, ids, edge, full_pred, g, fitted, n, classes)
            elif name.startswith("text"):
                pos = positions(replay["node_ids"], ids)
                changed = replay["changed_mask"][pos]
                el = replay["l"][pos]
                epl = replay["llm_prob"][pos].clone()
                epl[~changed] = pl[ids][~changed]
                tab = build70(epg, epl, ids, edge, full_pred, g, fitted, n, classes)
            else:
                shifted = replay["g"]
                changed = (
                    torch.tensor([assigned[int(i)] == "graph" for i in ids])
                    if "mixed" in name
                    else torch.ones(len(ids), dtype=torch.bool)
                )
                with torch.inference_mode():
                    prob = head(shifted).softmax(1)
                eg = shifted[ids].clone()
                eg[~changed] = g[ids][~changed]
                epg = prob[ids].clone()
                epg[~changed] = pg[ids][~changed]
                proto_g = g.clone()
                proto_g[ids] = eg
                tab = build70(
                    epg, epl, ids, replay["edge_index"], prob.argmax(1), proto_g, fitted, n, classes
                )
            tab[~changed] = clean[~changed]
        environments.append(
            dict(environment=name, node_ids=ids, gnn_prob=epg, llm_prob=epl, tab70=tab, g=eg, l=el)
        )
    assembled = organize(dataset, environments, gold_ids=ids, gold_targets=y, num_nodes=n)
    mapping = {}
    max_error = 0.0
    for branch in ("agreement", "disagreement"):
        expected = artifacts.tensor(cell["branches"][branch]["train"])
        actual = assembled[branch]
        if set(actual) != set(expected):
            raise ValueError("Reconstructed training schema differs")
        for key, value in actual.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
            elif value != expected[key]:
                raise ValueError("Training bank metadata differs")
        name = f"{branch}_train.pt"
        save_tensor(destination / name, actual)
        mapping[cell["branches"][branch]["train"]] = name
    for role, id_key in [("validation", "valid_ids"), ("query", "unlabeled_ids")]:
        rows = torch.tensor(split[id_key])
        tab70 = build70(pg[rows], pl[rows], rows, edge, full_pred, g, fitted, n, classes)
        for branch in ("agreement", "disagreement"):
            mask = (pg[rows].argmax(1) == pl[rows].argmax(1)) == (branch == "agreement")
            tab = tab70[mask]
            if branch == "agreement":
                tab = torch.cat((tab[:, :27], tab[:, 54:]), 1)
            generated = dict(
                ids=rows[mask],
                tab=tab.float().contiguous(),
                g=g[rows][mask].float().contiguous(),
                l=hidden[rows][mask].half().contiguous(),
                gnn_pred=full_pred[rows][mask],
                llm_pred=pl.argmax(1)[rows][mask],
            )
            expected = artifacts.tensor(cell["branches"][branch][role])
            for key, value in generated.items():
                torch.testing.assert_close(
                    value,
                    expected[key],
                    atol=2e-6 if key == "tab" else 0,
                    rtol=2e-6 if key == "tab" else 0,
                )
                if key == "tab" and value.numel():
                    max_error = max(max_error, float((value - expected[key]).abs().max()))
            # Preserve metadata and validation fold masks. Features come from reconstruction.
            actual = dict(expected, **generated)
            name = f"{branch}_{role}.pt"
            save_tensor(destination / name, actual)
            mapping[cell["branches"][branch][role]] = name
    seal(
        destination,
        dict(
            dataset=dataset,
            shot=shot,
            seed=seed,
            test_truth_loaded=False,
            exact_training_banks=True,
            evaluation_feature_max_abs_error=max_error,
            objects=mapping,
            bundle_manifest_sha256=sha256(Path(bundle_root) / "manifest.json"),
            frontend_manifest_sha256=sha256(Path(frontend_root) / "frontend_manifest.json"),
            slurm_job_id=os.environ["SLURM_JOB_ID"],
        ),
    )
    print("BANKS FROZEN", shot, seed, dataset, max_error, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    cells = [(s, r, d) for s in SHOTS for r in SEEDS for d in DATASETS]
    if not 0 <= args.index < len(cells):
        parser.error("Index must be 0..44")
    build_banks(
        args.bundle, args.frontend, args.output, *cells[args.index], expert_root=args.experts
    )


if __name__ == "__main__":
    main()
