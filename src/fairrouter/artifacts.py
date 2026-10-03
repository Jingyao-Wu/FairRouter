"""Load prepared data and validate model inputs and experiment outputs."""

import hashlib
import json

from .object_redirects import OBJECT_REDIRECTS
from .compatibility import canonical_split
from pathlib import Path

DATASETS = ("cora", "citeseer", "pubmed", "arxiv", "ogbn-products")
SHOTS = (3, 5, 10)
SEEDS = (42, 43, 44)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_tensor(path, value):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)


def seal(directory, metadata):
    directory = Path(directory)
    write_json(
        directory / "FROZEN.json",
        dict(
            metadata,
            files={
                str(p.relative_to(directory)): sha256(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            },
        ),
    )


def verify_seal(directory):
    directory = Path(directory)
    sealed = read_json(directory / "FROZEN.json")
    for name, expected in sealed["files"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Changed frozen run file: {name}")
    return sealed


class Bundle:
    """A content-addressed bundle; all paths must remain inside its root."""

    def __init__(self, root, bank_root=None, expert_root=None):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "manifest.json")
        if self.manifest["schema"] != 1:
            raise ValueError("Unknown bundle schema")
        self.checked = set()
        self.bank_overrides = {}
        expected = {(s, r, d) for s in SHOTS for r in SEEDS for d in DATASETS}
        actual = [(c["shot"], c["seed"], c["dataset"]) for c in self.manifest["cells"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Bundle must contain exactly 45 unique cells")
        if bank_root is not None:
            bank_root = Path(bank_root).resolve()
            for cell in self.manifest["cells"]:
                directory = bank_root / f"shot{cell['shot']}/seed{cell['seed']}/{cell['dataset']}"
                frozen = read_json(directory / "FROZEN.json")
                if frozen["test_truth_loaded"] is not False or frozen[
                    "bundle_manifest_sha256"
                ] != sha256(self.root / "manifest.json"):
                    raise ValueError("Unsafe or mismatched reconstructed banks")
                if (frozen["shot"], frozen["seed"], frozen["dataset"]) != (
                    cell["shot"],
                    cell["seed"],
                    cell["dataset"],
                ):
                    raise ValueError("Reconstructed bank identity differs")
                expected_objects = {
                    cell["branches"][b][r]
                    for b in ("agreement", "disagreement")
                    for r in ("train", "validation", "query")
                }
                if set(frozen["objects"]) != expected_objects:
                    raise ValueError("Reconstructed banks are incomplete")
                for original, name in frozen["objects"].items():
                    path = (directory / name).resolve()
                    if not path.is_relative_to(directory):
                        raise ValueError("Reconstructed bank path escapes cell")
                    self.bank_overrides[original] = (path, frozen["files"][name])

        if expert_root is not None:
            expert_root = Path(expert_root).resolve()
            for cell in self.manifest["cells"]:
                directory = expert_root / f"shot{cell['shot']}/seed{cell['seed']}/{cell['dataset']}"
                frozen = read_json(directory / "FROZEN.json")
                if (frozen["shot"], frozen["seed"], frozen["dataset"]) != (
                    cell["shot"],
                    cell["seed"],
                    cell["dataset"],
                ):
                    raise ValueError("Rebuilt expert identity differs")
                if not frozen["exact_logits"] or frozen["test_truth_loaded"] is not False:
                    raise ValueError(
                        "Only exact, label-isolated expert refits may replace frozen inputs"
                    )
                if frozen["bundle_manifest_sha256"] != sha256(self.root / "manifest.json"):
                    raise ValueError("Expert refit belongs to another bundle")
                self.bank_overrides[cell["inputs"]["upstream"]] = (
                    directory / "upstream.pt",
                    frozen["files"]["upstream.pt"],
                )

    def authenticated(self, name, digest):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact path escapes bundle")
        key = (str(path), digest)
        if key not in self.checked:
            if sha256(path) != digest:
                raise ValueError(f"Artifact hash mismatch: {name}")
            self.checked.add(key)
        return path

    def object(self, digest):
        record = self.manifest["objects"][digest]
        if record["sha256"] != digest:
            raise ValueError("Object identity differs")
        if digest in OBJECT_REDIRECTS:
            # Resolve normalized metadata copies through the fixed object mapping.
            replacement = OBJECT_REDIRECTS[digest]
            return self.authenticated(f"objects/{replacement}.pt", replacement)
        return self.authenticated(record["path"], digest)

    def tensor(self, digest):
        import torch

        if digest in self.bank_overrides:
            path, expected = self.bank_overrides[digest]
            key = (str(path), expected)
            if key not in self.checked:
                if sha256(path) != expected:
                    raise ValueError("Reconstructed bank changed")
                self.checked.add(key)
        else:
            path = self.object(digest)
        return torch.load(path, weights_only=True, map_location="cpu", mmap=True)

    def split(self, cell):
        split = canonical_split(read_json(self.authenticated(cell["split"], cell["split_sha256"])))
        check_split(split, cell)
        reference = self.cell(10, cell["seed"], cell["dataset"])
        standard = read_json(self.authenticated(reference["split"], reference["split_sha256"]))
        if split["standard_eval_ids"] != standard["standard_eval_ids"]:
            raise ValueError("Evaluation IDs and their order must match across label budgets")
        return split

    def cell(self, shot, seed, dataset):
        return next(
            c
            for c in self.manifest["cells"]
            if (c["shot"], c["seed"], c["dataset"]) == (shot, seed, dataset)
        )


def check_split(split, cell):
    if (split["shots"], split["split_seed"], split["dataset"]) != (
        cell["shot"],
        cell["seed"],
        cell["dataset"],
    ):
        raise ValueError("Split identity mismatch")
    if split["test_truth_exported"] is not False:
        raise ValueError("Unsafe split schema")
    groups = [set(split[k]) for k in ("support_ids", "valid_ids", "unlabeled_ids")]
    if any(
        len(split[k]) != len(g)
        for k, g in zip(("support_ids", "valid_ids", "unlabeled_ids"), groups)
    ):
        raise ValueError("Duplicate split IDs")
    if any(groups[a] & groups[b] for a in range(3) for b in range(a)):
        raise ValueError("Support/validation/query overlap")
    if set.union(*groups) != set(range(split["num_nodes"])):
        raise ValueError("Incomplete graph partition")
    ids = split["standard_eval_ids"]
    if len(ids) != len(set(ids)) or len(ids) != 1000 or not set(ids) <= groups[2]:
        raise ValueError("Standard eval must contain exactly 1000 distinct query nodes")
