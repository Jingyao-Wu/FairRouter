"""Read prepared inputs using the current module and metadata names."""

from contextlib import contextmanager
from functools import lru_cache
import importlib
from importlib.resources import files
import json
import sys
from threading import RLock

_MODULE_LOCK = RLock()


@lru_cache(maxsize=1)
def input_specification():
    resource = files("fairrouter").joinpath("resources/input_compatibility.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def normalize_metadata(value):
    """Translate serialized names without changing arrays or estimator parameters."""
    names = input_specification()["metadata_names"]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = names.get(key, key) if isinstance(key, str) else key
            if name in result:
                raise ValueError(f"Conflicting input metadata: {name}")
            result[name] = normalize_metadata(item)
        return result
    if isinstance(value, list):
        return [normalize_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_metadata(item) for item in value)
    if isinstance(value, str):
        return names.get(value, value)
    return value


@contextmanager
def input_modules():
    """Resolve class paths only while reading a prepared model file."""
    names = input_specification()["module_names"]
    missing = object()
    with _MODULE_LOCK:
        previous = {}
        try:
            for source, target in names.items():
                module = importlib.import_module(target)
                previous[source] = sys.modules.get(source, missing)
                sys.modules[source] = module
            yield
        finally:
            for source, module in previous.items():
                if module is missing:
                    sys.modules.pop(source, None)
                else:
                    sys.modules[source] = module


def load_estimator(path):
    """Load an already authenticated joblib file and normalize its metadata."""
    import joblib

    with input_modules():
        return normalize_metadata(joblib.load(path))


def canonical_split(split):
    """Keep one evaluation field alongside the support/validation/query partition."""
    fields = (
        "dataset", "shots", "split_seed", "num_nodes", "graph_sha256",
        "support_ids", "valid_ids", "unlabeled_ids", "standard_eval_ids",
        "test_truth_exported",
    )
    return {key: split[key] for key in fields if key in split}
