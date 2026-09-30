from __future__ import annotations
import re
from collections.abc import Mapping

class LeakageError(ValueError):
    """Raised when an artifact or fitting operation crosses a declared boundary."""
_ALLOWED_CACHE_KEYS = frozenset({'node_ids', 'hidden', 'metadata'})
_FORBIDDEN_PATTERN = re.compile('(?:^|_)(?:gold|label|target|correct|ground_truth|y_true)(?:$|_)', re.IGNORECASE)

def assert_label_free_cache(payload: Mapping[str, object]) -> None:
    keys = set(payload)
    unexpected = sorted(keys - _ALLOWED_CACHE_KEYS)
    missing = sorted(_ALLOWED_CACHE_KEYS - keys)
    forbidden = sorted((key for key in keys if _FORBIDDEN_PATTERN.search(str(key))))
    metadata = payload.get('metadata')
    if isinstance(metadata, Mapping):
        forbidden.extend((f'metadata.{key}' for key in metadata if _FORBIDDEN_PATTERN.search(str(key))))
    if forbidden or unexpected:
        raise LeakageError(f'cache has forbidden/unexpected fields: {sorted(set(forbidden + unexpected))}')
    if missing:
        raise LeakageError(f'cache contract missing fields: {missing}')
