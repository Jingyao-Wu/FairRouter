from __future__ import annotations
import hashlib
from fairrouter.compatibility import input_specification
from collections.abc import Iterable

def mixed_assignment(node_ids: Iterable[int], *, seed: int) -> dict[int, str]:
    labels = ('clean', 'graph', 'text')
    namespace = input_specification()['assignment_namespace']
    result: dict[int, str] = {}
    for raw_node_id in node_ids:
        node_id = int(raw_node_id)
        digest = hashlib.sha256(f"{namespace}:{int(seed)}:{node_id}".encode()).digest()
        result[node_id] = labels[int.from_bytes(digest[:8], 'big') % len(labels)]
    return dict(sorted(result.items()))
