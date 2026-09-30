#!/usr/bin/env bash
# Run FairRouter across datasets, label budgets and seeds, then evaluate.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash run.sh [BUNDLE] [NEW_OUTPUT] [refit-all|refit-joint|replay]"
    echo "Defaults: data/frozen outputs/reproduction refit-all"
    echo "Set FAIRROUTER_PYTHON and FAIRROUTER_PARTITION for your Slurm cluster."
    exit 0
fi
if [[ $# -gt 3 ]]; then
    echo "Usage: bash run.sh [BUNDLE] [NEW_OUTPUT] [refit-all|refit-joint|replay]" >&2
    exit 1
fi
exec bash "$ROOT/scripts/submit.sh" \
    "${1:-$ROOT/data/frozen}" "${2:-$ROOT/outputs/reproduction}" "${3:-refit-all}"
