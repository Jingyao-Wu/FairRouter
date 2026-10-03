#!/usr/bin/env bash
# Submit all experiment settings and evaluation to Slurm.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash run.sh BUNDLE NEW_OUTPUT [refit-all|refit-joint|replay]"
    echo "Set FAIRROUTER_PYTHON and FAIRROUTER_PARTITION for your Slurm cluster."
    exit 0
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash run.sh BUNDLE NEW_OUTPUT [refit-all|refit-joint|replay]" >&2
    exit 1
fi
exec bash "$ROOT/scripts/submit.sh" "$1" "$2" "${3:-refit-all}"
