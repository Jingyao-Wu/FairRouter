#!/usr/bin/env bash
#SBATCH --job-name=fairrouter-features
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
set -euo pipefail
EXTRA=()
if [[ -n "${FAIRROUTER_EXPERTS:-}" ]]; then EXTRA=(--experts "$FAIRROUTER_EXPERTS"); fi
exec "$FAIRROUTER_ROOT/scripts/job.sh" -m fairrouter.frontend \
    --bundle "$FAIRROUTER_BUNDLE" --frontend "$FAIRROUTER_FRONTEND" \
    --output "$FAIRROUTER_OUTPUT/banks" --index "$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"
