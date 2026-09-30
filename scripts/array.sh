#!/usr/bin/env bash
#SBATCH --job-name=fairrouter-cells
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
set -euo pipefail
EXTRA=()
if [[ -n "${FAIRROUTER_BANKS:-}" ]]; then EXTRA=(--banks "$FAIRROUTER_BANKS"); fi
if [[ -n "${FAIRROUTER_EXPERTS:-}" ]]; then EXTRA+=(--experts "$FAIRROUTER_EXPERTS"); fi
exec "$FAIRROUTER_ROOT/scripts/job.sh" -m fairrouter run \
    --bundle "$FAIRROUTER_BUNDLE" --output "$FAIRROUTER_OUTPUT/cells" \
    --index "$SLURM_ARRAY_TASK_ID" --mode "$FAIRROUTER_MODE" "${EXTRA[@]}"
