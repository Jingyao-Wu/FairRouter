#!/usr/bin/env bash
#SBATCH --job-name=fairrouter-experts
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
set -euo pipefail
exec "$FAIRROUTER_ROOT/scripts/job.sh" -m fairrouter.expert \
    --bundle "$FAIRROUTER_BUNDLE" --frontend "$FAIRROUTER_FRONTEND" \
    --output "$FAIRROUTER_OUTPUT/experts" --index "$SLURM_ARRAY_TASK_ID"
