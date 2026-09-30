#!/usr/bin/env bash
#SBATCH --job-name=fairrouter
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Slurm copies this file into its spool. Export FAIRROUTER_ROOT when submitting.
PROJECT_ROOT="${FAIRROUTER_ROOT:-${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "${FAIRROUTER_PYTHON:-python}" -B -u "$@"
