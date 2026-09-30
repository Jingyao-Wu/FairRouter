#!/usr/bin/env bash
# Submit all 45 independent cells, followed by evaluation only if all succeed.
set -euo pipefail
export FAIRROUTER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="${1:?Usage: submit.sh BUNDLE OUTPUT [replay|refit-joint|refit-all]}"
OUTPUT="${2:?Supply a new output directory}"
MODE="${3:-refit-joint}"
PARTITION="${FAIRROUTER_PARTITION:-cpu}"
BUNDLE="$(realpath "$BUNDLE")"
if [[ ! -f "$BUNDLE/manifest.json" ]]; then
    echo "Missing frozen artifact bundle. Prepare it as described in README.md." >&2
    exit 1
fi
case "$MODE" in
    replay|refit-joint|refit-all) ;;
    *) echo "Mode must be replay, refit-joint, or refit-all" >&2; exit 1 ;;
esac
if [[ -n "${FAIRROUTER_FRONTEND:-}" && ! -f "$FAIRROUTER_FRONTEND/frontend_manifest.json" ]]; then
    echo "Missing frontend artifact manifest. See README.md." >&2
    exit 1
fi
OUTPUT="$(realpath -m "$OUTPUT")"
if [[ -e "$OUTPUT" ]]; then
    echo "Refusing existing run directory: $OUTPUT" >&2
    exit 1
fi
mkdir -p "$OUTPUT/logs"
export FAIRROUTER_BUNDLE="$BUNDLE" FAIRROUTER_OUTPUT="$OUTPUT" FAIRROUTER_MODE="$MODE"
DEPENDENCY=()
EVALUATION_EXTRA=()
if [[ "${FAIRROUTER_REFIT_EXPERTS:-0}" == "1" ]]; then
    : "${FAIRROUTER_FRONTEND:?Set FAIRROUTER_FRONTEND for expert refits}"
    : "${FAIRROUTER_GPU_PARTITION:?Set a GPU partition for expert refits}"
    export FAIRROUTER_FRONTEND="$(realpath "$FAIRROUTER_FRONTEND")"
    : "${FAIRROUTER_GPU_PARTITION_5SHOT:?Set a GPU partition for 5-shot expert refits}"
    EXPERT_JOB=$(sbatch --parsable --partition="$FAIRROUTER_GPU_PARTITION" --array="0-14,30-44%${FAIRROUTER_GPU_CONCURRENCY:-2}" \
        --output="$OUTPUT/logs/experts-%A_%a.log" "$FAIRROUTER_ROOT/scripts/expert_array.sh")
    EXPERT_FIVE=$(sbatch --parsable --partition="$FAIRROUTER_GPU_PARTITION_5SHOT" --array="15-29%${FAIRROUTER_GPU_CONCURRENCY:-2}" \
        --output="$OUTPUT/logs/experts-%A_%a.log" "$FAIRROUTER_ROOT/scripts/expert_array.sh")
    DEPENDENCY=(--dependency="afterok:$EXPERT_JOB:$EXPERT_FIVE")
    export FAIRROUTER_EXPERTS="$OUTPUT/experts"
    printf 'experts=%s\nexperts_5shot=%s\n' "$EXPERT_JOB" "$EXPERT_FIVE" > "$OUTPUT/expert_job.txt"
fi
if [[ -n "${FAIRROUTER_EXPERTS:-}" ]]; then
    EVALUATION_EXTRA=(--experts "$FAIRROUTER_EXPERTS")
fi
if [[ -n "${FAIRROUTER_FRONTEND:-}" ]]; then
    export FAIRROUTER_FRONTEND="$(realpath "$FAIRROUTER_FRONTEND")"
    FRONTEND_JOB=$(sbatch "${DEPENDENCY[@]}" --parsable --partition="$PARTITION" --array="0-44%${FAIRROUTER_CONCURRENCY:-4}" \
        --output="$OUTPUT/logs/features-%A_%a.log" "$FAIRROUTER_ROOT/scripts/frontend_array.sh")
    DEPENDENCY=(--dependency="afterok:$FRONTEND_JOB")
    export FAIRROUTER_BANKS="$OUTPUT/banks"
    printf 'frontend=%s\n' "$FRONTEND_JOB" > "$OUTPUT/frontend_job.txt"
fi
JOB=$(sbatch "${DEPENDENCY[@]}" --parsable --partition="$PARTITION" --array="0-44%${FAIRROUTER_CONCURRENCY:-4}" \
    --output="$OUTPUT/logs/cell-%A_%a.log" "$FAIRROUTER_ROOT/scripts/array.sh")
EVALUATION=$(sbatch --parsable --partition="$PARTITION" --dependency="afterok:$JOB" \
    --output="$OUTPUT/logs/evaluate-%j.log" "$FAIRROUTER_ROOT/scripts/job.sh" \
    -m fairrouter evaluate --bundle "$BUNDLE" --run "$OUTPUT/cells" \
    --output "$OUTPUT/evaluation" "${EVALUATION_EXTRA[@]}")
printf 'cells=%s\nevaluation=%s\n' "$JOB" "$EVALUATION" | tee "$OUTPUT/jobs.txt"
