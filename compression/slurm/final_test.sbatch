#!/usr/bin/env bash
# Use the resource flags shown in README_UNITY.md. This job consumes an existing,
# validated export; no training or code updates occur here.
set -euo pipefail
MANIFEST=${1:?campaign manifest}
SELECTION=${2:?frozen selection JSON}
STAGE=${3:?prepare/evaluate/aggregate}
: "${TINY_CLM_VENV:?}"
SNAPSHOT=$("$TINY_CLM_VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$MANIFEST")
export TINY_CLM_COMMIT=$(cat "$SNAPSHOT/SNAPSHOT_COMMIT")
export PYTHONPATH="$SNAPSHOT/clm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
if [[ "$STAGE" != aggregate ]]; then module load apptainer/latest; fi
export TMPDIR="${TMPDIR:?}/final-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
export APPTAINER_TMPDIR="$TMPDIR" SINGULARITY_TMPDIR="$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$SNAPSHOT"
srun "$TINY_CLM_VENV/bin/python" "$SNAPSHOT/compression/final_test.py" --manifest "$MANIFEST" --selection "$SELECTION" --stage "$STAGE"
