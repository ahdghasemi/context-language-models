#!/usr/bin/env bash
# All resource requests are supplied explicitly by submit.py or resume.py.
# A direct sbatch call must supply resources and the two positional arguments.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
MANIFEST=${1:?absolute campaign manifest required}
STAGE=${2:?stage required}
: "${TINY_CLM_VENV:?source the same unity.env used at submission}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
if [[ "$STAGE" == prepare ]]; then
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
fi
# Official Unity Apptainer module; no CUDA module is assumed by this wheel workflow.
if [[ "$STAGE" == prepare || "$STAGE" == contextbench ]]; then
  module load apptainer/latest
fi
SNAPSHOT=$("$TINY_CLM_VENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' "$MANIFEST")
export TINY_CLM_COMMIT=$(cat "$SNAPSHOT/SNAPSHOT_COMMIT")
export PYTHONPATH="$SNAPSHOT/clm${PYTHONPATH:+:$PYTHONPATH}"
cd "$SNAPSHOT"
mkdir -p "${TMPDIR:?}"
export TMPDIR=$(mktemp -d "$TMPDIR/tclm-${SLURM_JOB_ID:-local}-XXXXXX")
export APPTAINER_TMPDIR="$TMPDIR" SINGULARITY_TMPDIR="$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
case "$STAGE" in
  correctness|smoke|prepare|aggregate) STEP_RESOURCE=--cpu ;;
  environment|train|export|evaluate|profile|contextbench) STEP_RESOURCE=--gpu ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
"$TINY_CLM_VENV/bin/python" -m tiny_clm.slurm "$STEP_RESOURCE" -- \
  "$TINY_CLM_VENV/bin/python" "$SNAPSHOT/compression/run_stage.py" "$MANIFEST" "$STAGE"
