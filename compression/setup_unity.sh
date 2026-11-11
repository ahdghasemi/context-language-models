#!/usr/bin/env bash
set -euo pipefail
: "${TINY_CLM_WORK:?source your configured unity.env first}"
: "${TINY_CLM_VENV:?}"
: "${TINY_CLM_TORCH_INDEX:?select a driver-compatible PyTorch wheel index}"
mkdir -p "$TINY_CLM_WORK" "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR" "$APPTAINER_CACHEDIR"
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -e "$TINY_CLM_VENV/READY.json" ]]; then
  echo "Environment is already frozen. Use a new TINY_CLM_VENV for updates." >&2
  exit 1
fi
"${TINY_CLM_PYTHON:-python3.12}" -m venv "$TINY_CLM_VENV"
"$TINY_CLM_VENV/bin/python" -m pip install --upgrade pip
"$TINY_CLM_VENV/bin/python" -m pip install torch==2.8.0 --index-url "$TINY_CLM_TORCH_INDEX"
"$TINY_CLM_VENV/bin/python" -m pip install -r "$ROOT/compression/requirements.txt"
# Non-editable package install. Jobs put their immutable snapshot first on PYTHONPATH.
"$TINY_CLM_VENV/bin/python" -m pip install "$ROOT[compression]"
"$TINY_CLM_VENV/bin/python" -m pip check
"$TINY_CLM_VENV/bin/python" -m pip freeze > "$TINY_CLM_VENV/freeze.txt"
"$TINY_CLM_VENV/bin/python" - <<'PY'
import hashlib,json,os,pathlib,sys
p=pathlib.Path(os.environ['TINY_CLM_VENV'])
(p/'READY.json').write_text(json.dumps({'python':sys.version,'freeze_sha256':hashlib.sha256((p/'freeze.txt').read_bytes()).hexdigest()},indent=2))
PY
echo "Environment created. No model downloads or Slurm jobs were submitted."
