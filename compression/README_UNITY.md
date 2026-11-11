# Unity / Slurm operating instructions

This workflow is implemented but **has not been submitted on Unity**. CPU tests were
executed elsewhere. `gpu`, `gpu-preempt`, `cpu` partition terminology and
`module load apptainer/latest` were checked against official Unity documentation on
2026-11-11. Your account association, permitted QoS, available GPU feature names,
driver compatibility, quotas and fakeroot/Singularity runtime remain unverified.
No account, QoS or GPU type is invented or silently selected.

Official references:
[sbatch](https://docs.unity.rc.umass.edu/documentation/jobs/sbatch/),
[GPUs](https://docs.unity.rc.umass.edu/documentation/tools/gpus/),
[Unity documentation](https://docs.unity.rc.umass.edu/),
[PyTorch wheels](https://pytorch.org/get-started/previous-versions/).
Do not run GPU training on a login node. The CLI rejects it outside an allocation.

## 1. Retrieve and configure once

Use a new checkout directory; preserve an existing checkout's uncommitted work. The
commands below fetch the implementation branch, then detach at its resolved commit.
For the delivered version, substitute the final commit given in the delivery report
for `TINY_CLM_REV` if the branch has since advanced.

```bash
cd /work/pi_aghasemi_umass_edu
git clone --branch tiny-clm-constrained https://github.com/ahdghasemi/context-language-models.git
cd context-language-models
TINY_CLM_REV=$(git rev-parse HEAD)
git checkout --detach "$TINY_CLM_REV"
git status --short
git rev-parse HEAD
cp compression/unity.env.example /work/pi_aghasemi_umass_edu/tiny-clm.env
```

Edit `/work/pi_aghasemi_umass_edu/tiny-clm.env` once. Paths may all be changed, including
project, work, assets, snapshots, campaigns/results/checkpoints/logs, Python environment,
HF/dataset/pip/container caches and temporary storage. Nothing large defaults into home.
Set `TINY_CLM_GPU_CONSTRAINT` to an actual available feature for BF16-capable hardware
with at least 24 GiB VRAM (32+ preferable; requirements are estimates). Account/QoS may
remain empty only if your association permits it. Do not add a CUDA module to a binary
wheel environment without a concrete compatibility reason.

Exact diagnostics to determine user-specific settings:

```bash
sinfo -o '%P %a %l %D %G %f'
sinfo -N -o '%N %P %G %m %f'
sacctmgr show assoc where user="$USER" format=Cluster,Account,User,Partition,QOS
unity-slurm-gpu-usage
unity-slurm-partition-usage
module spider Python
module spider apptainer
module load apptainer/latest
apptainer --version
command -v singularity
python3.12 --version
df -h /work/pi_aghasemi_umass_edu
test -w /work/pi_aghasemi_umass_edu
```

If `sacctmgr` access is restricted, obtain your association from Unity support; that
command failure does not establish an account name. If Python 3.12 is absent, select
an available Python 3.12 or 3.13 using the actual `module spider` output and update
`TINY_CLM_PYTHON`. A valid environment must satisfy this repository's Python range.
Use the filesystem's documented quota tooling as applicable; `df` does not show your quota.

```bash
source /work/pi_aghasemi_umass_edu/tiny-clm.env
cd "$TINY_CLM_ROOT"
bash compression/setup_unity.sh
source "$TINY_CLM_VENV/bin/activate"
python -m pip check
```

Setup installs PyTorch 2.8.0 from the explicitly configured index (example CUDA 12.8),
then pinned compression/protocol dependencies and a non-editable package. It freezes the
resolved environment and writes `READY.json`. It **does not submit anything or download
models/data**. An existing frozen environment is not modified; updates require a new
`TINY_CLM_VENV`. Every stage checks the freeze hash. Driver compatibility is checked by
the allocated environment job, not assumed from a successful pip install.

## 2. Environment check and inexpensive correctness

All submission drivers default to **dry run**. Read their complete printed resource
requests and job graph before adding `--submit`. CPU/memory/time/account/QoS/GPU
constraint/concurrency are environment-controlled; small correctness/smoke/aggregation
jobs request 2 CPUs, 8G and 30 minutes. Exactly one GPU is supported; allocating more
is rejected because distributed training is not implemented or needed for this pilot.

```bash
python compression/submit.py environment --campaign "$TINY_CLM_RUNS/env-check"
python compression/submit.py environment --campaign "$TINY_CLM_RUNS/env-check" --submit
python compression/submit.py correctness --campaign "$TINY_CLM_RUNS/correctness"
python compression/submit.py correctness --campaign "$TINY_CLM_RUNS/correctness" --submit
squeue -u "$USER"
tail -n 80 "$TINY_CLM_RUNS/env-check"/logs/*.out
tail -n 80 "$TINY_CLM_RUNS/env-check"/logs/*.err
```

The environment job records actual GPU/VRAM/runtime and exercises CUDA GEMM/backward/SVD.
It requires one visible BF16-capable GPU. Correctness runs pytest, then a separate CPU
random-weight end-to-end smoke only after tests succeed. Inspect `correctness.xml`, smoke
JSONs and Slurm exit states. Tiny smoke numbers are not research results.

## 3. Prepare pinned public assets and task containers

Default model/dataset IDs and immutable revisions are in `configs/pilot.json`. Preparation
downloads the model/tokenizer once, packs native data splits, hashes files, clones the
pinned ContextBench source, builds an Apptainer image from the pinned base recipe and
packages original tasks. It tests Harbor's actual isolation flags using the `singularity`
compatibility command. This requires working fakeroot support on the allocated CPU node.
No login-node container execution or substitute grader is used.

```bash
python compression/submit.py prepare --config compression/configs/pilot.json \
  --campaign "$TINY_CLM_RUNS/prepare-pilot"
python compression/submit.py prepare --config compression/configs/pilot.json \
  --campaign "$TINY_CLM_RUNS/prepare-pilot" --submit
squeue -u "$USER"
tail -n 100 "$TINY_CLM_RUNS/prepare-pilot"/logs/*.out
tail -n 100 "$TINY_CLM_RUNS/prepare-pilot"/logs/*.err
```

Wait for successful preparation before pilot submission. Asset and task lock files are
required. Exclusive locks reject simultaneous writers instead of corrupting downloads.
Partial task preparation without a final lock fails clearly; preserve the directory for
diagnosis and rename it before retrying. A different model/data request needs a new
`TINY_CLM_ASSETS`; different ContextBench settings get a distinct hashed task directory.
Do not modify shared sources, images or lock files used by active jobs. Gated alternatives
require your authorized HF credentials via the normal token mechanism; never place tokens
in Git, config JSON, logs or an sbatch command line. Default assets were public during
local verification. Data/model licenses remain those of their providers.

## 4. Pilot, logs and recovery

```bash
python compression/submit.py pilot --config compression/configs/pilot.json \
  --campaign "$TINY_CLM_RUNS/pilot-001"
python compression/submit.py pilot --config compression/configs/pilot.json \
  --campaign "$TINY_CLM_RUNS/pilot-001" --submit
export TINY_CLM_CAMPAIGN="$TINY_CLM_RUNS/pilot-001"
cat "$TINY_CLM_CAMPAIGN/jobs.json"
squeue -u "$USER"
tail -n 100 "$TINY_CLM_CAMPAIGN"/logs/train-*.out
```

The graph is environment → correctness → smoke → training array; export uses
`aftercorr` on the corresponding successful training task; evaluation/profile/ContextBench
use `aftercorr` on export. Aggregation waits `afterany` on all three tails, records missing
runs and exits nonzero if anything is incomplete. `--kill-on-invalid-dep=yes` prevents
failed parents leaving permanently pending children. Each stage array has concurrency
cap `TINY_CLM_CONCURRENCY` (2 by default); this is a per-array cap, not a global cap across
evaluation/profile/ContextBench arrays. Reduce to 1 if resource pressure warrants it.

Each row has a unique `run_id`; `manifest.json` maps row indices to configs, immutable
code snapshot, environment hash, asset paths and commit. `runs/<run_id>/` holds training
logs, metadata, latest checkpoint, status, export, deployment gap, evaluation, profiling
and ContextBench trials. Slurm stdout/stderr names contain job/array IDs.

```bash
python - <<'PY'
import json,os
p=os.environ['TINY_CLM_CAMPAIGN']
m=json.load(open(p+'/manifest.json'))
for i,r in enumerate(m['rows']): print(i,r['run_id'])
print('Snapshot:',m['snapshot'])
print('Jobs:',','.join(json.load(open(p+'/jobs.json')).values()))
PY
sacct -j "$(python -c 'import json,os;print(",".join(json.load(open(os.environ["TINY_CLM_CAMPAIGN"]+"/jobs.json")).values()))')" \
  --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

Training checkpoints every 10 optimizer updates and at stage boundaries. State includes
model, optimizer, scheduler, projected proxies/ranks/μ, best validation model/score,
sampler permutation/position/generator, Python/NumPy/Torch/CUDA RNGs, counters and config.
Writes use same-directory temporary files, fsync and atomic replace. The Slurm advance
`USR1` signal arrives 180 seconds before termination; the Python process catches it and
saves after completing the current optimizer update, exiting 75 so export stays blocked.
`TERM` is handled similarly. If an update/checkpoint exceeds the warning interval, or
SIGKILL/node failure occurs, restart from the last completed atomic checkpoint. Exact GPU
bitwise resumption is not promised; CPU same-environment restart was tested bitwise.
No automatic requeue hides a failed or incomplete experiment.

For an interrupted training row, e.g. index 8 (inspect your manifest first):

```bash
python compression/resume.py --manifest "$TINY_CLM_CAMPAIGN/manifest.json" --indices 8 --from-stage train
python compression/resume.py --manifest "$TINY_CLM_CAMPAIGN/manifest.json" --indices 8 --from-stage train --submit
```

The resume driver retains the old snapshot, rejects active original jobs and requires
an incomplete checkpoint. Completed training with a failed export uses `--from-stage export`;
single failed downstream stages use `evaluate`, `profile` or `contextbench`. Keep the old
logs/failure record. ASVD sensitivity initialization is not checkpointed mid-search; a
failure before the first checkpoint needs a new run directory/campaign and its cost
must be retained. For ContextBench, successful episode records are reused with identity
checks; failed records remain failed. Before an explicit retry, move that episode's
JSON, requests/server/Harbor logs and matching `trials/<episode>/` into an audit directory.
Do not delete evidence or mix old requests into a new episode. Mid-episode exact resumption
is not implemented.

After all resumed jobs finish, aggregate using the campaign's code snapshot:

```bash
TINY_CLM_SNAPSHOT=$(python -c 'import json,os;print(json.load(open(os.environ["TINY_CLM_CAMPAIGN"]+"/manifest.json"))["snapshot"])')
PYTHONPATH="$TINY_CLM_SNAPSHOT/clm" python "$TINY_CLM_SNAPSHOT/compression/aggregate.py" \
  --manifest "$TINY_CLM_CAMPAIGN/manifest.json"
PYTHONPATH="$TINY_CLM_SNAPSHOT/clm" python "$TINY_CLM_SNAPSHOT/compression/plot.py" \
  --summary "$TINY_CLM_CAMPAIGN/summary"
cat "$TINY_CLM_CAMPAIGN/summary/gate.json"
```

Outputs are `runs.csv`, `runs.json`, paired CLM intervals, gate JSON and trade-off PNG/PDF.
No missing value is fabricated. Inspect the plotted workload/hardware before comparing
runtime. Run export/evaluation/profiling as allocated jobs through the driver; their exact
CLI invocations are printed in Slurm logs and can be used inside a diagnostic allocation.

## 5. Main grid and focused ablations

Do not proceed unless pilot correctness and quality gates pass. Main tasks require their
own preparation (same model/data asset lock, different ContextBench task lock).

```bash
python compression/submit.py prepare --config compression/configs/main.json \
  --campaign "$TINY_CLM_RUNS/prepare-main" --submit
# Wait and inspect preparation exit status and logs.
python compression/submit.py main --config compression/configs/main.json \
  --pilot-gate "$TINY_CLM_RUNS/pilot-001/summary/gate.json" --campaign "$TINY_CLM_RUNS/main-001"
python compression/submit.py main --config compression/configs/main.json \
  --pilot-gate "$TINY_CLM_RUNS/pilot-001/summary/gate.json" --campaign "$TINY_CLM_RUNS/main-001" --submit
python compression/submit.py ablations --config compression/configs/pilot.json \
  --pilot-gate "$TINY_CLM_RUNS/pilot-001/summary/gate.json" --campaign "$TINY_CLM_RUNS/ablations-001"
# Add --submit to the reviewed ablation command only when ready.
```

Main is 31 runs; ablations are 9. A `--submit` main/ablation request without a passing
pilot gate is rejected. This is a screening guard, not a scientific guarantee.

## 6. Freeze selection and evaluate held-out data

After validation analysis, include dense controls and selected compressed runs. The
following example IDs exist in the main grid; they are **not a recommendation or an
observed best model**. Replace the list/rationale with your validation decision.

```bash
export TINY_CLM_CAMPAIGN="$TINY_CLM_RUNS/main-001"
TINY_CLM_SNAPSHOT=$(python -c 'import json,os;print(json.load(open(os.environ["TINY_CLM_CAMPAIGN"]+"/manifest.json"))["snapshot"])')
PYTHONPATH="$TINY_CLM_SNAPSHOT/clm" python "$TINY_CLM_SNAPSHOT/compression/freeze_selection.py" \
  --manifest "$TINY_CLM_CAMPAIGN/manifest.json" \
  --run-ids original-keep0.8-seed0,dense_matched-keep0.8-seed0,constrained-keep0.8-seed0 \
  --rationale 'Replace with the actual validation selection rationale' \
  --test-seeds 3101,3102,3103 --output "$TINY_CLM_CAMPAIGN/selection.json"
PYTHONPATH="$TINY_CLM_SNAPSHOT/clm" python "$TINY_CLM_SNAPSHOT/compression/submit_final.py" \
  --manifest "$TINY_CLM_CAMPAIGN/manifest.json" --selection "$TINY_CLM_CAMPAIGN/selection.json"
PYTHONPATH="$TINY_CLM_SNAPSHOT/clm" python "$TINY_CLM_SNAPSHOT/compression/submit_final.py" \
  --manifest "$TINY_CLM_CAMPAIGN/manifest.json" --selection "$TINY_CLM_CAMPAIGN/selection.json" --submit
```

Final preparation precedes the selected-artifact evaluation array; aggregation waits for
all rows, including failures. Results appear in `final-test-<selection-hash>/results.json`.
Test evaluation does not retrain or select models. The final helper refuses duplicate
campaign submissions; if an infrastructure retry is necessary, use the printed exact
single-stage sbatch command with the failed array indices after old jobs finish, preserving
failed outputs as above. Never turn repeated test attempts into model selection.

## 7. Safe updates

Every submitted campaign runs `git archive` at a fixed commit in a separate read-only-file
snapshot. It does not execute from the mutable checkout. Source commits, configs,
environment freeze and asset hashes are recorded; jobs verify environment and assets.
Do not edit snapshots, environment or shared assets underneath running jobs.

To obtain updates without altering the active version:

```bash
cd "$TINY_CLM_ROOT"
git status --short
git fetch origin tiny-clm-constrained
git log --oneline HEAD..origin/tiny-clm-constrained
TINY_CLM_NEXT=$(git rev-parse origin/tiny-clm-constrained)
git worktree add --detach "$TINY_CLM_PARENT/context-language-models-${TINY_CLM_NEXT:0:12}" "$TINY_CLM_NEXT"
```

Use that new worktree and a new frozen environment for subsequent campaigns when
dependencies change. Existing campaigns resume against their old manifest/snapshot.
No `git pull`, editable install update, or checkpoint/cache overwrite occurs in jobs.
