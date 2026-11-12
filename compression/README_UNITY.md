# Unity / Slurm operating instructions

After the passed client-transport probe, the next bounded diagnostic is the
[inference-only 4B backbone screen](docs/BACKBONE_SCREEN.md). It reuses the existing
environment/container, runs two episodes, and cannot authorize compression.

The user-submitted `baseline-003` completed all nine Slurm stages, including real-model
evaluation and GPU profiling. **Its capability gate failed: all four CLM and four
append-only episodes scored zero. Do not advance to compression on this baseline.**
Constrained real-model compression remains untested. Earlier campaigns failed at job
launch and dependency fingerprinting; those issues cleared in baseline-003.
`gpu`, `gpu-preempt`, `cpu` partition terminology and
`module load apptainer/latest` were checked against official Unity documentation on
2026-11-11. User output subsequently verified account `pi_aghasemi_umass_edu`, allowed
QoS `long,normal,short`, `a100-80g`, driver compatibility on an A100, and the task
container build/runtime. Current quotas and resource availability still depend on Unity.
No account, QoS or GPU type is invented or silently selected.

Update from user-supplied Unity terminal output: the environment check, all 43 tests,
and the tiny CUDA smoke passed inside an existing A100-80GB allocation. The first
container builds failed on inherited /nas and /work binds. Clearing the inherited
bind/mount variables passed its two new tests on Unity and allowed `%post` to start.
That exposed an independent host fakeroot GLIBC incompatibility. The task base is now
digest-pinned to Python 3.13.15 on Debian trixie; see the diagnosis below.
The rebuilt container and all four task packages passed on Unity. Model/data hashes
also passed verification. The first baseline environment job passed, but CPU steps
failed with an invalid GRES request; see the launch fix below. The later baseline-003
finished but failed the quality criterion, as detailed in
[Unity validation](docs/UNITY_VALIDATION.md#baseline-003-completed-workflow-zero-task-reward).

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

## 4. Establish the dense baseline, then run the pilot

The baseline stage runs the original dense checkpoint through export, general validation,
profiling and both ContextBench modes with zero optimizer updates. If it cannot reach
reward .50 on each pilot CLM task, compression quality is not yet interpretable. The
pilot submission requires its passing gate:

```bash
python compression/submit.py baseline --config compression/configs/pilot.json \
  --campaign "$TINY_CLM_RUNS/baseline-001"
python compression/submit.py baseline --config compression/configs/pilot.json \
  --campaign "$TINY_CLM_RUNS/baseline-001" --submit
# Wait for completion; inspect logs and this gate before proceeding.
cat "$TINY_CLM_RUNS/baseline-001/summary/gate.json"
```

If this gate fails, inspect the dense trajectories and compatibility errors first.
Do not lower the floor merely to let a sweep run. A new backbone/protocol requires new
preparation and baseline evidence.

Pilot, logs and recovery:

```bash
python compression/submit.py pilot --config compression/configs/pilot.json \
  --baseline-gate "$TINY_CLM_RUNS/baseline-001/summary/gate.json" \
  --campaign "$TINY_CLM_RUNS/pilot-001"
python compression/submit.py pilot --config compression/configs/pilot.json \
  --baseline-gate "$TINY_CLM_RUNS/baseline-001/summary/gate.json" \
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

The resume driver retains the old snapshot, rejects active original and previous retry jobs and requires
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
snapshot. Its file hash manifest is checked before reuse and before each job executes;
row configs are compared against the campaign manifest. This detects accidental edits,
not a security boundary against a user deliberately replacing all manifests. It does not execute from the mutable checkout. Source commits, configs,
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


## Additional fixed settings and tunable resources

Every stage supports `TINY_CLM_<STAGE>_CPUS`, `_MEM`, and `_TIME` overrides; for example
`TINY_CLM_SMOKE_MEM=12G` or `TINY_CLM_EXPORT_TIME=02:00:00`. Stage names are ENVIRONMENT,
CORRECTNESS, SMOKE, PREPARE, TRAIN, EXPORT, EVALUATE, PROFILE, CONTEXTBENCH and AGGREGATE.
The environment check defaults to 15 minutes; small CPU checks to 30 minutes. Resource
flags appear exactly once in dry runs. The implemented process mapping is always one
task on one GPU; changing GPU count above one is rejected, not silently ignored.

The ContextBench preparation also caches and hashes o200k_base. It is passed to both
agents through each job's environment. `calibrate_budget=false` pins the common benchmark
token ruler; actual Qwen token usage remains in episode logs. Re-run preparation in a new
destination if migrating a lock created by the earlier branch revision: old locks without
this cache are intentionally rejected.

Initial and periodic training checkpoints are atomic. In `gnn_persistent` mode Adam
moments survive intermediate stage boundaries; only completed runs discard them.
GPU numerical reproducibility still depends on the hardware and library kernels.

Local inference disables proxy routing (`NO_PROXY=*`) and sets
`LITELLM_LOCAL_MODEL_COST_MAP=True` in the episode subprocess environment. This prevents
an unrelated pricing-map download at import time on compute nodes. It does not supply
credentials or contact an external model API. The readiness check imports both original
agent classes as well as checking their budget counters.

## Container build failure: inherited Unity filesystem mounts

The observed Unity module exports `APPTAINER_BINDPATH` with eleven host filesystem
paths. Apptainer 1.5.5 forwards BINDPATH/MOUNT variables into its nested writable `%post`
execution, where missing bind destinations cause failure. The first `/nas`-only recipe
fix merely exposed the next missing directory, `/work`, and has been removed.

`tiny_clm.container.container_environment` now clears `APPTAINER_BIND`,
`APPTAINER_BINDPATH`, `APPTAINER_MOUNT` and their `SINGULARITY_` counterparts in a copy
of the child environment for image building, the runtime preflight and Harbor episodes.
Explicit command-line mounts, containment flags, cache paths and temporary paths are
preserved. The calling shell and system configuration are untouched. In the observed
system configuration, `mount hostfs=no` and the only configured bind paths are the
standard `/etc/localtime` and `/etc/hosts` files. A different system configuration may
require further investigation; this is not a general bypass of all site mounts.

For the reported failure, preserve the failed preparation log and rename the incomplete
task destination to a unique `.failed-bind-<timestamp>` sibling before retrying. Keep the
model/data assets and shared OCI cache: they are independent of this failed task build.
Do not retry concurrently with an active preparation process. Use the committed updated
recipe and a new log filename. The task lock records the new recipe and image hashes.

This fix changes Python source but not dependencies. A non-editable installed package
still contains its old source: direct diagnostic commands must select the updated
checkout explicitly, without modifying the frozen environment:

```bash
export PYTHONPATH="$TINY_CLM_ROOT/clm${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import tiny_clm.contextbench as c; print(c.__file__)'
python -m pytest -q compression/tests/test_container_env.py
```

The printed module must come from the updated repository's `clm/tiny_clm` directory.
New submitted jobs already put their immutable source snapshot first on PYTHONPATH.
Before updating the checkout, confirm that tracked changes are absent and no active
process is using it directly. Existing campaigns retain their original snapshots.

The xattr `ENOTSUP` warnings were not the fatal error in the supplied logs.

## Container build failure: host fakeroot requires GLIBC_2.38

After the mount fix, the next Unity attempt reached `%post` but failed while loading
`/.singularity.d/libs/libfakeroot.so`: `GLIBC_2.38 not found`. With no `/etc/subuid`
entry, Apptainer uses a root-mapped namespace and the host fakeroot command/library.
The old pinned base is Python 3.13.15 on Debian bookworm. The replacement is the same
Python version on Debian trixie, pinned by OCI digest, whose inspected amd64 base layer
contains libc6 2.41-12+deb13u4. See `docs/UNITY_VALIDATION.md` for exact provenance.

This addresses the observed library requirement for both building and Harbor task
execution, which also uses `--fakeroot`. The build recipe's `%test` now fails immediately
on any failed check, and the existing runtime preflight reports Python and libc versions
under Harbor's isolation flags. Do not bypass this preflight. No host libraries, system
configuration, installed training dependencies, or containment flags need changing.

We do not add `--ignore-fakeroot-command`: it removes part of the privilege emulation,
which can break Debian package installation, and changing only the build command would
leave runtime fakeroot unaddressed. Apptainer documents this incompatibility and the
limitations of that flag at https://apptainer.org/docs/user/1.5/fakeroot.html.

Preserve the failed destination as a unique `.failed-glibc-<timestamp>` sibling, then
prepare again from the new committed recipe inside a live compute allocation. Do not
move a destination with a completed lock or an active preparation process. The OCI cache
can be reused. Check `squeue -j "$SLURM_JOB_ID" -o "%.18i %.10T %.12L"` first; use a new
allocation if the current session is about to expire. A new base must be downloaded and
built; completion time has not yet been measured on Unity.

This is an explicit OS-level task-environment change from upstream ContextBench. Use
the same resulting image and lock for all dense/compressed and CLM/append-only controls.
Do not combine results produced using different image hashes. The task definitions,
graders and Python version remain fixed, but equivalent task behavior still requires
the baseline episode checks. A successful image build alone does not establish it.

## CPU jobs fail before Python: invalid GRES specification

In `baseline-001`, jobs 1467313 (correctness) and 1467320 (aggregate) were allocated
two CPUs and 8 GB on uri-cpu074 but `srun` failed to create their steps with
`Invalid generic resource (gres) specification`. The GPU environment job 1467312
passed. No test failure, model execution or compression result follows from this
launcher error. The exact inherited variable was not captured in those failed jobs;
GPU/GRES option inheritance from the submitting GPU VS Code session is the working
diagnosis, supported by Slurm's documented environment-to-option mapping.

All three submission drivers now pass a filtered child environment to `sbatch`:
parent allocation variables (`SLURM_*`, `SLURMD_*`), scheduler option overrides
(`SBATCH_*`, `SRUN_*`, `SALLOC_*`) and parent GPU visibility masks are excluded.
`SLURM_CONF` and `SLURM_CONF_SERVER` are retained for client connectivity. Project,
module, Python, cache and temporary paths remain. The calling terminal is untouched.
Resource requests must be set using `TINY_CLM_*` configuration; arbitrary inherited
SBATCH/SRUN settings are intentionally not an alternative configuration interface.
`--export=ALL` exports this filtered environment, and Slurm supplies the new allocation.
The campaign manifest records removed variable names, never their values.

Inside the new allocation, `tiny_clm.slurm` retains its actual job/array IDs and device
visibility while removing GPU/GRES/TRES option overrides. It then replaces itself with
`srun`: CPU stages explicitly use `--gres=none`; GPU stages use `--gpus=1`. Both use
one task and the allocated CPU count. This also applies to final-test job steps;
resume and final-test submission drivers use the same submission filter. Signal
forwarding still goes through srun. Do not use the submission filter inside an already
running allocation, where clearing its job identity would be incorrect.

Keep `baseline-001`, its logs and its immutable snapshot as failure evidence. The user
submitted `baseline-002` with the launcher repair: its CPU/GPU steps reached Python,
then failed the environment check described below. No training checkpoint exists to
resume in either failed campaign. Do not edit old snapshots, requeue their failed jobs,
rebuild the working container, or redownload verified assets for these launcher repairs.
The baseline resources were `long`, one A100-80GB, four CPUs, 32 GB, one-hour short GPU
stages and a six-hour ContextBench stage. Each new campaign repeats correctness before
releasing model work.

Primary references:
https://slurm.schedmd.com/srun.html (GPU/GRES environment options and `--gres=none`),
https://slurm.schedmd.com/sbatch.html (export and input/output environment).

## Environment hash mismatch after the Slurm launch repair

`baseline-002` reached Python on both CPU and GPU nodes, then failed the raw `pip
freeze` hash check. This does not by itself establish that dependencies were changed.
The former batch launcher appended the submitting terminal's PYTHONPATH, which can
contain the mutable checkout's `clm/` directory. A non-editable setuptools build leaves
ignored `*.egg-info` there. This metadata can shadow the installed distribution's
`dist-info`, changing a direct-URL requirement to `clm==0.1.0` in pip's output. That
mechanism was reproduced locally; the actual Unity package difference has not yet
been captured, so it remains a diagnosis to verify rather than an observed fact.

The installed-dependency check now runs the venv's Python with `-I -m pip freeze`.
Isolated mode excludes PYTHONPATH, the current directory and user site packages from
this check. Batch jobs also use only their immutable snapshot on PYTHONPATH; the
ContextBench runner still adds its separately pinned source explicitly where needed.
Setup records new environments with the same isolated command. Existing environments
are **not** re-frozen: their saved freeze.txt hash must still match READY.json, and the
observed isolated freeze must match that exact hash. Campaign execution additionally
requires READY.json to equal the original manifest's copy.

Submission, resume and final-test drivers check this before submitting any jobs.
Each running stage checks again. A mismatch prints a unified diff with remote URLs
redacted, then stops; no lock is rewritten and no package is excluded from the list.
This remains an installed-requirements check, not a hash of every dependency's bytes.

From the updated checkout, a read-only preflight is:

```bash
PYTHONPATH="$TINY_CLM_ROOT/clm" python -m tiny_clm.environment --venv "$TINY_CLM_VENV"
```

For diagnosis against a previous campaign, call `verify_environment(spec["venv"],
spec["environment"])` using its manifest. Do not rerun setup, install/upgrade packages,
delete egg-info or edit READY.json/freeze.txt merely to suppress a mismatch. Preserve
both failed campaigns. The user subsequently verified the original hash and completed
`baseline-003` using the existing assets and container. Its quality gate failed; the next
step is inspecting its saved replies, not submitting a compression pilot.

Primary references: https://docs.python.org/3.12/using/cmdline.html#cmdoption-I and
https://pip.pypa.io/en/stable/cli/pip_freeze/.

## Inspect a completed but unsuccessful baseline

The inspection script uses only the Python standard library, reads saved outputs, and
does not import model libraries, submit jobs, change thresholds, or modify the campaign.
Use the current checkout for diagnostics while preserving the original run snapshot:

```bash
python3 compression/inspect_contextbench.py \
  --campaign "$TINY_CLM_RUNS/baseline-003" \
  --episode clm--kv-store-R1000-seed1101 \
  --episode clm--kv-store-R130-seed1101 \
  --episode append_only--kv-store-R130-seed1102
```

It prints episode rewards, the recorded termination reason, command/context-edit
counters, server finish reasons, and bounded early/late assistant replies plus the
following tool result or reminder. Both upstream layouts are supported: CLM has a
final `trajectory.json`; append-only uses `context_snapshots/turn-*.json`. Early
snapshots are used because context editing can remove earlier messages from the final
transcript. Excerpts are explicitly truncated and are not raw generated token IDs.
Missing diagnostic files are identified; an entirely absent transcript fails clearly.

If replies appear as shell code in ordinary assistant text instead of structured tool
calls, verify the client boundary before changing the model or parser:

```bash
PYTHONPATH="$TINY_CLM_ROOT/clm" "$TINY_CLM_VENV/bin/python" \
  compression/check_client_protocol.py
```

This invokes the original `ClmAgent._query_with_retry` through the installed LiteLLM
client against a temporary localhost HTTP fixture. It checks that the bash schema,
messages, thinking flag and decoding parameters arrive intact, then that a synthetic
structured bash reply survives the SDK and original action extractor. It executes no
shell action, loads no weights, uses no real API credentials, and submits no jobs.
Passing verifies this transport path, not the model's ability to generate tool calls.
The same check is included in the correctness suite when harness dependencies exist.
