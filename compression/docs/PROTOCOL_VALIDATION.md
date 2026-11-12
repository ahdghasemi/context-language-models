# Fixed-protocol dense behavior validation and training-memory probe

## Observed result and its limits

The user-supplied Unity report for the `kv-answer-format-v1` development run
records 23/24 correct answers, 24 answered queries, and 50 executed tool calls
for Qwen3-4B-Instruct-2507 in CLM mode. The historical original-prompt run had
zero recognized answers and 26 calls. The reminder run used 625,998 prompt
tokens, 9,929 completion tokens, and 394.59 seconds for that episode. These
are observations from the user's report, not an assistant-run benchmark.

This is useful behavior on **one development task with a modified prompt**.
It does not revise the original zero score or establish a compression result.
The reminder may alter lookup decisions as well as formatting. Append-only
still made zero tool calls and reached no GET queries. Its zero reward is an
observed failure, not a clean measurement of information-retention capacity.
Do not claim general CLM superiority or a latency advantage from this contrast.

The next campaign reruns R130 and R1000, seeds 1101 and 1102, in both modes:
eight serial episodes with the same fixed reminder and all original generation,
context-management, container, task, and grader settings. It reuses the verified
dense export and existing task lock. There are no downloads or container builds.
R130/1101 is explicitly a **development repeat**; the other three tasks are
reported separately as additional validation. There are only two independent
task seeds, no final-test claim, and no token-level pseudo-replication.
The provisional reward floor remains 0.5 **for every CLM episode**, not an
average that could hide an entire failed task. Do not adjust it after results.

## Memory check before quality training

The current trainer keeps all model weights and gradients in FP32, uses BF16
autocast for the LM forward, and retains FP32 Adam states. It refreshes dense
hard-rank proxies using sequential exact reduced SVDs and adds the unnormalized
Frobenius penalty once per optimizer update. The following values are analytical,
not measurements (P=4,022,468,096 unique parameters):

| Selected weights | FP32 weights + gradients + Adam + proxies + saved penalty differences |
| --- | ---: |
| All MLP projections | 85,876,269,056 bytes (79.98 GiB) |
| All 36 MLP down projections | 71,531,749,376 bytes (66.62 GiB) |

Activations, temporary casts, SVD/Adam workspaces and allocator overhead are
excluded. The first subtotal alone exceeds the observed A100 device capacity
of 85,094,825,984 bytes. **Do not run the all-MLP 4B configuration on that GPU.**

The resource probe uses down projections at selected keep=0.6 (rank 1,216).
This policy is an experimental architecture choice, not inherited from GNNs.
Its analytical whole-model parameter reduction is about 8.92%, not 40%.
All layers remain trainable; this is not adapter training or a frozen backbone.
There is no actual compressed inference artifact from the probe.

The probe calls the actual trainer for two updates: batch=1, sequence length=512,
accumulation=4, FP32 master weights, BF16 autocast, gradient checkpointing, and
persistent Adam/StepLR. Synthetic random tokens deliberately isolate resource
behavior from task-quality claims. It exercises a second projection with Adam
state resident, full gradients, validation, best-weight CPU storage, and atomic
checkpoint writing. The source export stays unchanged; temporary training
checkpoints are removed. The two-update stop produces an intentionally incomplete
training checkpoint, which is never used as a trained model.

`training.adam_foreach=false` selects PyTorch's single-tensor Adam execution,
avoiding a multi-tensor optimizer workspace. The equations, betas, learning rate,
moment persistence, proxy update and penalty schedule are unchanged. Existing
configs default to `null`, preserving PyTorch's previous automatic selection.
The setting is saved in checkpoints and included in configuration compatibility.
CPU tests compare single-tensor and foreach updates within stated tolerances and
verify exact resume within each setting. CUDA numerical equivalence between the
two kernels is not claimed. Future matched controls must use the same setting.

The probe requests 128G host RAM and requires 160 GiB free under configured
`TMPDIR` for atomic checkpoints. It refuses a GPU with less than the analytical
down-only subtotal plus a provisional 8 GiB workspace allowance free before load.
A measured pass additionally requires at least 4 GiB of headroom relative to peak
CUDA reserved memory. These margins are provisional resource criteria, not a
guarantee for longer sequences, different ranks or arbitrarily long training.
Peak allocated/reserved memory, host peak RSS, GPU identity, time, checkpoint
size, training configuration and the source weight hash are recorded.

If the probe fails, retain its report. Do not silently reduce the workload or
ignore an OOM. Training on real tokens, longer continuation, stage transitions,
and resume on this 4B model remain later validation steps. The existing small
correctness tests cover resumption, but are not large-model memory evidence.

## Unity submission

Fetch the delivered commit and check it out detached before running these commands.
The source is the **passing format-development** campaign, not the zero-score
original screen. No environment reinstall is needed.

```bash
source /work/pi_aghasemi_umass_edu/tiny-clm.env
source "$TINY_CLM_VENV/bin/activate"
cd "$TINY_CLM_ROOT"
export PYTHONPATH="$TINY_CLM_ROOT/clm"
unset TRANSFORMERS_CACHE
export TINY_CLM_PARTITION=gpu-highvram
export TINY_CLM_CPU_PARTITION=cpu
export TINY_CLM_GPU_CONSTRAINT=a100-80g
export TINY_CLM_GPUS=1
export TINY_CLM_ACCOUNT=pi_aghasemi_umass_edu
export TINY_CLM_QOS=long
export TINY_CLM_VALIDATION_CAMPAIGN="$TINY_CLM_RUNS/validate-qwen3-4b-format-001"

# Dry-run only; no files or jobs are created.
python compression/validate_protocol.py \
  --source-manifest "$TINY_CLM_RUNS/format-qwen3-4b-001/manifest.json" \
  --campaign "$TINY_CLM_VALIDATION_CAMPAIGN"

# Submit after inspecting the printed plan.
python compression/validate_protocol.py \
  --source-manifest "$TINY_CLM_RUNS/format-qwen3-4b-001/manifest.json" \
  --campaign "$TINY_CLM_VALIDATION_CAMPAIGN" --submit
```

These scheduler settings were established in the user's supplied Unity output;
other users must select their own account, partition and GPU constraints.

| Stage | GPUs | CPUs | Host RAM | Time | Dependency |
| --- | ---: | ---: | --- | --- | --- |
| validation_check | 0 | 2 | 8G | 30 min | none |
| validation_contextbench | 1 | 4 | 64G | 6 h | successful checks |
| validation_memory | 1 | 4 | 128G | 2 h | successful checks; episodes finished, even if failed |
| validation_report | 0 | 2 | 8G | 15 min | both GPU jobs finished, even if failed |

Only one GPU job from this campaign can execute at once. Override resources
with `TINY_CLM_VALIDATION_MEMORY_MEM`, `TINY_CLM_VALIDATION_MEMORY_TIME`,
`TINY_CLM_VALIDATION_CONTEXTBENCH_TIME`, and corresponding `_CPUS` variables.
Submission strips inherited scheduler options, freezes the Git snapshot and
source evidence, and checks the installed environment. No training sweep is
submitted. Original campaigns, exports, task locks and environment remain intact.

After reconnecting, inspect without environment activation:

```bash
export TINY_CLM_VALIDATION_CAMPAIGN=/work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/validate-qwen3-4b-format-001
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ["TINY_CLM_VALIDATION_CAMPAIGN"])
jobs = json.loads((root/"jobs.json").read_text())
subprocess.run(["sacct", "-X", "-j", ",".join(jobs.values()),
    "--format=JobID%24,JobName%28,State%20,ExitCode,Elapsed"], check=True)
for p in sorted((root/"logs").glob("*.err")):
    if p.stat().st_size:
        print(p.name, "\n", "\n".join(p.read_text().splitlines()[-60:]))
for name in ("memory-probe.json", "validation-report.json"):
    p = root/name
    print("\n"+name+"\n"+(p.read_text() if p.exists() else "Not written yet."))
PY
```

Disconnecting the interactive session does not cancel these separately submitted
jobs. Do not resubmit for that reason. Interrupted probes/episodes are not resumed;
their partial reports remain visible and a retry requires a new campaign name.
No checkpoint from the resource probe is retained or eligible for recovery training.

## Decision boundary

The report distinguishes infrastructure completeness, dense behavior, and memory
feasibility. Missing/incompatible evidence cannot pass; low reward and inadequate
memory remain visible. The report retains `can_advance_to_compression=false` and
cannot satisfy the existing baseline/main gates. General-language validation and
inference profiling for this 4B model, a matched training-data design and an
explicitly protocol-matched compression pilot remain necessary. This step is a
broader behavior baseline plus a resource measurement, not a finished scientific
baseline or successful compression experiment.

Implementation tests run locally on CPU. Eight-episode evaluation and the actual
4B CUDA memory probe have not been executed by the assistant. Read the Unity
reports before drawing any conclusion about them.

The local full suite passed **170 tests, one SDK-dependent test skipped**. Tests
cover immutable inputs and scheduling, missing/mixed evidence, measured headroom,
development/validation separation, both Adam execution options, and resume.
