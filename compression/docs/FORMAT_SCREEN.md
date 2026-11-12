# One-factor answer-format experiment

## Diagnosis motivating this experiment

The user ran the read-only KV audit on the original-prompt 4B screen. All 24
GETs had observable responses, 19 contained the exact expected value as a line
in a successful tool result, and none contained a correct answer block. There
were no conflicting or unpaired tool-call records. This **19/24 diagnostic is
not the official task reward**, which remains zero.

The five other commands queried keys 105, 119, 108, 110 and 125 in batch file
`kv-0000.txt`, although the preceding offload put keys 100–129 in `kv-0001.txt`.
Their saved results contained only the readiness marker and an exit code of
zero. `grep ... | cut ... && echo READY_FOR_NEXT_OP` can hide the no-match
status: without pipefail, the pipeline takes the successful `cut` status and
advances. A local synthetic shell test reproduced this mechanism. That does
not prove the second offload file was intact, but the queries never searched it.

There are therefore at least two observed CLM errors: omitted answer framing
and wrong-file lookups followed by premature advancement. Append-only has a
separate failure: one truncated tool-call attempt followed by plain command
text, zero executed calls, and no delivered GETs. No compression training has
been evaluated by any of these observations.

## Precisely one intervention

`kv-answer-format-v1` appends the constant `FORMAT_TEXT` from
`clm/tiny_clm/answer_format.py` once to the initial task instruction in **both**
modes. It repeats the required output delimiters and gives a printing statement
with placeholder shell variables. It supplies no queried keys, values, lookup
implementation, file-selection rule, retries or automatic answer wrapper.

The original task already specified ANSWER blocks. This experiment tests whether
a more explicit reminder changes compliance; it does not repair a missing task
specification. The added prompt tokens count against the existing context budget.
As with any prompt change, subsequent model-generated retrieval decisions may
also change. A result is not proof that only formatting behavior was affected.

The adapters inherit the existing CLM and append-only agents and delegate the
modified instruction to their original `run` methods. The default agent paths
remain unchanged. Each adapted trial records original/modified instruction
hashes and the reminder's version/hash. Grading, commands, readiness handling,
context-management code, task files, model weights, generation settings,
1,024-token output cap, 128-step cap and provisional reward floor are unchanged.
In particular, the experiment does not fix the wrong-file lookups, enable
pipefail, repair truncated tool calls or change the upstream reminder loop.

The old completed screen supplies the historical control. The new campaign
runs the same single development task once per mode with the reminder. It
reuses the verified uncompressed export; it performs no download, export,
training, decomposition or compression. All relevant source evidence and
export metadata are hashed; weights are verified within the compute stages.
The source campaign and its scores are never edited.

Results carry `diagnostic_protocol` in the manifest, episode records and summary.
Missing or inconsistent application records fail the experiment. Ordinary
screen validation and baseline/main aggregation reject these changed-prompt
records. The report uses `gate_type=protocol_screen` and always sets
`can_advance_to_compression=false`. A positive result would justify an independent
full baseline and memory audit under a newly declared protocol, not reinterpret
the original screen or establish compression quality. If used later, the same
protocol must apply to dense and compressed backbones and be included in their
comparison identities. That later full-grid integration is not provided by this
diagnostic.

## Exact Unity execution

Fetch and check out the delivery commit first. The completed source screen must
still be available. No environment reinstall or task-container rebuild is needed.

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
export TINY_CLM_FORMAT_CAMPAIGN="$TINY_CLM_RUNS/format-qwen3-4b-001"

# Dry-run: prints all requests and dependencies; creates no jobs or campaign.
python compression/protocol_screen.py \
  --source-manifest "$TINY_CLM_RUNS/screen-qwen3-4b-004/manifest.json" \
  --campaign "$TINY_CLM_FORMAT_CAMPAIGN"

# Submit CPU checks -> GPU episodes -> CPU report.
python compression/protocol_screen.py \
  --source-manifest "$TINY_CLM_RUNS/screen-qwen3-4b-004/manifest.json" \
  --campaign "$TINY_CLM_FORMAT_CAMPAIGN" --submit
```

The partition/account/constraint/QoS above were verified in this user's earlier
Unity diagnostics. Other users must supply their own settings. Defaults:

| Stage | GPUs | CPUs | Host RAM | Wall time | Dependency |
| --- | ---: | ---: | --- | --- | --- |
| protocol_check | 0 | 2 | 8G | 30 minutes | none |
| protocol_contextbench | 1 | 4 | 64G | 2 hours | afterok checks |
| protocol_report | 0 | 2 | 8G | 15 minutes | afterany episodes |

Override stage resources through `TINY_CLM_PROTOCOL_CHECK_CPUS`,
`TINY_CLM_PROTOCOL_CONTEXTBENCH_MEM`, `TINY_CLM_PROTOCOL_CONTEXTBENCH_TIME`, etc.
Only one GPU is supported. The existing launcher clears inherited scheduler
options, uses explicit CPU/GPU steps, loads Apptainer for the task job, aligns
CPU threads with the allocation, and executes an immutable commit snapshot.
The CPU stage runs correctness tests and imports the real installed adapters
before the GPU dependency is released. Failed prerequisites prevent inference;
the reporting job records incomplete evidence as failure. A valid zero score
is a completed negative experiment, not an infrastructure failure.

After reconnecting, these read-only commands need no environment activation:

```bash
export TINY_CLM_FORMAT_CAMPAIGN=/work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/format-qwen3-4b-001
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ["TINY_CLM_FORMAT_CAMPAIGN"])
ids = json.loads((root / "jobs.json").read_text())
subprocess.run(["sacct", "-X", "-j", ",".join(ids.values()),
    "--format=JobID%24,JobName%28,State%20,ExitCode,Elapsed"], check=True)
for path in sorted((root / "logs").glob("*.err")):
    if path.stat().st_size:
        print("\n", path.name)
        print("\n".join(path.read_text().splitlines()[-60:]))
report = root / "protocol-report.json"
print(report.read_text() if report.exists() else "Report not written yet.")
PY
```

A disconnected interactive session does not own the separately submitted jobs.
Do not resubmit solely because the session ended. If a job is interrupted,
preserve its partial campaign and use a new campaign name; this bounded
diagnostic does not promise mid-episode resumption. Keep the shared source
export unchanged while queued/running jobs depend on it.

## Validation boundary

Local tests cover prompt-only delegation, unchanged original agents, shell
pipeline semantics, protocol identities, missing application records, rejection
of mixed/changed records, immutable-source checks, isolated submission
environments, dry-run behavior and the three-job dependency chain. No Slurm
submission, real-model inference, or task-quality improvement was executed by
the assistant for this change. The actual installed Harbor/ContextBench
adapter imports and GPU task runs remain Unity checks.

The full local suite passed from a read-only archive with no Git metadata:
**153 passed, 1 SDK-dependent test skipped**. Shell syntax validation also passed.
The test command was `PYTHONPATH=clm python -m pytest -q compression/tests
clm/tests/test_context_string.py`, with bytecode writes disabled and pytest
cache/JUnit outputs directed outside the archive. No model improvement follows
from these software checks.
