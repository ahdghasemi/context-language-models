# Paired history-format episodes

This is an opt-in development diagnostic, not compression or a reproduction of
the paper's serving protocol. It requires a completed `history_audit` whose
saved A/B/B/A replay reproduces the original non-call and produces valid calls
with native history. The driver verifies that evidence, not just a success flag.

## Intervention and controls

Four fresh CLM episodes run sequentially on one GPU: two controls, then two native
history episodes. Each arm uses R1000/1101 and R130/1102, the same dense BF16
checkpoint, task/container hashes, initial answer/tool reminder, greedy decoding,
1024-token cap, 128-call limit, and original command execution and grader. A fresh
model server starts per episode with the original task-index seed. Append-only
is not rerun here; this does not establish the eventual CLM/non-CLM comparison.

`HistoryFormatMixin._query_with_retry` records hashes of previously observed
structured bash arguments. Immediately before inference, the native arm rewrites
only a terminal `bash {JSON}` line in retained assistant content matching one of
those hashes, using the replay's native envelope. The control delegates unchanged
messages through the same wrapper. Unverified, edited, malformed and nonterminal
text stays unchanged. No structured `tool_calls` are added; no offloaded files
are read; removed turns are not restored; commands are not changed. Only the
original parser and agent execute new model-generated calls. Original CLM source
and all existing default paths remain unchanged.

The editable mirror and its budget counter still use original history text.
Native wrappers add model-input tokens, recorded by the server. This is an
inference-side representation intervention, not a token-matched test or a new
budget policy. The external call hashes are diagnostic bookkeeping, not trainable
memory or a retrieval mechanism. A production adapter needs separate validation.

Well-formed calls can still be wrong. The official KV generator emits unquoted
`SET key = value` lines. Extraction patterns requiring quoted values will not
match those lines unless the agent changed the storage representation. No command
or answer repair is added to this experiment.

## Submit on Unity

Check out the implementation commit supplied with this change, with a clean
working tree. Keep existing campaigns and task locks unchanged. After reconnecting:

```bash
source /work/pi_aghasemi_umass_edu/tiny-clm.env
source "$TINY_CLM_VENV/bin/activate"
cd "$TINY_CLM_ROOT"
export PYTHONPATH="$TINY_CLM_ROOT/clm"
unset TRANSFORMERS_CACHE
export TINY_CLM_PARTITION=gpu-highvram
export TINY_CLM_CPU_PARTITION=cpu
export TINY_CLM_GPU_CONSTRAINT=a100-80g
export TINY_CLM_ACCOUNT=pi_aghasemi_umass_edu
export TINY_CLM_GPUS=1
export TINY_CLM_QOS=normal

python compression/history_screen.py \
  --replay-manifest "$TINY_CLM_RUNS/history-qwen3-4b-001/manifest.json" \
  --campaign "$TINY_CLM_RUNS/history-screen-qwen3-4b-001"

python compression/history_screen.py \
  --replay-manifest "$TINY_CLM_RUNS/history-qwen3-4b-001/manifest.json" \
  --campaign "$TINY_CLM_RUNS/history-screen-qwen3-4b-001" \
  --submit
```

The first invocation is a read-only dry run. Allocation defaults:

| Stage | CPUs | Host memory request | GPUs | Wall time |
|---|---:|---:|---:|---|
| history_check | 2 | 8G | 0 | 00:30:00 |
| history_contextbench | 4 | 64G | 1 | 03:00:00 |
| history_report | 2 | 8G | 0 | 00:15:00 |

These are requests, not measured runtimes. Override resources with
`TINY_CLM_HISTORY_CHECK_{CPUS,MEM,TIME}`,
`TINY_CLM_HISTORY_CONTEXTBENCH_{CPUS,MEM,TIME}`, or
`TINY_CLM_HISTORY_REPORT_{CPUS,MEM,TIME}`. Scheduler settings remain configurable.
Check current QoS submission limits before launching the three-job chain:

```bash
sacctmgr -n -P show qos where name=short,normal,long \
  format=Name,Priority,MaxWall,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU
squeue --me -o "%.18i %.28j %.10T %.10q %.6D %.35R"
```

`MaxSubmitJobsPU` counts pending and running jobs, including interactive sessions.
It is distinct from the running-job limit and is not shown by `MaxJobsPU` alone.
The example uses the user's verified `normal` QoS. Use `short` only after checking
that its submission allowance can accommodate the whole chain plus existing jobs;
higher priority does not guarantee immediate scheduling. If the first submission
is rejected, no jobs from this chain were accepted, but its campaign already exists.
Preserve that directory and use a new campaign name. For a later rejection, inspect
`jobs.json` and Slurm for accepted jobs before starting another campaign.

Checks include the complete correctness suite and actual installed adapter
imports. Successful checks release inference (`afterok`). The report runs after
inference terminates (`afterany`), failing for incomplete evidence. A graded zero
reward is a complete negative result. Source/configuration hashes, environment
fingerprint and immutable snapshot are checked. Preserve partial outputs and
use a new campaign to retry; do not modify the queued snapshot or environment.

## Results and interpretation

```bash
export TINY_CLM_HISTORY_SCREEN="$TINY_CLM_RUNS/history-screen-qwen3-4b-001"
cat "$TINY_CLM_HISTORY_SCREEN/jobs.json"
cat "$TINY_CLM_HISTORY_SCREEN/history-screen.json"
```

`control/` and `native_history/` retain request logs, original context snapshots,
official rewards and per-trial `history_format.jsonl` application logs. These
contain hashes rather than command copies, and the report reverifies them. The
report compares reward, answered queries, model/tool calls, tokens and latency.
Timing is descriptive: fixed arm order, one trajectory per task/arm, no speedup claim.

`passed` means native formatting was applied on both tasks and each reached the
existing provisional reward floor. It does not require superiority to controls;
inspect the paired results. All outcomes keep `can_advance_to_compression=false`.
Even a pass requires fresh validation seeds and a frozen protocol before the
compression comparison. Earlier task scores are never revised.

## Inspect a completed negative result without another GPU job

Use system Python after reconnecting; no environment activation or Slurm
submission is needed. Set `TINY_CLM_HISTORY_SCREEN` to the actual campaign path.

```bash
python3 compression/inspect_contextbench.py \
  --campaign "$TINY_CLM_HISTORY_SCREEN" \
  --arm native_history \
  --episode clm--kv-store-R1000-seed1101 \
  --episode clm--kv-store-R130-seed1102 \
  --samples 1 --get-samples 3 --chars 2400
```

Use `--arm control` for controls, or omit `--arm` for both. The reader checks the
report's manifest hash, arm/protocol identities, episode coverage, saved record
agreement, and application-log hashes. It does not require a passing quality gate
and writes nothing. First-GET excerpts expose retained commands and following
messages; context edits mean these are not a complete execution transcript or an
exact alignment with server requests. No commands in those messages are executed,
and no task gold or live sandbox is read. Recognized calls and answered queries
alone do not establish correct retrieval; inspect the official rewards separately.

Local tests cover replay-transform equivalence, unchanged controls, unknown prose,
argument/prefix preservation, delegation, immutable inputs, negative outcomes,
missing/foreign episodes, gate isolation, dry runs and dependencies. Full model
episodes require Unity execution; this document does not assert they have run.
