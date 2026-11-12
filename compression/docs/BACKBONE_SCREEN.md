# Inference-only backbone screen after baseline-003

The 0.6B baseline completed but scored zero in both modes. Saved replies show
plain-text commands instead of tool calls in CLM, and invalid shell commands in
append-only. The user's subsequent Unity client probe at commit
`78e579dd0091837f53bbf5d92751596e8cb955dc` passed with LiteLLM 1.109.0, OpenAI 3.49.0
and Harbor 0.16.1: tool schema, decoding settings, structured response and the
original action extractor survived the round trip. That synthetic probe excludes
one transport failure; it does not certify model generation or task execution.

## Candidate and scope

Screen `Qwen/Qwen3-4B-Instruct-2507` at immutable revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. This is a general post-trained,
non-thinking instruction model, **not a CLM-RL checkpoint**. The publisher reports
improved instruction following and tool use. This motivates a test, not a prediction
that CLM will work. Both model size and post-training change relative to Qwen3-0.6B;
this is not an isolated model-scaling experiment or reproduction of the paper's RL model.

Primary sources inspected:

- [Official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
- [Pinned configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json).
- [Pinned weight index](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/model.safetensors.index.json).
- [Pinned tokenizer template](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/tokenizer_config.json).
- [Installed-version Qwen3 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py).

The same audited Qwen3 architecture is supported by the installed Transformers
4.57.1. Its native tool tags remain compatible with the reference server. The actual
weights and GPU export/cache checks have now been exercised by the user. The
screen at `3d3a7b8dd6d2cd2868ebb814dfc096269e3088ab` passed the native-reference
export contract but scored zero in both modes. See
[the observed response failures and read-only audit](KV_RESPONSE_AUDIT.md).

The screen runs only `kv-store-R130-seed1101` under both original agents. It preserves
all baseline-003 context settings, prompts, skills, tokenizer counter, task bytes,
container, decoding (temperature 0, top_p 1, thinking disabled), step/token caps and
grader. Selecting a task does not alter its original index or server RNG seed. The
full four-task lock remains unchanged and is verified before use. No container build,
dataset preparation, training, rank projection, compression or profile sweep occurs.

This is a development screen on already inspected tasks, not a held-out quality
estimate. `screen-report.json` uses `gate_type=backbone_screen`. Even if CLM reaches
the provisional 0.5 floor, it cannot satisfy the baseline/pilot gates. A pass leads
to a full independent baseline plus a training-memory audit. A valid zero score is
reported as complete, with passed=false. Missing/invalid episodes are failures and
cause the reporting job to exit nonzero. Scores are never replaced with fabricated
values, and failed records remain available.

## Resource estimates, not measurements

The pinned config has 36 layers, hidden width 2560, intermediate width 9728,
32 query heads, 8 KV heads, head width 128, vocab 151936, no attention biases,
per-head Q/K normalization, and tied input/output embeddings. Counting unique
parameters gives **4,022,468,096**, including normalization and the tied embedding
once. BF16 tensors would occupy 8,044,936,192 bytes (7.492 GiB). The published index
metadata instead declares 8,045,591,552 total bytes, a 655,360-byte discrepancy.
The metadata is recorded separately; we do not invent a cause or use it as the
parameter count. Runtime checks verify the loaded count and report actual serialized
bytes. The downloaded files and deployed weights receive hashes.

The inference request is one A100-80GB, 4 CPUs and 64G host RAM. A 36,864-token KV
cache alone is about 5.06 GiB in BF16; weights, logits, activations and other workspaces
are additional. This is an allocation estimate, not measured resident/peak memory.
The screen records peak allocated/reserved CUDA memory and host peak RSS for export.
These are loading/checking peaks, not a latency or serving-memory benchmark.
Allow at least 40 GiB free disk for roughly 8 GB of downloaded weights, another
8 GB export, and logs/workspace; filesystem free space does not verify project quota.
`HF_HOME` and campaign storage remain on the configured work filesystem.

A small inference footprint does **not** make full constrained training fit:

| FP32 training component | Analytical GiB |
| --- | ---: |
| Dense weights, gradients, two Adam moments (16P bytes) | 59.94 |
| Full all-MLP auxiliary matrices (4S bytes) | 10.02 |
| Saved penalty differences (4S bytes) | 10.02 |
| Subtotal before other workspaces/activations | 79.98 |

Default Adam foreach temporaries, decomposition workspaces, activations, logits,
allocator overhead and selection/checkpoint work are additional. The observed A100
has about 79.25 GiB usable total. **Do not launch the current all-MLP 4B training grid.**
Down-projection-only auxiliaries would reduce this subtotal to about 66.62 GiB,
but optimizer workspaces and activation peaks still require measurement and possibly
an explicit memory-saving optimizer implementation. That work is not implemented or
validated by this screen, and would be documented/tested separately.

## Submit from the existing Unity environment

Keep baseline-003 and all failed campaigns unchanged. First fetch and detach at the
exact delivery commit supplied with this update. No pip installation or lock refresh
is required. Use a new screen campaign name for each independent attempt.

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
export TINY_CLM_SCREEN_CAMPAIGN="$TINY_CLM_RUNS/screen-qwen3-4b-001"

# Read-only planner: no jobs or campaign directories created.
python compression/screen.py \
  --reference-manifest "$TINY_CLM_RUNS/baseline-003/manifest.json" \
  --campaign "$TINY_CLM_SCREEN_CAMPAIGN"

# Only this command submits jobs; no training or compression is possible here.
python compression/screen.py \
  --reference-manifest "$TINY_CLM_RUNS/baseline-003/manifest.json" \
  --campaign "$TINY_CLM_SCREEN_CAMPAIGN" --submit
```

Scheduler settings above come from the user's Unity output. Current availability and
quota are not guaranteed. The driver checks the original baseline environment before
submitting, strips the parent interactive allocation's Slurm resource variables, and
uses a hashed immutable source snapshot. Ending VS Code does not terminate these
independently submitted jobs. Do not update the environment while they are running.

| Job | CPU / host memory / wall time | Dependency |
| --- | --- | --- |
| screen_check | 2 / 8G / 30 min | none |
| screen_prepare | 4 / 32G / 2 h | checks succeed |
| screen_export | 4 / 64G / 1 h; one GPU | download succeeds |
| screen_contextbench | 4 / 64G / 2 h; one GPU | verified export succeeds |
| screen_report | 2 / 8G / 15 min | benchmark terminates, success or failure |

Checks include the full inexpensive test suite, synthetic client probe, CPU random
smoke and existing task/container/cache hashes. Preparation downloads this public
model only, with immutable revision and a file lock. No credentials are embedded;
if Hugging Face requires credentials for the user's connection, configure them using
the site's normal credential mechanism, never in Git or logs. An interrupted download
can reuse the Hugging Face cache; published model locks are never silently refreshed.
The export job loads BF16 directly, performs no FP32 training checkpoint/optimizer
allocation, verifies save/reload logits (rtol=atol=1e-4), and checks cached vs uncached
last-token logits (rtol=0.02, atol=0.25, allowing BF16 kernel differences). These small
checks do not assess task quality. The ContextBench job uses the same reference server. Its **loading/readiness**
allowance is 600 seconds instead of 180 to accommodate the larger checkpoint on
shared storage; the agent episode timeout remains 1800 seconds. This changes only
startup failure detection, not generation or task budgets.

Per-stage overrides are `TINY_CLM_SCREEN_CHECK_{CPUS,MEM,TIME}`, and equivalently
`SCREEN_PREPARE`, `SCREEN_EXPORT`, `SCREEN_CONTEXTBENCH`, `SCREEN_REPORT`.
`TINY_CLM_SCREEN_ASSETS` overrides the model-only asset directory. One GPU is enforced.
For download-only setup, add `--prepare-only` to the planner and submission commands
and use a separate campaign name. This submits only checks/downloads. A later full
screen with a new name reuses the locked model snapshot. Setup cannot submit a sweep.

## Inspect results and handle interruption

```bash
export TINY_CLM_SCREEN_CAMPAIGN=/work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/screen-qwen3-4b-001
IDS=$(python3 -c 'import json,os; from pathlib import Path; print(",".join(json.loads((Path(os.environ["TINY_CLM_SCREEN_CAMPAIGN"])/"jobs.json").read_text()).values()))')
sacct -X -j "$IDS" --format=JobID%24,JobName%28,State%20,ExitCode,Elapsed,NodeList%20
for f in "$TINY_CLM_SCREEN_CAMPAIGN"/logs/*.err; do
  if [ -s "$f" ]; then tail -n 80 -v "$f"; fi
done
if [ -f "$TINY_CLM_SCREEN_CAMPAIGN/screen-report.json" ]; then
  cat "$TINY_CLM_SCREEN_CAMPAIGN/screen-report.json"
fi
```

The uncompressed deployment is in `export/`; its loading checks/resource estimates
are in `export-audit.json`. Episodes, server request logs, generated tool calls,
commands, execution outcomes and grading remain in `contextbench/`. Use the existing
read-only inspector if reward remains low:

```bash
python compression/inspect_contextbench.py \
  --campaign "$TINY_CLM_SCREEN_CAMPAIGN" \
  --episode clm--kv-store-R130-seed1101 \
  --episode append_only--kv-store-R130-seed1101
```
 Full trajectories are
needed to distinguish valid-but-wrong calls from malformed/failed shell commands.
Do not increase caps or repair generated commands automatically to obtain a pass.

These bounded inference jobs have no optimizer to resume and do not request an
advance training-checkpoint signal. Wall-time termination leaves partial output for
diagnosis; no exact mid-generation resumption is promised. Use a **new** campaign for
a repeat after diagnosis; completed model downloads remain reusable. Existing export
or incomplete episode paths are not overwritten. Do not point the training resume
script at a screening campaign. If submission stops halfway, `jobs.json` records the
accepted jobs; inspect/cancel those job IDs before repeating in a new campaign.

## Validation boundary

Local tests exercise real Git snapshots and mocked scheduler process boundaries,
dependency ordering, read-only planning, environment mismatch rejection, task selection,
parameter arithmetic, mixed/partial result rejection, valid zero scores, and the rule
that a passing screen cannot authorize compression. CPU/GPU inference on the 4B model,
its download and these Slurm jobs have not been executed by the assistant. The user's
Unity output is required to establish those outcomes. Local tool/SDK dependencies are
insufficient for the full ML test suite; Unity runs it as the first prerequisite job.
