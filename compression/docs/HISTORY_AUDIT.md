# Isolating the history-format hypothesis

## What the user-supplied traces establish

The read-only inspection at `67feaa6db30c0052320402ec7745bdf64af27cdf`
examined the completed tool-guidance campaign at
`6b9e8b2b1e791bcba3003cdfb27f86718b31a66a`. The following observations come
from the user's pasted output; no direct Unity access is claimed.

- **CLM R1000/1101:** requests 1–10 contain parsed tool calls; usage records
  ten bash executions and ten real context syncs. Request 11, the first GET,
  is a non-tool reply of 226 tokens. All remaining replies are also non-tool:
  111 `stop` finishes, no `length` finish, and no output-cap hit. The retained
  first GET prints a `bash {"command": ...}` line rather than a native tool
  envelope. The episode ends at `budget_wall`, after repeated non-call replies
  grow the retained context. Raising the output cap does not address this
  observed failure. The supplied trace does not demonstrate whether that GET
  command would have retrieved the right value if executed.
- **Append-only R1000/1101:** the first reply attempts to spell out many supplied
  key/value pairs in a large command, reaches 1,024 tokens mid-call, and is not
  executed. The next 127 replies are six-token bare readiness commands. Usage
  confirms zero commands and termination at `max_steps`.
- **Append-only R130/1102:** the same truncation/recovery pattern occurs, but its
  proposed first command also creates random strings from `/dev/urandom` instead
  of storing the supplied values. The proposal was not executed. A larger output
  cap could address truncation without correcting this semantic error.

The append-only feedback claims a command ran normally despite zero executions;
this previously identified upstream feedback issue remains. None of these new
observations establishes successful compression, retrieval on R1000, or preserved
long-horizon capability.

## Why test historical representation

The original CLM deliberately permits arbitrary context-file edits. In
`clm/clm_harness/context_utils/context_string.py`, `rendered_text` serializes a
structured historical tool call as `name arguments`; `parse_back` rebuilds
assistant text and maps tool observations into user messages, without restoring
tool-call structure. `ContextEnv._apply_edit` in `context_env/env.py` applies that
conversion after an edit. It is documented original behavior, not an accidental
change introduced by compression.

The R1000 snapshots contain this plain historical `bash {...}` representation,
and the model later emits the same representation as its new action. This is a
plausible mechanism, not causal proof. The initial native-call reminder was still
protected. A prompt-history ablation is more informative here than another output-
cap increase or another full campaign with guessed prompt changes.

## Bounded A/B replay

`compression/history_audit.py` submits **one GPU job** with four greedy
completions of the same saved first-GET input, in A/B/B/A order:

- A: the tagged pre-call snapshot, unchanged.
- B: the same messages, with only terminal plain historical bash call lines
  expressed as native `<tool_call>` envelopes inside assistant **content text**.
  Each command must exactly match a structured call observed in an earlier
  tagged snapshot. An unknown or modified command fails validation. Protected
  instructions, command arguments, message roles, tool observations, the GET,
  weights, decoding and the 1,024-token cap are unchanged. Prompt token counts
  are reported because re-encoding changes representation length.

This is not parser relaxation, command recovery, forced decoding, or a new
agent. The replay never executes any historical or generated command and never
reads task gold. No live environment, container, dataset download or training is
needed. Original CLM serialization and all baseline/serving paths remain intact.

The request uses the existing reference Engine and the original bash schema.
It reconstructs the inference settings from the frozen campaign; it does not
assume byte-for-byte identity with the old SDK HTTP request. The report checks
the original prompt length against the historical server log and whether the
original non-call failure recurs. Repeat comparisons ignore generated call IDs.
These deterministic repeats test stability, not independent statistical trials.

`format_hypothesis_supported_on_this_prompt` requires matching original prompt
length, reproducible non-call output for A, stable repeats in each condition,
and valid single bash calls for both B completions. Even that outcome would only
support a representation effect on one saved prompt. It would not verify the
generated command, task reward, a whole-trajectory remedy, or append-only behavior.
A negative result remains a valid diagnostic. Compression gates remain closed.

The source report, configuration, export identity, prerequisite reports, request
log and selected/earlier snapshots are checked and hashed. Submission saves a
read-only input bundle and an immutable Git snapshot. Execution re-derives the
bundle and checks its identity, including the export weight checksum. A new
campaign is required after failure; partial completions are retained but not
treated as a complete result.

## Unity execution

Check out the delivery commit, then use the existing environment. These scheduler
values were verified by the user's Unity diagnostics. The default request is
one A100-80GB-compatible GPU, four CPUs, 64G host RAM and 30 minutes; the usual
`TINY_CLM_HISTORY_AUDIT_{CPUS,MEM,TIME}` overrides apply.

```bash
source /work/pi_aghasemi_umass_edu/tiny-clm.env
source "$TINY_CLM_VENV/bin/activate"
cd "$TINY_CLM_ROOT"
export PYTHONPATH="$TINY_CLM_ROOT/clm"
unset TRANSFORMERS_CACHE
export TINY_CLM_PARTITION=gpu-highvram
export TINY_CLM_GPU_CONSTRAINT=a100-80g
export TINY_CLM_GPUS=1
export TINY_CLM_ACCOUNT=pi_aghasemi_umass_edu
export TINY_CLM_QOS=long
export TINY_CLM_HISTORY_CAMPAIGN="$TINY_CLM_RUNS/history-qwen3-4b-001"

# Dry-run: no submission or campaign files.
python compression/history_audit.py \
  --source-manifest "$TINY_CLM_RUNS/tool-format-qwen3-4b-001/manifest.json" \
  --campaign "$TINY_CLM_HISTORY_CAMPAIGN"

python compression/history_audit.py \
  --source-manifest "$TINY_CLM_RUNS/tool-format-qwen3-4b-001/manifest.json" \
  --campaign "$TINY_CLM_HISTORY_CAMPAIGN" --submit
```

After reconnection, inspect the submitted job and concise results with system Python:

```bash
export TINY_CLM_HISTORY_CAMPAIGN=/work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/history-qwen3-4b-001
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ["TINY_CLM_HISTORY_CAMPAIGN"])
jobs = json.loads((root/"jobs.json").read_text())
subprocess.run(["sacct", "-X", "-j", ",".join(jobs.values()),
    "--format=JobID%24,JobName%28,State%20,ExitCode,Elapsed"], check=True)
for p in sorted((root/"logs").glob("*.err")):
    if p.stat().st_size:
        print(p.name, "\n", "\n".join(p.read_text().splitlines()[-60:]))
path = root/"history-audit.json"
if path.is_file():
    report = json.loads(path.read_text())
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}, indent=2))
    for item in report.get("observations", []):
        print(json.dumps(item, ensure_ascii=True))
else:
    print("Report not written yet; inspect job state before resubmitting.")
PY
```

The new GPU replay has not been executed by the assistant. The full local CPU
suite passed 204 tests, with one optional SDK-dependent test skipped. Tests cover
the upstream round trip, matched historical arguments, preservation of all other
message fields, refusal of unverified/later-only evidence, frozen inputs,
inference-only A/B controls, negative outcomes, and single-job Slurm submission.
No new quality or deployment result is claimed.
