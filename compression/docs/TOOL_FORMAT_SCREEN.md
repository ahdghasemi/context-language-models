# Native tool-call guidance after failed validation

## Observed Unity result

The user supplied the completed scheduler states and `protocol-report.json` for
`tool-format-qwen3-4b-001`, run at
`6b9e8b2b1e791bcba3003cdfb27f86718b31a66a`. All three stages exited successfully;
the report is complete, export verified, and has no infrastructure errors, but
`passed=false` and `can_advance_to_compression=false`. The report's manifest hash
is `dca40897bab983682603a2d75f79ed90906d181284de337c86f3f4391d0d2533`.
These are user-supplied observations, not jobs executed or artifacts accessed
directly by this assistant.

| Mode / task | Answer-only control correct / 24 | Added tool guidance correct / 24 | Model / parsed-tool calls | Operations delivered | Episode seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLM R1000/1101 | 0 | 0 | 121 / 10 | 11 / 34 | 854.009 |
| CLM R130/1102 | 0 | 23 | 55 / 55 | 26 / 26 | 270.550 |
| Append-only R1000/1101 | 0 | 0 | 128 / 0 | 1 / 34 | 198.814 |
| Append-only R130/1102 | 0 | 0 | 128 / 0 | 1 / 26 | 184.892 |

The R130 result supports a prompt-interface contribution on this development
case. It does not establish a reliable backbone across tasks. The R1000 agent now
produces parsed calls and receives more operations, but still answers none
correctly. Its completion tokens rose from 1,857 in the historical control to
26,913, with 2,510,769 prompt tokens in the new episode. More activity is not task
success or an efficiency improvement. The reported 10 parsed calls alone do not
certify their exit status; inspect saved tool results and usage.

The two CLM rewards average 23/48 = 0.479167. The complete failure on R1000 is
scientifically material even apart from the provisional 0.5 screen threshold.
Both append-only episodes still produced zero parsed calls and 1,786 completion
tokens. Matching previous totals suggests a recurring failure pattern, but does
not establish truncation in these new runs without inspecting their request logs.
R1000's termination code and growing context likewise do not locate the initial
failure; read the recorded termination reason and GET transition before assigning
a cause.

These are four development episodes with historical controls, not independent
replications of a general improvement. In particular, do not combine the earlier
R130/1101 result under the answer-only reminder with R130/1102 under the new
reminder and label both a common-protocol baseline. No compressed model was
trained or evaluated here. The original scores, provisional thresholds, and
compression gate remain unchanged.

### Next step: inspect saved responses, with no GPU submission

The read-only inspector now includes the last parsed call and the first transition
to a reply without a parsed call in its server-request samples. `--get-samples`
also shows the earliest distinct retained responses following the first observed
delivered GET header, even when that occurs after many ingestion calls. It reads
neither gold answers nor a live task environment. Context edits can alter retained
messages, so those excerpts are explicitly not aligned to server request numbers
and are not proof of command execution.

After checking out this diagnostic update, run with system Python:

```bash
python3 compression/inspect_contextbench.py \
  --campaign /work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/tool-format-qwen3-4b-001 \
  --episode clm--kv-store-R1000-seed1101 \
  --episode append_only--kv-store-R1000-seed1101 \
  --episode append_only--kv-store-R130-seed1102 \
  --samples 2 --get-samples 3 --chars 1600
```

This should distinguish an unexecuted first GET, a failing retrieval command,
truncation, and a later context-budget failure without another inference run.
Any subsequent intervention needs a separate declared comparison. A wider output
cap is justified as a control only where truncation is observed. A narrowed R130
study would require an explicitly revised scope and a new baseline under one
frozen protocol, rather than silently dropping failed R1000 configurations.

## What the saved traces establish

The user ran the read-only inspector at `4c15f23384cb107962fff94e87f01505127b0167`
on the completed four-task validation campaign. The supplied output distinguishes
two failure mechanisms:

| Inspected episode | Server finishes | Output-cap hits | Agent executions | Observed behavior |
| --- | --- | ---: | ---: | --- |
| CLM R1000/1101 | 128 stop | 0 | 0 | First reply describes an offload and prints Python as text; later replies repeat fenced readiness commands |
| CLM R1000/1102 | 128 stop | 0 | 0 | Same non-tool pattern; repeated claims of waiting for a new operation |
| CLM R130/1102 | 2 tool_calls, 126 stop | 0 | 2 | Two real offload calls, then a GET command printed as `bash {…}` text; repeated claims that the GET succeeded |
| Append-only R130/1101 | 1 length, 127 stop | 1 | 0 | First tool-call payload ends mid-string at 1,024 tokens; subsequent replies are bare readiness commands |

The three CLM failures did not reach the completion limit. Their first completions
were 422, 423 and 392 tokens, respectively. Increasing the limit alone does not
address the observed CLM mechanism. The append-only example has direct evidence
of truncation followed by non-tool output. The other append-only traces have not
been inspected here; do not generalize the exact sequence to all four solely
from matching aggregate counts.

This is a failure to use the structured calling interface reliably. For the
displayed bare/fenced command replies, the strict parser and harness are behaving
as specified by leaving the commands unexecuted. This does not prove every
possible native tool-call representation is supported. Earlier transport and
native/export checks passed, and the successful development episode made actual
tool calls; this evidence does not support a blanket claim that the SDK drops all
calls. The original model/system instructions already require bash calls, so the
new reminder is a prompt intervention, not a correction of a missing tool schema.

The append-only no-call path selects a readiness-marker explanation that says
the command ran normally even though `usage.json` reports zero executions. That
misleading feedback may reinforce the loop; its causal effect is unmeasured.
This experiment preserves it so feedback repair is not mixed with prompt guidance.

The single `n_answered` in CLM R130/1102 is not proof of a successfully executed
retrieval: the first GET command is ordinary text. The grader scans retained
message content, including assistant text, for answer blocks. Official reward
remains zero. Do not substitute recognized answer delimiters for verified task
success or tool execution.

Implementation evidence:

- `clm/clm_harness/clm_agent/prompts.yaml` requires a THOUGHT followed by bash.
- `_extract_tool_call` and the no-call branch in
  `clm/clm_harness/clm_agent/harness.py` consume structured calls and append a reminder.
- `clm/tiny_clm/serve.py:parse_response` recognizes complete native Qwen
  `<tool_call>` JSON, and leaves malformed or ordinary command text visible.
- The pinned [ContextBench agent](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/harbor/agents/base.py)
  supplies the bash schema, executes structured calls, and selects its no-call feedback.
- The pinned [grader reader](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/graders/base.py)
  flattens retained messages for scoring.

## One controlled addition

`kv-answer-and-tool-format-v1` retains the exact bytes of the existing
`kv-answer-format-v1` suffix, then appends one fixed tool-calling reminder.
It shows the native Qwen call envelope with a harmless `pwd` example, explains
that ordinary command text does not invoke the tool, and asks the model to wait
for its result. It explicitly accommodates the existing THOUGHT requirement.

The reminder contains no task keys, values, retrieval implementation, file-selection
rule, automatic repair or additional capabilities. Both modes receive the same
addition once to their initial instruction. The extra prompt tokens count toward
the same context budget. The model still chooses its commands and must generate
valid calls. Original agent loops, system instructions, feedback, parser, grader,
model weights, context management, 1,024-token output limit and all decoding
settings stay fixed. There is no forced-call decoding or text-to-shell execution.

The bounded test runs **R1000/1101 and R130/1102 in both modes**, four episodes.
These cover the two observed CLM failure patterns and use the existing validation
episodes as historical controls. They are now development cases selected after
inspecting failures, not untouched test data. A result cannot establish general
performance or preserve the earlier validation claim. Future final evaluation
requires new untouched seeds after the protocol is frozen.

This is an explicit compatibility experiment for the substituted backbone, not
part of the constrained compression algorithm or a novelty claim. No output-cap
change, model replacement, training or compression is bundled with it. A separate
cap-only control may be justified for truncation; it must retain all other settings
and its own identifier. A pass here only motivates a broader baseline under the
new declared protocol. It cannot satisfy the existing baseline or pilot gate.

## Unity commands

Check out the delivery commit first. Reuse the completed validation manifest,
verified export, task container, and installed environment. No setup, downloads,
container build, export, or memory probe is repeated.

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
export TINY_CLM_TOOL_CAMPAIGN="$TINY_CLM_RUNS/tool-format-qwen3-4b-001"

# Dry-run: creates no files and submits no jobs.
python compression/protocol_screen.py --intervention tool-call-format \
  --source-manifest "$TINY_CLM_RUNS/validate-qwen3-4b-format-001/manifest.json" \
  --campaign "$TINY_CLM_TOOL_CAMPAIGN"

# Submit CPU checks -> one GPU job with four serial episodes -> CPU report.
python compression/protocol_screen.py --intervention tool-call-format \
  --source-manifest "$TINY_CLM_RUNS/validate-qwen3-4b-format-001/manifest.json" \
  --campaign "$TINY_CLM_TOOL_CAMPAIGN" --submit
```

Scheduler values above come from the user's prior Unity diagnostics. The GPU job
requests one GPU, four CPUs, 64G host RAM and three hours. Checks request two CPUs,
8G and 30 minutes; reporting requests two CPUs, 8G and 15 minutes. Existing
`TINY_CLM_PROTOCOL_*_{CPUS,MEM,TIME}` overrides apply. Submission verifies the
unchanged environment, completed source reports and immutable source evidence;
execution uses a fixed Git snapshot. A valid negative source is allowed, but
missing/mutated/incompatible evidence fails. GPU evaluation depends on successful
checks; reporting runs after evaluation ends, including an evaluation failure.

The result is `protocol-report.json`, with `gate_type=tool_call_screen`, the new
protocol identity, control/variant rewards, and per-episode usage. Historical
control labels explicitly identify answer-format-v1, not the original prompt.
`can_advance_to_compression` remains false. Any incomplete coverage fails; zero
quality remains a completed negative result. Old scores and artifacts are intact.

After reconnecting, use system Python to inspect without loading the model:

```bash
export TINY_CLM_TOOL_CAMPAIGN=/work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/tool-format-qwen3-4b-001
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
root = Path(os.environ["TINY_CLM_TOOL_CAMPAIGN"])
jobs = json.loads((root/"jobs.json").read_text())
subprocess.run(["sacct", "-X", "-j", ",".join(jobs.values()),
    "--format=JobID%24,JobName%28,State%20,ExitCode,Elapsed"], check=True)
for p in sorted((root/"logs").glob("*.err")):
    if p.stat().st_size:
        print(p.name, "\n", "\n".join(p.read_text().splitlines()[-60:]))
p = root/"protocol-report.json"
print(p.read_text() if p.exists() else "Report not written yet.")
PY
```

Do not resubmit solely because the interactive session disconnects. Partial
episodes require a new campaign name; this diagnostic does not resume them.

## Verification boundary

The local CPU suite passed 194 tests, with one installed-SDK test skipped because
that optional SDK stack is unavailable locally. Local tests verify exact
preservation of answer-format-v1, one-time delegation,
native-example parsing, rejection of the observed non-call forms, unchanged
decoding/task settings, source mutation detection, distinct protocol identities,
gate isolation, and the dependency-aware dry-run/submission path. No parser or
upstream execution loop was changed. The subsequent Unity outcome is recorded at
the top of this document; successful job execution did not imply a passing
capability screen. New read-only transition diagnostics are tested locally;
their output on this campaign remains to be inspected.
