# Zero-reward KV screen: response protocol diagnosis

## Evidence and its limits

The user executed the screen at commit
`3d3a7b8dd6d2cd2868ebb814dfc096269e3088ab` with the pinned, uncompressed
Qwen3-4B-Instruct-2507 checkpoint. All five jobs completed. Parameter/buffer
identity, zero-error full-sequence reload, native-reference cache fidelity,
and native/export generation equality on the probes passed. These are export
checks, not a capability result.

On the single development task `kv-store-R130-seed1101`:

| Mode | Model calls | Tool calls | Delivered operations | Recognized answers | Official reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLM | 26 | 26 | 26/26 | 0/24 | 0 |
| Append-only | 128 | 0 | 1/26 | 0/24 | 0 |

The user then supplied first/last retained-message excerpts. In CLM, two batch
offload commands completed and context edits were reported as applied. The
sampled GET commands used `grep` and `cut` to print value-like strings, followed
by the readiness marker. They omitted the required `<<<ANSWER key=...>>>` and
`<<<ANSWER END>>>` delimiters even while the accompanying prose promised them.
The supplied excerpts cover four GET responses; they do not establish the
correctness of all 24 retrieved values. The stream ended as `stream_exhausted`.

In append-only, the first reply attempted a long tool-call JSON containing
individually copied SET values. It ended mid-command, and server accounting
records one `length` finish. No structured call was executed. Later replies
were plain `echo READY_FOR_NEXT_OP` text, not tool calls. The agent reached
`max_steps`. The upstream reminder saying "Your command ran normally" is
misleading in this branch: its saved usage reports zero commands. This does
not establish a model-server crash or an export regression.

Primary code inspected at ContextBench revision
`fb533c18c78a82d3f1b2047a9c1700735cf829ae`:

- [`tasks/kv_store/scaffold.md`](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/tasks/kv_store/scaffold.md): an ANSWER block must actually appear in context; merely finding a value is insufficient.
- [`graders/kv.py`](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/graders/kv.py) and `graders/base.py`: recognize blocks across saved contexts, keyed by the queried key.
- [`harbor/agents/base.py`](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/harbor/agents/base.py), `harbor/turn_source.py`, and `stream.py`: distinguish structured command execution from marker mentions and record delivery/termination.
- [`tasks/kv_store/generate.py`](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/tasks/kv_store/generate.py) and `harbor/package.py`: define query IDs and packaged gold/stream locations.
- Local `clm/tiny_clm/serve.py::parse_response`: incomplete tool markup stays visible as text; no command is invented or executed.

The original task already explains answer formatting. This is an observed
instruction-following failure on the sampled GETs, not evidence that the task
forgot to specify a format. The zero score remains valid under that contract.
The excerpts also show useful offloading/retrieval behavior that the single
reward does not describe. Neither observation justifies claiming preserved
CLM quality, successful compression, or a population-level model limitation.

## Read-only audit of all saved GET responses

Run with system Python; no dependencies, GPU, model download, or job submission:

```bash
python3 compression/audit_kv_responses.py --campaign "$TINY_CLM_SCREEN_CAMPAIGN"
```

Add `--details` for every query, input hash, and paired observation. Output goes
only to stdout. No file, score, gate, model, task, or environment is changed.
The script never executes commands from a trajectory or passes gold to a model.

The audit checks the screen's manifest identity, task-lock hash, task file tree,
episode record, operation-stream identity and delivery log. It uses snapshots
and the final trajectory. Only a tool result paired by call ID with a structured
bash call under a delivered GET is eligible. Identical repeated observations
count once; conflicting retained copies are excluded and listed. Unknown or
nonzero exit status is not success. Input SET values, unstructured replies and
unpaired outputs are not counted as retrieval evidence.

For each GET it separately reports:

1. Whether a response could be observed and its command exit status.
2. Whether the expected value appears as a complete line in a successful saved
   tool result (whitespace normalized).
3. Whether that result contains an answer block for the correct key and value.

These are post-hoc diagnostic counts, **not a replacement benchmark score**.
In particular, they do not score assistant answers, recover deleted outputs,
or infer success from an expected value elsewhere in the transcript. Missing
observable evidence is not proof of a failed lookup. Detailed output preserves
source hashes for inspection. `can_advance_to_compression` is always false.

Tests cover repeated frames, input leakage, wrong-query values, missing tool
calls, failed/unknown exits, JSON tool outputs, conflicting edited messages,
changed task identities, incomplete runs, and byte-for-byte preservation of
inputs. The actual full Unity transcripts still require this audit; only the
user's supplied excerpts were inspected during implementation.

Local CPU validation: **31 passed** for the new audit tests, existing retained-
message inspector tests and reference-server protocol tests:

```bash
PYTHONPATH=clm python -m pytest -q compression/tests/test_kv_response_audit.py \
  compression/tests/test_inspect_contextbench.py compression/tests/test_protocol.py
```

An initial combined invocation omitted `PYTHONPATH=clm` and failed collection
of the existing protocol test; the corrected command above passed. No new GPU
inference or Unity submission was performed while adding this diagnostic.

## Next experiment depends on this diagnosis

The user subsequently executed the full audit: **19/24** queries contained the
exact expected value as a complete line; **0/24** had correct answer blocks.
There were no conflicting or unpaired call records. The five other queries
searched the first batch file for keys in the second batch and returned only
the readiness marker. See [the separate formatting experiment](FORMAT_SCREEN.md)
for the diagnosis, unchanged official scores and a bounded next experiment.

If retrieval is correct but answer framing is consistently missing, the next
bounded experiment should isolate a disclosed protocol-formatting reminder,
with the original failed screen retained as its control. It must leave the
official grader unchanged and use the same prospective protocol for dense and
compressed models. Such a variant changes the evaluation setup and must be
identified in configuration, provenance and later claims. It is implemented
separately; this read-only audit neither applies nor evaluates that reminder.

Append-only's output-cap failure is a separate issue. Increasing the output
cap, changing the reminder, and changing the prompt together would confound the
diagnosis. No new inference campaign, model-size escalation, weakened quality
gate, or compression training is justified by these diagnostic counts.
