# Four-task validation: memory fits, behavior gate fails

Follow-up: the user supplied the requested saved-response inspection. All three
inspected failed CLM episodes emitted ordinary command text without hitting the
output cap; the append-only example first truncated a tool call. See
[the diagnosis and isolated tool-call-guidance test](TOOL_FORMAT_SCREEN.md).

## Evidence received

The user supplied Unity output for the campaign produced by commit
`09ec4b9c73846218ededdee81a80951b83db7d11`. All four Slurm stages exited zero.
The report was complete, with no infrastructure errors, but its dense-behavior
gate failed. The source model was the uncompressed Qwen3-4B-Instruct-2507 with
the fixed `kv-answer-format-v1` reminder and 1,024-token completion limit.
These are user-executed measurements, not an assistant-run GPU experiment.

| CLM task | Reward | Answered | Operations delivered | Model calls | Parsed tool calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| R130, seed 1101 (development repeat) | 23/24 | 24 | 26 | 50 | 50 |
| R130, seed 1102 | 0 | 1 | 3 | 128 | 2 |
| R1000, seed 1101 | 0 | 0 | 1 | 128 | 0 |
| R1000, seed 1102 | 0 | 0 | 1 | 128 | 0 |

All four append-only episodes also scored zero, delivered only one operation,
and made 128 model calls with zero parsed tool calls. Here `tool_calls` comes
from server response parsing; the agent's usage and tool-result logs are needed
to confirm execution. Do not equate every parsed call with a successful command.

The development result repeated exactly in reward, token counts, and call counts.
**None of the three additional validation tasks scored above zero.** This is a
completed negative validation result. It does not support a robust dense CLM
baseline or justify starting the compression-quality grid. The zero-command
episodes stop before retrieval is tested; this output alone cannot locate the
failure in generation, syntax, parsing, or upstream action handling. It is not
evidence of a long-context retention limit or an effect of compression.

An earlier append-only trace had a truncated first tool-call attempt followed by
plain command text. That makes completion truncation a concrete hypothesis for
the new runs, but it is not verified by these aggregate counts. Do not increase
the output limit, repair the parser, add prompt examples, or change the model
simultaneously. First read the saved failed responses and server finish reasons.
Any later protocol change must be versioned and applied to all compared methods;
this failed campaign and its threshold stay unchanged. Further tuning on these
tasks also consumes validation evidence: a final study needs fresh untouched seeds.

## Resource measurements

The separate two-update probe passed its provisional memory criterion:

| Quantity | Observed |
| --- | ---: |
| Device capacity | 79.25 GiB |
| Peak CUDA allocated | 72.12 GiB |
| Peak CUDA reserved | 74.21 GiB |
| Capacity minus peak reserved | 5.04 GiB |
| Peak host RSS | 23.52 GiB |
| Temporary checkpoint file | 64.73 GiB |
| Whole probe elapsed | 427.47 s |
| Trainer elapsed | 115.21 s |
| Projection time | 61.86 s |
| Forward/backward and optimizer-update time | 4.09 s |

The probe used all-trainable FP32 master weights, BF16 autocast, gradient
checkpointing, down projections at rank 1,216, single-tensor Adam, batch 1,
sequence length 512, and four accumulated microbatches. It processed 4,088
scored random tokens. Its `trainer_status.status=incomplete` is intentional:
the resource test stops at two updates and deletes its temporary checkpoint.
It is not a failed resume or an unreported quality-training run.

The projection total includes the two optimizer-boundary refreshes and the
validation refresh. It must not be extrapolated as a measured per-update rate
for a different validation frequency. The trainer elapsed time also includes
validation, checkpoint I/O and other overhead; the whole probe additionally
loads/verifies the model and cleans up temporary files. No inference speedup
was measured here. Frequent roughly 65-GiB training checkpoints will carry
substantial storage and I/O cost.

The configuration fits this short workload with limited headroom. This does
not establish long-run stability, full continuation-stage behavior, resume
memory, or the safety of a larger batch, context, rank or target set. The all-MLP
4B configuration remains analytically too large for this GPU. No physically
compressed inference artifact or preservation of CLM quality was produced.
The headroom calculation uses PyTorch's allocator statistics; it does not
include every allocation made outside that allocator.

## Read-only next diagnostic

`compression/inspect_contextbench.py` now supports `protocol_screen` and
`protocol_validation` as well as original screens and baseline campaigns. It
does not require a passing quality gate, load weights, execute saved commands,
write to a campaign, or submit Slurm jobs. It checks episode/summary protocol
identities and records, and checks server usage totals and weight identity.

It prints bounded early/final retained-message excerpts, following tool results
or reminders, agent command counters, finish-reason counts, and selected server
request metadata. Request numbers are one-based. A `length` finish reason and
an unfinished tool-call payload would support truncation; a complete malformed
payload or bare command text would indicate a different boundary failure.
Reaching the configured token cap alone is not proof of truncation. The server
does not save raw generated token IDs, and edited snapshots are not a full
execution history or a guaranteed one-to-one match to request-log rows.

From the updated checkout, run with system Python; no environment activation
or GPU allocation is required:

```bash
python3 compression/inspect_contextbench.py \
  --campaign /work/pi_aghasemi_umass_edu/tiny_clm_work/campaigns/validate-qwen3-4b-format-001 \
  --episode clm--kv-store-R1000-seed1101 \
  --episode clm--kv-store-R1000-seed1102 \
  --episode clm--kv-store-R130-seed1102 \
  --episode append_only--kv-store-R130-seed1101 \
  --samples 3 --chars 1200
```

This diagnostic has local synthetic-log tests. The assistant has not accessed
the new Unity trajectories; their contents remain needed before selecting the
next intervention. No new GPU experiment is submitted by this change.
