# Screen export fidelity against native caching

## Observed diagnostic result

The user executed the standalone cache diagnostic for the pinned
Qwen3-4B-Instruct-2507 checkpoint on an A100. It completed successfully:
`same_path_export_passed=true` and `same_path_generation_equal=true`.
The reported cached-versus-full-sequence differences were the same for native
and exported models:

| Diagnostic | Maximum logit error | Maximum total variation | Argmax agreement |
|---|---:|---:|---:|
| BF16 default, final prompt token | 0.5625 | 0.0000307076 | 1.0 |
| BF16 default, eight prompt suffix tokens | 0.875 | 0.0664594 | 1.0 |
| BF16 default, eight suffix tokens of synthetic 2048-token input | 0.75 | 0.0000457178 | 1.0 |
| BF16 math, final prompt token | 0.28125 | 0.0000128855 | 1.0 |
| FP32 math, final prompt token | 0.0000848770 | 0.0000000187154 | 1.0 |

Explicit versus implicit cache positions produced zero logit difference in all
reported cases. Native/export equality of generated tokens was checked
separately for cached and uncached generation. That result does **not** assert
that cached generation equals uncached generation. Native/export logit
agreement was tested at `rtol=atol=1e-4`, not reported as bitwise identity.

These controls support a BF16/backend numerical explanation for the original
screen failure. They do not show that cached and uncached probability
distributions are interchangeable. In particular, total variation reached
0.0664594 at a tested suffix position. Unchanged argmax choices on this limited
probe do not establish preservation over long trajectories or under sampling.
No ContextBench task capability or compression result follows from this audit.

## Explicit change in the screening criterion

Previously, export required the exported model's cached and full-sequence
logits to agree within `rtol=0.02, atol=0.25`. The native model itself failed
that criterion. It therefore could not isolate an export-induced regression.

The revised screen declares `cache_verification_contract=native-reference-cache-v1`
in its immutable campaign manifest, export audit and final report. It requires:

1. Identical parameter/buffer fingerprints and the existing full-sequence
   reload comparison at `rtol=atol=1e-4`.
2. Native versus exported logits at that **same unchanged export tolerance**
   for full-sequence, implicit-cache and explicit-cache execution, on each of
   the three BF16 default-backend inputs listed above.
3. Explicit versus implicit cache positions within `rtol=atol=1e-4` for each
   model/input, plus correct cache lengths at every teacher-forced update.
4. Identical native/export greedy output token IDs for each generation path,
   with up to sixteen generated tokens and unchanged generation settings.

Cached-versus-full comparisons remain in `cached_vs_full_diagnostics`, including
the original numerical tolerance and its pass/fail result, argmax agreement,
RMSE, KL divergence and total variation. They are now diagnostic measurements,
not an export-acceptance condition. This is a change in **what the gate tests**;
it must not be described as the old cache-equivalence criterion passing.

The deployment dtype, SDPA backend, cache implementation, tokenizer, prompts,
task subset, greedy decoding and provisional CLM reward floor remain unchanged.
The screen uses default BF16 inference; it does not silently switch serving to
the more accurate math/FP32 diagnostic modes. The constrained compression
algorithm and training implementation are untouched.

## Execution and limitations

Submit a **new** screen with the usual `compression/screen.py` dry-run and
`--submit` commands. Existing downloads and the task container are reused after
integrity checks; a new deployment export and native-reference audit are made.
The established dependency chain remains checks → prepare → export → two
ContextBench episodes → report. Old campaigns/audits are not modified, and an
old `cache_passed` flag cannot satisfy the new contract. The standalone
diagnostic report itself still does not authorize a screen or compression run.

Regression tests inject cached-logit errors below the former 0.25 tolerance,
full-sequence errors, incorrect positions, changed generated tokens, nonfinite
values, missing cases and widened tolerances. They also exercise a real random
BF16 model through all three probe inputs and physical save/reload. Tests keep
native cached/full differences visible even when export fidelity passes.

The full CPU suite passed from a read-only archive without Git metadata:
**118 passed, 1 SDK-dependent test skipped**. The integrated screen must still
run on the target GPU. A successful export permits the two ContextBench
episodes to start. Only adequate observed
CLM reward permits a full baseline and training-memory audit; it does not
authorize compression training or establish publication-level quality.
