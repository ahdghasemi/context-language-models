# Experimental protocol

This is a staged, exploratory study. Thresholds below are **provisional**, not user-specified
scientific acceptance criteria or evidence of retention. Configuration files and selection
records must be fixed before each stage. Never tune on final test outputs.

## Questions and primary endpoints

1. Does factor export actually reduce whole-model parameters and serialized weights?
2. Does the accessible dense backbone perform useful CLM behavior under the original
   context-editing harness? If not, do not interpret a small dense–compressed difference
   near zero reward as successful retention.
3. At comparable whole-model size, does constrained training improve general NLL and
   CLM task quality over post-hoc decomposition/recovery and activation-aware compression?
4. Does smaller storage improve measured memory or runtime on the same backend/hardware?
   A slowdown is a result, not a reason to omit the configuration.

Primary quality measures: validation/final-test token-weighted CE NLL and perplexity for
general language modeling; original ContextBench reward and family-specific verifier
metrics for CLM behavior. Report all KV/log/needle/sudoku metrics, failures, operation
delivery/termination codes and trajectories, not only the macro mean. The capped
WikiText split is a documented sample, not canonical full-corpus WikiText perplexity.
No claim of broad reasoning/instruction retention follows from it.

Deployment measures: unique parameters; selected/whole-model fractions; actual serialized
weight bytes; GPU peak allocated/reserved memory; parameter residency bytes; KV bytes;
prefill latency/throughput; fixed-length decoding throughput; complete task latency,
prompt/completion token totals, tool calls, model calls and realized context lengths.
Activation memory is not separately isolated: peak allocated includes activations,
workspace and cache. Host peak RSS includes loading. Compression-training cost includes
tokens, elapsed time, projection time, initialization/calibration/search, validation and
checkpoint overhead. Scheduler wall time is an additional check.

## Comparisons and budgets

| Method ID | Initialization / extra training | Role |
|---|---|---|
| `original` | Original pinned checkpoint; zero optimizer updates | Uncompressed baseline in the same reference server/CLM setup |
| `dense_matched` | Same checkpoint, CE updates/tokens matched to constrained | Separates fine-tuning effects from compression |
| `svd` | Exact post-hoc hard SVD; zero recovery tokens | Decomposition-only control |
| `svd_recovery` | Same post-hoc factors, then CE factor training with matched updates/tokens | Essential recovery control |
| `asvd_uniform` | Activation channel scaling from 16 training blocks, fixed matched ranks | Activation-aware control with identical rank allocation |
| `asvd_adaptive` | ASVD scaling plus calibration-NLL layer sensitivity allocation | Stronger contemporary baseline; extra search cost is disclosed |
| `constrained` | Dense CE+hard-rank coupling continuation, final factor export | Proposed method only |

`asvd_uniform_recovery` is also supported for a separately configured recovery extension;
it is not silently included in the default grid. No external teacher or quantization is
used. ASVD is an independent exact-SVD implementation of the cited mechanism, not an
official-paper-result reproduction. It operates on the same audited MLP target set,
uses actual parameter budgets and fails rather than randomizing a failed decomposition.

First run the **baseline stage** (one original model, zero optimizer updates). Its
CLM rewards must meet the provisional .50 floor before pilot submission.

Pilot: seed 0, keeps .8/.6, **10 runs** including original/dense, paired SVD/SVD-recovery,
ASVD-uniform and constrained. Main: keeps .8/.6/.4, trained seeds 0/1/2, original and
deterministic post-hoc controls once per budget, adaptive ASVD once per budget: **31 runs**.
Each run has its own output/config and every manifest row appears in aggregation,
including incomplete/failed ones. ASVD adaptive allocations can overshoot the requested
reduction slightly due to rank granularity; compare actual whole-model size, not just
requested keep. If materially unmatched, report a size/quality frontier or add matched
budget points, not a method advantage from a different size.

All trained controls share the original checkpoint, training tokens/order for a given
seed, learning rate, stage lengths/optimizer policy, precision and validation frequency.
Constrained selects best validation augmented loss, while dense/factor recovery selects
validation CE; this is explicit, source-derived algorithm behavior. Baseline tuning is
limited, so superiority cannot be claimed from this default grid alone. Recovery rates
or budgets may need fair additional validation tuning and those trials must be counted.

## CLM capability and controlled system comparison

Use original `clm_contextbench:ClmStream`, unchanged, and the original ContextBench
append-only agent `contextbench.harbor.agents.base:ContextBenchAgent`, both against the
same dense/compressed reference endpoint. This 2×2 backbone × context-method comparison
helps distinguish generic backbone degradation from loss specific to context editing.
The two context methods have their native different prompts/skills; within each method,
dense/compressed prompts, task data and decoding are identical. It is not a comparison
with all paper baselines or an exact zero-shot paper reproduction.

Pilot tasks: KV store R130 and R1000, seeds 1101/1102 (4 episodes per method, 8 per model).
Main adds needle retention K4-C8, log triage L1600 and sudoku sketchpad t30, seeds
2101/2102/2103 (15 per method, 30 per model). These probe updating/recalling state,
retention under distraction, accumulated logs and long-horizon state use. R1000 is
deliberately a difficult boundary; failure is informative. Inspect delivered/total
operations and termination reasons to distinguish model errors from truncation.

Paper-level task feasibility budgets stay 32768 tokens with 2048 reserve. Reference
decoding is fixed to temperature 0, top_p 1, thinking disabled, 1024 output tokens,
128 pilot / 256 main maximum steps and 1800 seconds per episode. These differ from
the paper's thinking/sampling/output settings and are recorded scope limitations.
Timeout is not an acceptable way to manufacture a quality win through shorter latency.
Infrastructure exceptions remain failed episodes; a valid graded zero is a negative
quality result. Hashes of original tasks, container and exact source revision are checked.

Cross-request prefix reuse and SCR are off. Within-generation KV caching is on in every
model. Thus weight savings cannot be conflated with shorter context or suffix-cache
savings. Production cache interactions require a separate future backend experiment.

## Stage gates and final test

Correctness must pass before smoke, and smoke before training arrays. The baseline
reference campaign must pass its CLM competence gate before the pilot can be submitted.
Main/ablation gates bind model and dataset revisions, loss/precision/optimizer settings,
layer policy and decoding; changing a model ID alone cannot bypass a new pilot. Correctness
includes the defining projection/gradient/deployment properties, original context mirror,
generation/caches and restart. A random smoke only demonstrates software behavior.

The pilot gate requires every expected run to complete and at least one constrained
candidate to meet: >=5% whole-model reduction; perplexity <=1.10× **both** original and
matched dense; save/reload maximum logit error <=1e-4; and on every pilot CLM episode,
both dense references have reward >=.50 and compressed drop is <=.05 absolute. The
per-episode screen is intentionally strict and small-sample; it is not a significance
test. A real speedup is not required to run the main quality study, but is required to
claim a speedup. A failed gate blocks the main submission driver.

If no gate passes, diagnose the dense baseline, protocol compatibility, residual/export
gap, μ strength, recovery budget and layer policy before revising any thresholds.
Record the old failed campaign and the reason for changes. A larger backbone requires
new memory estimates and a new pilot; hybrid Qwen3.5 support requires an architecture
implementation/audit, not a config-only substitution.

Use validation to select operating points, including original and matched dense controls
in `freeze_selection.py`. The resulting read-only record fixes artifact hashes, rationale
and disjoint final ContextBench seeds (default 3101/3102/3103). `submit_final.py` builds
fresh tasks, evaluates selected artifacts on the native WikiText test cap and those new
episodes, and aggregates all selected outputs. Jobs use the original code snapshot and
environment. Existing successful tests are read rather than overwritten. Failed episodes
must be preserved before a documented retry. Do not reselect models after observing test.

## Profiling, variability and failure analysis

Reference profile workloads use real validation token sequences at lengths 128, 512 and
2048, batch size 1, forced 32 decoding steps, 2 warmups and 7 repeats. CUDA synchronization
brackets timing. Raw repeated samples, median/mean/SD, software/GPU identity and memory
are saved. These repeated kernel timings are not independent model-training replicates.
Use matched GPU constraints, verify the actual GPU in metadata, and avoid mixing hardware
in a latency claim. Longer CLM contexts are measured through actual tasks, where behavior
can change token counts/calls. Fixed-workload speedups and end-to-end task speedups may
have different signs.

Aggregation pairs the same tasks and training seeds against matched dense. Exploratory
hierarchical bootstrap resamples training seeds and evaluation-seed clusters; all levels
sharing an evaluation seed stay together. Three seeds is weak statistical evidence.
Report intervals and raw per-run values, not token-level pseudo-replications. CLM final
results retain complete verifier dictionaries; publication plots should analyze each
family and trajectory length, not just pooled reward. Deterministic original/SVD/ASVD
controls are not relabeled as three independent training runs.

`submit.py ablations` schedules one seed for original/dense, continuation versus constant
weak/strong μ (identical optimizer policy/update counts), down-only versus attention selection, and
their matched SVD-recovery controls: 9 runs. Down keep .4 and attention keep .7 approximately
match all-MLP keep .8 whole-model size; use actual counts to check. Constant μ is explicitly
an ablation of continuation. The main grid already tests compression budget. Replicate
interesting ablations over more seeds only after this focused screen.

Inspect: penalty/residual evolution; pre-export vs exported NLL/logit changes; cached
generation failures; first task turn that changes state incorrectly; divergence growth
with trajectory length; extra model/tool calls; low-rank kernel launch overhead; SVD
training cost. If NLL collapses before export, suspect optimization/data/schedule. If only
export collapses, the coupling has not closed the deployment gap. If quality survives
but runtime slows, investigate the backend/kernel sizes rather than claiming algorithmic
speedup. Negative cases remain in tables and plots.


Budget counter: both methods explicitly use o200k_base with the CLM served-token ratio
calibration disabled. The preparation job downloads the exact counting vocabulary and
records its hash; missing caches fail before episodes rather than silently selecting
character-count fallbacks. Qwen native token totals are reported separately, so a 32768
benchmark-token budget must not be described as exactly 32768 native Qwen tokens.
The mode comparison retains each upstream method's own prompt and finish behavior.
