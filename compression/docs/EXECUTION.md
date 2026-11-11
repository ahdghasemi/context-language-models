# Current execution and evidence report

Date: 2026-11-11 (user timezone). This revision started from the existing feature branch
at `624369b637988eb7e06550395494406e99a4396d`; the fork/upstream default branches were both
`483198229ab7b90c0eed4651306ab3b3505aaf50`. The earlier branch's report is retained as
[PRIOR_EXECUTION_REPORT.md](PRIOR_EXECUTION_REPORT.md), explicitly historical. Its raw
measurements were not present in Git and are not treated as current observations.

The core audit/fixes were committed and pushed as
`6ba02ba536b567c5d7222ca19a259f463fdf8ddb`. Final documentation and the additional offline
agent initialization fix are in the subsequent branch commit. The delivery response
identifies that exact final SHA. No default-branch/upstream commit was changed.

## What changed in this revision

* Reconstructed the supplied GNN equation/code and Flow transfer again. Default training
  now retains GNN Adam/StepLR state across penalty stages; Flow Adam/cosine resets remain
  an explicit alternative. Initial, periodic and stage-boundary checkpoints preserve the
  state needed by the selected policy.
* Added a required dense CLM baseline stage before pilot submission; bound pilot-to-main
  gates to model/data revisions and protocol settings. Empty/duplicate/mismatched episode
  summaries cannot masquerade as a complete experiment.
* Added hashed source snapshots and configuration checks before execution; strengthened
  retry detection to include previous resubmission jobs; made all stage resources tunable.
* Enforced physically factorized modules at export and corrected explicit rank overrides.
* Materialized the benchmark counting vocabulary; pinned the same o200k budget ruler for
  both modes; prevented offline pricing downloads/proxy routing for localhost inference.
* Preserved original CLM files and their baseline entry points. The extension remains
  under `clm/tiny_clm` and `compression`.

## Observed status

| Item | Implemented | Executed this revision | Outcome / limitation |
|---|---|---|---|
| Supplied PDF/LaTeX/nested source audit | Yes | Yes | Hashes match attached inputs; algorithm page visually inspected |
| GitHub retrieval and dedicated-branch writes | Yes | Yes | Existing branch continued; core remote tree equals local staged tree |
| Projection/gradient/rank/factor/tying/save-load/cache/generation tests | Yes | Yes | Passed on CPU |
| Optimizer-policy restart and actual SIGUSR1 checkpoint test | Yes | Yes | Passed; both policies resume bitwise on the CPU fixture |
| Snapshot, configuration, budget, gate and scheduler dependency tests | Yes | Yes | Passed; scheduler submission is simulated in tests |
| Full inexpensive test suite | Yes | Yes | **43 passed**, one upstream Starlette/AnyIO deprecation warning |
| End-to-end random-weight smoke | Yes | Yes | 6 updates, export/reload/NLL/profile passed; software only |
| Pinned real model and WikiText acquisition | Yes | Yes | Weights, tokenizer and train/validation/test blocks prepared |
| Real dense NLL and SVD residual audit | Yes | Yes | First two validation blocks only; final test not evaluated |
| Real original and post-hoc SVD export/reload | Yes | Yes | Smaller SVD artifact; very poor post-hoc language quality |
| Real original vs SVD CPU profiling | Yes | Yes | Short diagnostic workload; see measurements below; no GPU claim |
| Original ContextBench task generation | Yes | Yes | 4 pilot and 15 main tasks; deliberately marked non-runnable without container |
| Both original agent classes and exact budget counters | Yes | Yes | Import/construction and tokenizer checks passed after offline fix |
| Real constrained/recovery/ASVD training | Yes | **No** | No CUDA; full optimizer/proxy footprint exceeds local RAM headroom |
| Apptainer/container and complete CLM episodes | Yes | **No** | No installed Apptainer/Singularity runtime or Unity session |
| Unity environment/pilot/main/final-test submissions | Yes | **No** | Dry-run planning and simulated dependencies only |
| Complete experimental aggregation/figures | Yes | Partly | Missing-result handling tested; no completed scientific campaign to plot |

Environment: Python 3.12.14, PyTorch 2.8.0+cpu, Transformers 4.57.1, datasets 4.4.1,
HF hub 0.36.2, safetensors 0.8.0, NumPy 2.3.4, Harbor 0.16.1, LiteLLM 1.109.0. Native
CPU math thread counts were two. `/proc/meminfo` reported 10,206,468 kB total RAM.
No GPU, Slurm executable or container runtime was available. `pip check` passed.
Syntax compilation, shell parsing, `git diff --check`, and Ruff E9/F checks passed.
A broader Ruff style pass reports pre-existing/import-order/style suggestions; it is
not reported as clean. No style-only wholesale rewrite was applied.

## Measured diagnostics, not compression-training results

Model: `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.
Data: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Preparation produced 2048 train, 64 validation
and 128 final-test blocks of 512 tokens. Final-test bytes were prepared but never scored.

The random model smoke reduced parameters from 20,672 to 14,144 and produced NLL
4.159977 on random validation data. This is not useful-language evidence.

The real-model diagnostic uses **only 2 validation blocks / 1,022 scored tokens**.
It is not a full benchmark, and neither row has any optimizer updates:

| Control | Unique parameters | BF16 serialized weight bytes | NLL | Perplexity |
|---|---:|---:|---:|---:|
| Original dense | 596,049,920 | 1,192,135,112 | 2.83301187 | 16.99658 |
| Post-hoc SVD, all MLP keep .8, rank 614 | 543,064,064 | 1,086,173,520 | 11.70195723 | 120,807.93 |

SVD reduced whole-model parameters by 8.8895% and the selected weights by about 20.05%.
Its degradation is severe. Serialization/reload error was exactly zero for both rows.
The SVD export's zero serialization gap is not a dense-to-SVD quality comparison: that
control was already factorized before export. The dense-vs-SVD table captures the loss.
The proposed constrained method's real deployment gap has **not** been measured.

The separate FP32 dense audit gave NLL 2.83404303 / PPL 17.01411. Exact reduced SVD of all
84 original MLP matrices took 71.53 seconds and peak process RSS reached 4,843,216,896
bytes. Squared-tail sums at keep .8/.6/.4 were 29,423.5575 / 53,963.9702 / 88,925.7371.
These are initial-weight diagnostics without backward passes or retained full proxy sets.

The two artifacts were subsequently profiled **sequentially**, with BF16 CPU weights,
two math threads, batch size 1, 128 prompt tokens and 4 forced decode steps. Each used
two warmups and three timed repetitions. This short shared-host diagnostic is not a
representative serving benchmark or a confidence interval across independent runs.

| CPU diagnostic | Dense | Post-hoc SVD |
|---|---:|---:|
| Median prefill seconds | 4.29245 | 4.62378 |
| Median seconds for 4 decode steps | 0.81987 | 1.11631 |
| Decode tokens/second (median-derived) | 4.8788 | 3.5832 |
| Peak process RSS bytes | 2,711,965,696 | 2,608,226,304 |
| Observed KV cache bytes | 15,138,816 | 15,138,816 |

The factorized control was smaller but slower in these measurements. RSS includes
process/framework/loading allocations and is not isolated model or activation memory.
Compression of MLP weights does not reduce the KV dimensions. Raw timing samples are
included in the supplementary execution evidence. An earlier concurrent timing probe
was excluded from this comparison. No GPU speedup or system-task latency was measured.

## Commands actually executed

Paths below are relative to the repository root; `../checks` contains local outputs.

```bash
git clone https://github.com/ahdghasemi/context-language-models.git context-language-models
git switch -c tiny-clm-constrained --track origin/tiny-clm-constrained
python -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r compression/requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip install harbor==0.16.1 -r compression/requirements.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  compression/tests clm/tests/test_context_string.py --junitxml=../checks/correctness-final.xml
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m tiny_clm.smoke --output ../checks/smoke
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HOME=/workspace/scratch/ee03ac542625/cache/hf \
  HF_DATASETS_CACHE=/workspace/scratch/ee03ac542625/cache/data \
  .venv/bin/python -m tiny_clm.cli prepare --config compression/configs/pilot.json \
  --destination ../checks/real-assets
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python compression/audit_backbone.py \
  --assets ../checks/real-assets/assets-lock.json --output ../checks/real-backbone-audit.json --svd
```

Diagnostic configs were copied from pilot and changed to `method=original` or `svd`,
`training.precision=float32`, and `training.validation_batches=2`; export remained BF16.
For each method, the following commands ran with OMP/MKL threads set to two:

```bash
.venv/bin/python -m tiny_clm.cli train --config ../checks/original-diagnostic.json \
  --assets ../checks/real-assets/assets-lock.json --run ../checks/real-original --device cpu
.venv/bin/python -m tiny_clm.cli export --run ../checks/real-original --device cpu
.venv/bin/python -m tiny_clm.cli train --config ../checks/svd-diagnostic.json \
  --assets ../checks/real-assets/assets-lock.json --run ../checks/real-svd --device cpu
.venv/bin/python -m tiny_clm.cli export --run ../checks/real-svd --device cpu
```

The profiling command ran for `original` followed by `svd`:

```bash
for method in original svd; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m tiny_clm.cli profile \
    --artifact "../checks/real-$method/export" \
    --output "../checks/real-$method/profile-sequential.json" \
    --lengths 128 --decode-tokens 4 --repeats 3 --device cpu
done
```

The subcommand name `train` does not imply optimization for `original` or `svd`: both
record zero training tokens and zero optimizer steps. Metadata records the actual source
HEAD and `dirty=true` while edits were in progress; these diagnostic runs are not falsely
attributed to a clean final commit. Campaign jobs require clean committed snapshots.

ContextBench was cloned at its configured revision and `tiny_clm.contextbench.prepare`
was called for pilot/main with `skip_container=True`. The result explicitly has
`runnable=false`. Both upstream counters were checked for exact o200k usage. Both
`ClmStream` and `ContextBenchAgent` were constructed with a localhost endpoint; no actual
agent episode or bash action was executed. CLI dry runs covered baseline, pilot, main
and ablations. GitHub Git-data calls created the reviewed trees/commits and advanced only
the feature ref with an expected-old-SHA lease and `force=false`.

## Failures and interpretation

* A first dependency-install command hit a log-directory creation race; retried after
  the directory existed, and dependency validation passed. No experiment was lost.
* The first actual CLM-agent import tried to fetch LiteLLM's remote cost map and failed
  through an unsupported local SOCKS proxy configuration. The episode subprocess now
  forces the bundled price map and disables proxy routing for localhost-only inference;
  both agent classes then imported/constructed successfully. No model/API credential was used.
* A single large GitHub create-tree request timed out (HTTP 422). Smaller incremental
  trees succeeded, and the final remote tree matched `git write-tree` exactly before
  publishing the commit. This was a transfer failure, not a failed training run.
* PyTorch emits a `torch_dtype` deprecation notice under the pinned Transformers version;
  the tested paths remain functional. The test suite emits one upstream deprecation warning.

No evidence here establishes preserved CLM capability or an advantage for the constrained
method. The useful next evidence is an allocated dense-baseline campaign, followed by
constrained/recovery training only if that baseline is competent. Short pilots may fail
because of insufficient optimization, so preserve residual/validation/deployment-gap
logs and distinguish that from a fundamental method failure. Current baselines and tiny
backbone scope do not justify a publication-level superiority or novelty claim.
