# Execution and evidence report

Date: 2026-11-11. Development started at
`483198229ab7b90c0eed4651306ab3b3505aaf50`; measurements made before the delivery commits
record that starting SHA with `dirty=true`. They are not falsely attributed to a clean
release commit. Future Slurm campaigns require clean committed code and record their
exact immutable snapshot. Delivery commits are available in the feature-branch history.

## Status

| Item | Implemented | Executed here | Outcome / limit |
|---|---|---|---|
| Supplied paper/LaTeX/code audit | Yes | Yes | Exact Constrained mechanism and material disagreements documented |
| Target fork retrieval/read/write access | Yes | Yes | Clone and authenticated feature-branch creation succeeded; no upstream/default-branch update |
| Projection, gradient, factor, tying, save/load, generation/cache tests | Yes | Yes | Passed on CPU |
| Restart/config collision and matched-token controls | Yes | Yes | CPU resumed weights match uninterrupted weights bitwise |
| HTTP/native tool parsing and upstream context-file contract | Yes | Yes | Passed software tests; malformed tool calls remain visible |
| Slurm dependency/grid/failure/task-integrity tests | Yes | Yes, planner/unit tests | Passed; no actual scheduler available |
| Complete test suite | Yes | Yes | **33 passed**, one upstream Starlette/AnyIO deprecation warning |
| Random tiny end-to-end smoke | Yes | Yes | Train → physical export → reload → evaluation → profile passed; software only |
| Public model/data preparation | Yes | Yes | Real pinned weights/tokenizer and native dataset splits loaded |
| Real uncompressed language/interface check | Yes | Yes | Limited NLL and native text/tool-call formatting worked; no tool execution |
| Real post-hoc SVD export | Yes | Yes | Physically smaller and reload-exact; language quality failed badly |
| Real-model constrained/recovery/ASVD training | Yes | **No** | No CUDA, insufficient local RAM for full optimizer footprint |
| Original ContextBench task packaging | Yes | Yes, without container | 4 pilot and 15 main task manifests generated; marked non-runnable |
| Apptainer build and original CLM/append-only episodes | Yes | **No** | No local Apptainer/Singularity or Unity allocation |
| Main/final-test Slurm campaigns | Yes | **No** | Dependency planning/static tests only |
| CUDA deployment profiling | Yes | **No** | No GPU measurements, speedup or VRAM-retention claim |
| Real dense vs post-hoc CPU profiling | Yes | Yes | Smaller SVD weights, lower observed decoding throughput; diagnostic shared CPU environment |
| Aggregation/plot/frozen held-out workflow | Yes | Partly | Missing-run aggregation tested; actual complete CLM campaign/plots/final tests not run |

Environment: Linux x86_64 runtime, Python 3.12.14, PyTorch 2.8.0+cpu, Transformers
4.57.1, datasets 4.4.1, HF hub 0.36.2, safetensors 0.8.0, NumPy 2.3.4, Harbor 0.16.1,
LiteLLM 1.109.0. Thread counts were fixed to 2. No CUDA device; `/proc/meminfo` reported
10,206,472 kB total RAM. `lscpu` could not access the virtualized CPU inventory, so a CPU
model name is not asserted. These are local diagnostic results, not Unity benchmarks.

## Observed numerical results

The random tiny Qwen fixture trained 6 optimizer updates / 180 scored tokens. Parameter
count fell from 20,672 to 14,144; random validation NLL was 4.161430. This says nothing
about useful language or CLM retention.

Actual Qwen3-0.6B model revision was
`c1899de289a04d12100db370d81485cdf75e47ca`; WikiText revision was
`b08601e04326c79dfdd32d625aee71d232d685c3`. Preparation produced 2,048 train, 64 validation
and 128 test blocks of 512 tokens. Test blocks were prepared but **not evaluated**.

The following is a **two-validation-block diagnostic**, 1,022 scored tokens, same native
tokenizer and BF16 exported weights. It is not a pilot-quality table and does not include
the proposed constrained method:

| Control | Unique parameters | Serialized weight bytes | NLL | Perplexity |
|---|---:|---:|---:|---:|
| Original dense | 596,049,920 | 1,192,135,112 | 2.83405304 | 17.01428 |
| Post-hoc SVD, all MLP keep .8, rank 614 | 543,064,064 | 1,086,173,520 | 11.70345831 | 120,989.41 |

The SVD artifact is 8.8895% smaller in parameter count but is **not useful compression**
on this diagnostic. No recovery updates were applied. Its export/reload maximum logit
error was 0.0; its before/after *serialization* gap was also zero because this control
was already factorized before export. That zero does not conceal the large dense-to-SVD
quality loss. The earlier FP32 dense diagnostic was NLL 2.83403063 / PPL 17.01390.
Neither measurement establishes long-horizon CLM capability.

Real BF16 CPU reference profiling used batch 1, 32 fixed decoding steps, 2 warmups and
5 measured repeats per prompt length, with 2 threads, sequentially in the same runtime:

| Control | Prompt tokens | Median prefill seconds | Decode tokens/second |
|---|---:|---:|---:|
| Dense | 128 | 0.6501 | 9.8273 |
| SVD keep .8 | 128 | 0.6129 | 8.6012 |
| Dense | 512 | 3.1710 | 6.5765 |
| SVD keep .8 | 512 | 3.4025 | 6.0414 |

Dense resident parameter bytes were 1,192,099,840 versus 1,086,128,128 for factors.
KV cache bytes were identical: 18,350,080 / 62,390,272 at prompt lengths 128 / 512 plus
32 decode steps. Peak process RSS was 2,695,090,176 / 2,595,065,856 bytes for dense / SVD;
this includes loading, not isolated activation memory. Decoding was slower for factors
in these observations; the 128-token prefill difference is small relative to observed
variation (sample SD about .10 s). CPU allocation/exclusivity and physical model name
were unavailable. These are reference CPU diagnostics, not evidence of GPU performance
or a stable production speed ranking. Raw repeated samples remain in local profile JSONs.

The real dense reference generated `hello` on a one-word request, and produced a native
`bash` tool call with JSON argument `{"command":"pwd"}` when requested. The command was
**not executed**. This verifies a narrow generation/serialization path, not tool success,
context editing or an evaluated CLM episode.

Full reduced SVD of all 84 original MLP matrices took 55.33 seconds on CPU, peak RSS
4,820,869,120 bytes, without backpropagation or retained proxies. Initial summed squared
tail errors at keeps .8/.6/.4 were 29,423.5572 / 53,963.9696 / 88,925.7379. These measurements
motivate the penalty audit and resource cautions in the specification. A small deployment
artifact does not imply a cheap compression-training procedure.

## Commands and validation scope

The working checkout was `.../context-language-models`, the venv `../.venv` and diagnostic
outputs `../checks/`. The following commands were actually executed (outputs remained
outside Git):

```bash
git clone https://github.com/ahdghasemi/context-language-models.git
git checkout -b tiny-clm-constrained
../.venv/bin/python -m pip install -e '.[compression]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=clm ../.venv/bin/python -m pytest -q \
  compression/tests clm/tests/test_context_string.py --junitxml=../checks/final-correctness.xml
../.venv/bin/python -m pip check
../.venv/bin/ruff check clm/tiny_clm compression
python -m compileall -q clm/tiny_clm compression
bash -n compression/setup_unity.sh compression/slurm/job.sbatch compression/slurm/final_test.sbatch
git diff --check
python compression/submit.py pilot --campaign /tmp/tiny-clm-dry-run
python compression/submit.py main --config compression/configs/main.json --campaign /tmp/tiny-clm-main-dry-run
python compression/submit.py ablations --campaign /tmp/tiny-clm-ablation-dry-run
```

`tiny_clm.smoke` was run to `../checks/smoke`. Asset preparation was executed through the
CLI with project-local HF/data caches. The original preparation requested `main`, resolved
and saved the exact revisions above; released configs now pin those resolved revisions.
Read-only Python calls loaded that manifest for the first dense/interface/SVD-residual
audits. `compression/audit_backbone.py` packages that read-only model/NLL/SVD audit for
reproduction; it is not a trainer. The task builder was called programmatically with
`skip_container=True`, which explicitly writes `runnable=false`.

The real post-hoc diagnostic copied the pilot config to `../checks/real_svd_diagnostic.json`,
set `method=svd`, CPU precision FP32 and validation_batches=2, and retained the already
resolved asset request. The dense diagnostic changed only `method=original`. Both export
BF16. Actual commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m tiny_clm.cli train \
  --config ../checks/real_svd_diagnostic.json --assets ../checks/real_assets/assets-lock.json \
  --run ../checks/real_svd_run --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m tiny_clm.cli export \
  --run ../checks/real_svd_run --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m tiny_clm.cli train \
  --config ../checks/real_dense_diagnostic.json --assets ../checks/real_assets/assets-lock.json \
  --run ../checks/real_dense_run --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m tiny_clm.cli export \
  --run ../checks/real_dense_run --device cpu
```

The CLI subcommand is named `train`, but these two methods execute **zero optimizer
updates**; they must not be reported as fine-tuned models.

Both real artifacts were also profiled with these actual commands:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m tiny_clm.cli profile \
  --artifact ../checks/real_dense_run/export --output ../checks/real_dense_run/profile.json \
  --lengths 128 512 --decode-tokens 32 --repeats 5 --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python -m tiny_clm.cli profile \
  --artifact ../checks/real_svd_run/export --output ../checks/real_svd_run/profile.json \
  --lengths 128 512 --decode-tokens 32 --repeats 5 --device cpu
```

## Failures, limitations and remaining evidence

Initial test collection failed because installing the upstream base dependencies first
selected HF hub 1.33, incompatible with the pinned Transformers requirement. Dependency
resolution also exposed Harbor requirements for datasets>=4.4.1 and FastAPI>=.128. The
compression extra/setup now pins a compatible combined environment; `pip check` passes.
Torch's deprecated `torch_dtype` argument and the single Starlette test warning are
nonfatal under the pinned versions.

Shell `git push --dry-run` failed because this checkout had no HTTPS credential helper.
Authenticated GitHub repository permissions did include push; the Git data API is used
for creating commits and advancing only the dedicated branch, followed by remote fetch
and content verification. Lack of shell credentials is not treated as lack of GitHub
write access. Unity access remains absent regardless of GitHub access.

Before publication-level conclusions: run the allocated environment/container checks;
establish dense CLM competence; train the constrained/recovery/ASVD grid; quantify real
deployment gaps and quality frontiers; measure GPU memory/runtime on matched hardware;
inspect failed and long-horizon episodes; freeze and execute held-out evaluation; broaden
general capability assessment and repeat useful operating points over enough independent
seeds. No constrained-model retention, GPU speedup, original RL reproduction or new
theoretical guarantee has yet been established.
