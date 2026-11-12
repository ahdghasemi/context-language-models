# Constrained compression of a CLM backbone

This fork implements the **Constrained formulation** from the supplied GNN work:
detached hard-rank auxiliary projections, an unnormalized squared-Frobenius coupling
penalty scaled by **1/(2μ)**, and decreasing-μ continuation. The default preserves the
original GNN Adam/StepLR state across stages. The separately supplied Regularized method is not used. SVD, recovery training and ASVD are separately named controls.

**Evidence boundary:** the latest local CPU suite passed 153 tests, with one SDK-
dependent test skipped, from a read-only source archive. Earlier CPU dense/SVD checks
are recorded in the [execution report](docs/EXECUTION.md). The user subsequently
executed Unity baseline evaluation, GPU profiling and ContextBench episodes. The
0.6B baseline and the corrected 4B screen scored zero; see
[export precision](docs/EXPORT_PRECISION.md) for limits on the legacy 0.6B results.
The 4B screen passed native-reference export checks but failed answer formatting
and some lookups. A [separate formatting-reminder experiment](docs/FORMAT_SCREEN.md)
is implemented but untested on Unity. Real-model constrained training and preserved
CLM capability after compression remain unestablished; no deployment speedup is claimed.

CLM is a context-file-editing harness around a language model. This extension compresses
the neural backbone, while using the original `ClmStream` agent and ContextBench grader.
The pilot uses the general post-trained **Qwen3-0.6B**, not the paper's Qwen3.5-9B CLM-RL
model. The supplied fork does not contain that trained checkpoint or its complete RL
training system. Small-backbone competence must be established before compression results
are interpretable; a failed dense CLM baseline now blocks the pilot itself.

## Read and run

1. [Source audit and feasibility](docs/SOURCE_AUDIT.md): inspected artifacts, disagreements, scope.
2. [Adaptation specification](docs/METHOD.md): equations, update order, ranks, memory and cost.
3. [Unity operating instructions](README_UNITY.md): setup, diagnostics, separate preparation,
   dry-run submission, resumption, immutable snapshots, final evaluation.
4. [Experimental protocol](docs/EXPERIMENTS.md): controls, acceptance gates, held-out data,
   focused ablations, failures and uncertainty.
5. [Related work and claims](docs/RELATED_WORK.md): what this transfer does and does not establish.
6. [Executed checks](docs/EXECUTION.md): observations distinguished from estimates and untested code.

All commands are run from the repository root unless stated otherwise. The Python package
is `tiny_clm` under `clm/`; the upstream packages remain unchanged. Installation uses
`pip install '.[compression]'` after choosing the PyTorch wheel. For a CPU software check:

```bash
OMP_NUM_THREADS=2 python -m pytest -q compression/tests clm/tests/test_context_string.py
OMP_NUM_THREADS=2 python -m tiny_clm.smoke --output /absolute/new/smoke-directory
```

The smoke uses random weights and toy tokens. It is not a language-compression experiment.
GPU training is deliberately refused outside a Slurm allocation.

## Artifact contract

Training checkpoints contain dense model/optimizer/continuation/sampler/RNG state.
`tiny-clm export` produces a different, deployable directory: `model.safetensors`, model
and generation configs, tokenizer, and `compression.json`. Selected weights are replaced
by two sequential linears. There are no training proxies, optimizer states, selected
dense weights, or forward-time dense reconstructions in this artifact. Tied embeddings
and output weights are counted and serialized once. Always load through
`tiny_clm.model.load_export`; generic `AutoModelForCausalLM.from_pretrained` does not know
the custom factors. vLLM/SGLang support for them is not implemented.

The reference endpoint uses Transformers/SDPA with normal within-generation KV caching.
Cross-request prefix reuse and CLM suffix-cache reuse are disabled for **all** comparisons.
Both dense and factorized models use this same server. Results cannot be advertised as
production-serving benchmarks or as SCR savings.

The default MLP budgets predict only **8.89%, 17.78%, and 26.61% whole-model parameter
reduction**, despite approximately 20%, 40%, and 60% reduction in selected weights.
These are analytical counts, not measured retained quality or runtime improvements.

No supplied manuscripts, private source archives, datasets, credentials, checkpoints,
caches or experiment outputs are committed. Source attribution and inspected revisions
are recorded in the audit; the adapted ContextBench container recipe has its original
license and change notice in `third_party/`.
