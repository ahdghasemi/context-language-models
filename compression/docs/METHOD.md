# Precise adaptation specification

## Objective and update

Let θ be all backbone parameters, S the selected matrices, and r_s their fixed ranks.
The intended constrained problem is

\[
\min_\theta L_{CE}(\theta),\qquad \operatorname{rank}(W_s)\le r_s\ (s\in S).
\]

The executable penalty formulation is

\[
\min_{\theta,\{V_s\}} L_{CE}(\theta)+\sum_{s\in S} I_{\operatorname{rank}\le r_s}(V_s)
+\frac{1}{2\mu}\sum_{s\in S}\|W_s-V_s\|_F^2.
\]

`L_CE` is mean next-token cross entropy on non-overlapping 512-token blocks: 511
scored targets per block. It is general compression-aware fine-tuning from a pretrained
post-trained checkpoint, not training from scratch or CLM RL. Every backbone parameter
is fine-tuned; embeddings/head are retained dense. Context management itself remains
the original harness and is evaluated separately.

At each optimizer update, compute the exact reduced SVD and detach
`V_s = U[:, :r] diag(σ[:r]) Vh[:r, :]`. Then hold V fixed for the accumulated
microbatches and one Adam step. The penalty gradient is `(W_s - V_s)/μ`.
There is **no** singular-value soft threshold, nuclear norm, learned adapter, pruning,
quantization, distillation, dual multiplier or ADMM update. SVD inside this alternating
method is not a replacement by post-hoc SVD.

```text
load pinned pretrained θ; fix selected matrices/ranks; seed sampler and RNGs
initialize Adam and StepLR (step_size=stage_steps, factor=sqrt(γ))
for μ in [μ_init, γ μ_init, ...] while μ >= μ_min:
    best_score = infinity
    for exactly stage_steps optimizer updates:
        for each selected matrix: V_s = stop_gradient(P_rank<=r_s(W_s))
        zero gradients
        accumulate mean CE gradients across equal-sized microbatches
        backpropagate sum_s ||W_s - V_s||² / (2 μ), once
        Adam step; StepLR step; advance sampler/counters
        periodically evaluate validation CE + freshly projected penalty
        retain the actual best post-update weights and score on the host
        periodically atomically save all restart state
    restore best weights; retain Adam moments and scheduler for next stage
project final selected matrices and export their factors
measure validation pre-export/export gap and save/reload agreement
```

All stages are finite update budgets; no convergence certificate is claimed. The
continuation list includes the minimum when reached to floating-point tolerance.
Smaller μ means stronger coupling. Final dense weights need not satisfy the exact
rank constraint; final factors do. Final projection can damage quality and is tested.

## Source-to-code mapping

| Original mathematical component | Original code | CLM implementation |
|---|---|---|
| Task objective L | GNN `utils.py:sr_loss`, Flow `src/tinyfm/train.py` CFM loss | `tiny_clm/train.py:nll`, `_train`: token CE |
| Selected W_s | GNN `utils.py:apply_proximal_and_compute_penalty`; Flow interior linears | `tiny_clm/lowrank.py:select`, explicit Qwen3 policy |
| Rank indicator proximal map | GNN `utils.py:ProximalOperatorNuclearNorm.apply`; Flow `model.py:svd_projection` | `lowrank.py:factors`, `project`, `Constraint.refresh` |
| Unnormalized 1/(2μ) Frobenius coupling | GNN `utils.py:272`; Flow `train.py:54` | `Constraint.penalty`, once per accumulated optimizer update |
| Decreasing μ warm start | GNN `main_train_gradual_part1.py:103–120`; Flow `train.py:39–58` | `mu_schedule`, `_train` stage loop |
| Best stage weights | GNN held-out augmented loss; Flow sampled training CFM loss | Held-out augmented CE objective; explicit best-score update |
| Final rank reduction / factorized execution | GNN `generate_result.py` reconstructs dense low-rank matrices; separate `model_lowrank.py` has factor modules; Flow `model.py:factorize_model` physically factors | `FactorizedLinear`, `model.export_model`, `evaluate.export_run`: physical factors mandatory |
| Resume/reproducibility | Not complete in supplied source loops | `io.py`, `TokenStream`, `train.py` checkpoint; new engineering |

Paths starting `tiny_clm/` resolve under `clm/`. Tests directly check detached projection,
rank/tail error, coupling gradients, stage direction, rectangular factor/bias behavior,
generation/cache compatibility, accounting and deterministic CPU restart.

## What changes and how to assess it

| Choice | Why | Test or ablation |
|---|---|---|
| Pretrained Qwen3 and CE replace random GNN/sum-rate or Flow/CFM | Preserve the accessible LM objective and avoid scratch pretraining | Original and matched dense controls; general NLL plus original CLM tasks |
| Per-matrix ranks set by storage budget | Heterogeneous LM matrices; rank fraction alone does not imply smaller storage | Whole-model accounting, noncompressive rank rejection, budget sweep |
| GNN persistent Adam/StepLR and held-out augmented selection | Match the original constrained implementation; source steps its scheduler per epoch, here per optimizer update with a stage-boundary decay | Restart tests verify moment step counters across stages; `flow_reset` is a separate Adam/cosine ablation, fixed across its controls |
| Fixed steps, gradient accumulation/checkpointing, FP32 masters and BF16 autocast | Feasible one-GPU memory and matched token budgets | Equal-token tests; compare future FP32/BF16 checks if numerical problems arise |
| Reduced full SVD replaces legacy full-size SVD storage | Same hard projection with smaller temporary factors | FP64 matrix equivalence tests; no randomized/approximate projection |
| Safetensors factors, custom loader, reference API | Make low rank physically deployable with native tokenizer/tools | No dense selected state keys; reload/logit tests; real tool-call formatting check |
| New absolute μ values | GNN/Flow losses and matrix counts have different scales | Initial actual residual audit; constant-μ controls with identical optimizer policy/stage counts |

An optimizer update with accumulation is treated as one effective minibatch, with proxies
held fixed until its update. The source refresh per minibatch is therefore preserved at
optimizer-update granularity; increasing `projection_every` is explicitly rejected.
The optional `ablation_constant_mu` freezes μ while retaining the same optimizer policy and number of
steps. It is labeled an ablation, not the default continuation algorithm.

## Layer and rank policy

Default S: all 28 Qwen3 MLP `gate_proj`, `up_proj`, `down_proj` matrices, 84 total.
The SiLU gate multiplication, residual paths, norms, attention layout and tokenizer remain
unchanged. Attention projections, or only MLP down projections, are explicit ablation
policies, not defaults. `layer_ids` and exact named `ranks` overrides are supported and
strictly validated. Fused, shared, hybrid and unknown modules are rejected.

For selected weight keep fraction k, `r = floor(k mn/(m+n))`, with strict checks
`0 < r <= min(m,n)` and `r(m+n) < mn`. Biases are retained once. This rule is a new
budget policy, not an inferred rule from the GNN paper. No validation-driven per-layer
rank search is used for the proposed method. The main sweep uses k = .8, .6, .4; pilot
uses .8 and .6. Any later validation rank search must be logged as additional tuning cost.

The actual Qwen3-0.6B architecture has 596,049,920 unique parameters. Its selected MLP
weights contain 264,241,152; the other 331,808,768 remain. All 84 selected matrices have
dimensions 3072×1024 or 1024×3072.

| Selected weight keep request | Rank | Selected factor parameters | Whole parameters | Whole reduction |
|---:|---:|---:|---:|---:|
| .8 | 614 | 211,255,296 | 543,064,064 | 8.8895% |
| .6 | 460 | 158,269,440 | 490,078,208 | 17.7790% |
| .4 | 307 | 105,627,648 | 437,436,416 | 26.6108% |

These are analytical counts verified against the loaded architecture, not successful
compressed-model results. BF16 dense raw weight storage is 1,192,099,840 bytes; the
factor budgets predict 1,086,128,128 / 980,156,416 / 874,872,832 bytes plus serialization
headers. `compression.json` records actual serialized bytes. Embedding/head ties count
once. Neither KV dimensions nor sequence lengths shrink from these weight replacements.

## Resource and penalty estimates

One GPU/process, microbatch 1, pilot accumulation 4 and main accumulation 8; effective
batch sizes are 4 and 8 blocks. Pilot 3×20 updates scores 122,640 training tokens per
trained run. Main 3×40 updates scores 490,560. The original and post-hoc controls have
zero fine-tuning tokens. ASVD calibration/search incurs separately reported work.

The mmap token sampler uses zero data-loader workers. OMP/MKL/OpenBLAS thread counts
are set from `SLURM_CPUS_PER_TASK`, avoiding an independent pool of loader processes.

For P=596,049,920 and selected S=264,241,152, FP32 weights + gradients + two Adam
moments require approximately `16P = 9.537 GB`; dense FP32 proxies add `4S = 1.057 GB`.
Thus persistent training tensors alone approach **9.87 GiB**. Add the differentiable
penalty temporaries, logits, checkpointed activations, Adam workspaces, CUDA allocator
reserve and one sequential SVD workspace. A 24 GiB BF16-capable GPU is a provisional
floor; 32 GiB or more gives safer headroom. Host RAM is requested at 64G, including
best-model CPU copies (~2.38 GB), checkpoint staging and asset loading. Actual peak GPU
memory is unmeasured. These estimates do not justify a claim that 24 GiB is sufficient.

Mid-stage atomic checkpoint writes can approach 10.6 GB (dense model, Adam, best model,
proxies); the old and temporary new checkpoint coexist during replacement. At intermediate stage boundaries, GNN mode retains optimizer state and drops only the best-model copy; Flow mode drops both. At final completion both are dropped. A completed constrained checkpoint is about
3.44 GB before serialization overhead. Reserve at least 250 GB for the pilot and 500 GB
for the main grid plus logs/containers/caches, then check quotas; these are planning
allowances, not measured final usage. Concurrent checkpoint writers require additional
space and can saturate shared storage.

Exact reduced SVD costs O(mn min(m,n)) each time, independently of the number of retained
singular vectors. The full initial 84-matrix CPU audit took 71.53 seconds and peak process
RSS was 4.84 GB in this environment; it did not retain all proxies or backpropagate.
It is not a GPU runtime estimate. Training logs split projection/update/total costs;
total time also includes validation, checkpointing and initialization. ASVD sensitivity
search requires up to 84×6×16 calibration forward passes plus decompositions, a sizable
cost despite zero recovery-training tokens. Use pilot timings and `sacct` to choose
wall time, not a presumed speedup.

Actual initial squared-tail sums were 29,423.56 / 53,963.97 / 88,925.74 for k=.8/.6/.4.
At μ=10,000 these imply penalties 1.471 / 2.698 / 4.446, compared with preliminary
uncompressed NLL 2.834 on only two validation blocks. The configured sequence
10,000→1,000→100 is **provisional and increasingly strong**, not copied numerically
from either earlier task. Its final counterfactual penalty at the initial weights would
be 147–445; useful training must reduce the residual without destroying task capability.
Failure of that balance is a valid negative result.

The export projection is computed from the actual FP32 trained selected weights, then
the factors are rounded to the requested deployment precision. Dense pre-export and
factorized post-export evaluation both execute at that same precision. Export temporarily
keeps a CPU copy of selected FP32 weights (~1.06 GB here) to avoid projecting already-rounded
BF16 weights; a dedicated test checks this ordering.

## Deployment and theoretical limit

Export uses `A=U_r diag(σ_r)` and `B=Vh_r`, then computes `A(Bx)+bias` with two kernels.
Factor scaling differs from symmetric square-root splitting but represents the same
rank-r matrix in exact arithmetic; finite-precision agreement is tested. No compressed
backend silently reconstructs selected dense matrices. The reference load temporarily
instantiates a dense architecture on the host before replacing selected modules; this
can affect loading peak RSS, not steady GPU parameter residency.

For a single linear map and the same input, the elementary bound
`||(W−AB)x||₂ <= ||W−AB||₂ ||x||₂` holds. It does not establish an end-to-end logit,
loss or trajectory guarantee without further assumptions about all intervening maps,
inputs and decoding. No GNN/Flow minimizer-limit claim is transferred to finite-step
nonconvex LM training, and no weight bound is asserted to preserve context management.

### Optimizer policy and provenance

The earlier branch used Flow-style optimizer resets as its default. This revision makes
`training.optimizer_policy="gnn_persistent"` explicit in both released configurations:
Adam moments persist after restoring best stage weights, and StepLR multiplies the
learning rate by sqrt(gamma) at each stage boundary. This matches the supplied GNN code,
including its decision not to roll back moments with the best weights. In the GNN code a
scheduler step represents an epoch; here it represents an optimizer update. The stage
boundary, not the absolute number of GNN epochs, defines the decay timing. The source
learning rate and mu values are not presumed transferable.

`flow_reset` remains an optional, labeled transfer ablation with new Adam and cosine
annealing per stage. Do not pool its outputs with the GNN policy. Checkpoint resume
rejects a changed configuration/policy. Both policies have exact CPU restart tests.

The small update budgets are feasibility screens, not established sufficient recovery.
In particular, strong final coupling with only 20 updates per stage may leave a large
deployment gap. Inspect residual decrease and validation loss before selecting a longer
budget; retain failed pilot outputs and count any extra tuning/recovery tokens. A failed
short pilot does not establish an inherent limitation of the constrained formulation.
