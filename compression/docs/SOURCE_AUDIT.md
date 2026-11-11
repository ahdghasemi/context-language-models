# Source audit and feasibility

Audit date: 2026-11-11. Implementation baseline: user fork
`ahdghasemi/context-language-models`, default branch `main`, starting commit
`483198229ab7b90c0eed4651306ab3b3505aaf50`. The upstream HEAD and fork HEAD matched at
inspection; no material fork divergence affected the implementation. The new feature
branch is `tiny-clm-constrained`. Upstream files are preserved except the README link,
optional dependencies/package registration, and ignore rules.

## Evidence inspected

The attached PDFs were extracted to text; the GNN algorithm page was also rendered and
visually checked. LaTeX and the nested code folders were inspected directly, including
equations, updates, export, model definitions and submission scripts. The Regularized
archive was not used to implement the proposed method.

| Artifact | Relevant source locations | SHA-256 of supplied attachment |
|---|---|---|
| `IEEE_TSP.pdf` | Section III-A, Eqs. (7)–(14), Algorithm 1; complexity discussion | `a1c47bf1c46954c931c6851c6f1bee89be0294c5a997ae1e3f038a97c9b2bf63` |
| `IEEE_TSP.zip` | `bare_jrnl_new_sample4_Ahmad_v3.tex`; nested `code/Constrained.zip` | `cba657d905f91ead5acba2c3fafd67dd732705b0efbb098c6a106e760795e4d3` |
| `CLM_arXiv.pdf` | Context-as-file, zero-shot/RL, SCR, evaluation and appendix settings | `ac5c926814434d8ad2f499183cd2a81e53b4c14f327587f99980ade9157db01b` |
| `22968_Context_Language_Models.pdf` | Alternate supplied CLM manuscript; architecture/system and evaluation | `c02490993efc92c5b64aeeb6e473ce2906dd763e172004b39dd6217d3336b685` |
| `Tiny_Flow_Matching_For_MLSP__Copy_clean_for_slide_prepration_(2).pdf` | Constrained OT-CFM and factor deployment | `6c8a2a2d4fabf613474225de86bf0ebb8c2dd647f87a672cb9d1722a4bff6dbf` |
| `Tiny_Flow_Matching_For_MLSP__Copy_clean_for_slide_prepration_(1).zip` | `mlsp_template_camera_ready_v7.tex`; `code/tiny_flow_matching_repro_final_v1.zip` | `71c7adaedb7c310a5513946bcd91ddf9b1def2cbd20f5251d5bd981b68ef9660` |

GNN code below means the extracted `Constrained/` directory. Flow code means
`tiny_flow_matching_repro/` inside its nested archive. These paths are source references,
not files redistributed in this fork.

## The original method, including discrepancies

The paper is titled **Constrained Optimization for Low-rank Training of Graph Neural
Networks**. Its method is called the **Constrained formulation / Constrained Problem**;
Algorithm 1 is **Gradual low-rank GNN training**. No new acronym is assigned here.

| Finding | Paper/code evidence | Implemented interpretation and consequence |
|---|---|---|
| Rank indicator, not nuclear regularization | GNN Eq. (11); `utils.py:238–273` | Exact hard truncation to rank r. Class name `ProximalOperatorNuclearNorm` and its lambda/nuclear-norm docstrings are misleading. |
| Penalty coefficient is 1/(2μ) | GNN Eq. (11); `utils.py:272`; Flow `src/tinyfm/train.py:54` | Decrease μ to increase coupling. No normalization by layer size or parameter count. |
| Algorithm loop condition is reversed | PDF Algorithm 1 line 5 / LaTeX `while μ <= μ_min`; GNN `main_train_gradual_part1.py:103` uses `>=` | Follow equations, prose and executable `>=` condition. Literal pseudocode would skip normal μ_init > μ_min runs. |
| Best-loss pseudocode omits updating its minimum | Algorithm 1 initializes L_min but does not assign the new L_min | Follow executable best-score tracking. Otherwise the selected iterate would not actually be the best. |
| Abstract subproblem argmin is not an exact neural minimizer | Algorithm 1 vs `utils.py:183–204` | One stochastic Adam update per newly projected proxy, as executed in source. No convergence-to-global-optimum claim. |
| Stage selection differs between transfers | GNN `test_proximal` returns task and task+penalty, latter selected in driver; Flow selects sampled training CFM loss | Use held-out **validation** NLL+penalty for constrained stage selection. The GNN variable called `test_loader` is not a license to tune our final test set. |
| Optimizer history differs | GNN Adam/StepLR persist through best-weight restoration; Flow Adam/cosine reset every stage | Use the explicit Flow reset convention; document as a transfer choice. Restored best weights start a fresh stage optimizer. |
| Flow selection timing | Flow `train.py:47–58` compares pre-update sampled CFM loss and saves post-update weights | Do not copy this noisy/misaligned selection. Evaluate the exact saved post-update state on held-out blocks. |
| Stage rounding can exceed nominal epochs | GNN logarithmic ceil count; Flow ceil(total/stages) | Configure a fixed number of optimizer updates per stage; totals are explicit and matched across trained controls. |
| Truncation does not make full SVD cheap | GNN `torch.svd(..., some=False)`; Flow reduced full SVD | New exact reduced full SVD costs O(mn min(m,n)), not O(mnr). GPU workspace/runtime must be measured. |
| GNN evaluation rank reduction is still dense storage | `generate_result.py:102–105` calls `utils.py:reduce_rank`, which reconstructs UΣVᵀ into `layer.weight` | Those quality evaluations do not demonstrate a smaller deployed artifact. The separate `model_lowrank.py` has factor modules, but that is not the cited constrained-result evaluation path. This transfer uses physical factors, as also implemented in Flow, and explicitly measures the deployment gap. |

GNN initializes its dense neural model randomly for beamforming. Its graph construction,
message passing, weighted sum-rate loss, channel generation and wireless evaluation do
not transfer. Flow uses its OT-CFM loss/data and optionally warm-starts from dense
weights; its coverage and transport results do not predict language behavior. The Flow
configuration explicitly labels several settings as **reimplementation defaults** and
asks that archived run settings replace them if different. It is useful source code,
not proof of bitwise reproduction of the reported Flow experiments.

The GNN archive contains no Unity shell/Slurm scripts found in the Constrained code.
Flow's `scripts/setup_unity.sh`, `scripts/slurm/submit_all_unity.sh` and array scripts
use configurable project/data roots, one GPU, 4 CPUs, 32G and 24 hours, without a fixed
partition/account. The new workflow reuses configurable `/work` storage and arrays;
it does not inherit those resource requests as validated LM requirements.

## What CLM actually contains

| Component | Repository evidence | Compression relevance |
|---|---|---|
| Neural backbone, supplied via model endpoint | `clm/clm_harness/clm_agent/harness.py` model/query path | Accessible open weights can be compressed. API-only weights cannot. |
| Context editing execution framework | `ClmAgent`, `context_env/env.py`, `context_utils/context_string.py` | Python control logic; no trainable weight matrix to factorize. |
| Live context file and external files | Mirrored `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`, bash tools and sandbox state | External state is not model weight storage. Token reduction is not parameter compression. |
| CLM-specific ContextBench adapter | `clm/clm_contextbench/clm_stream.py:ClmStream` | Reused unchanged for experimental tasks and original skill text. |
| RL training | `clm/clm_rl/README.md`, dual-channel advantage patches | Full trainer is explicitly absent; requires external ProRL-Agent-Server/Slime. |
| Prefix/suffix cache mechanisms | Paper SCR section and serving extensions | Distinct inference mechanism. Disabled in this reference comparison. |

The paper's RL base is **Qwen3.5-9B** with OpenResearcher `rl3040` prompts. Its inspected
public model config uses a hybrid linear/full-attention architecture and a vision
component. It is not safely handled by blindly substituting Qwen3 linears. The paper's
zero-shot models include **Qwen3.6-27B** and closed endpoints. Neither the supplied
attachments nor the inspected fork supplied a linked trained CLM-RL checkpoint.
This is an accessibility finding about these artifacts, not a claim that no such
checkpoint exists anywhere. The RL result and closed-backbone results cannot currently
be reproduced by this package.

The pilot instead uses `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`, a **general post-trained** Qwen3 model with
thinking/tool/chat support, not a CLM-trained model and not a base pretraining checkpoint.
The actual weights and tokenizer were downloaded and loaded. Config and installed
Transformers implementations (`models/qwen3/modeling_qwen3.py`) were inspected:
28 layers; hidden size 1024; MLP width 3072; vocabulary 151936; tied embedding/output;
16 query heads, 8 KV heads, head dimension 128. In particular the query projection
width is **2048**, not 1024. MLP gate/up/down are separate bias-free matrices; Q/K
normalization and residual dimensions must remain intact. Selection rejects all
non-Qwen3 architectures, shared target weights and unknown layer names.

`Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3` supplies general causal-LM fine-tuning data.
It does not teach context editing. Native train/validation/test splits are retained;
fixed caps are packed into EOS-separated 512-token blocks. ContextBench comes from
`RulinShao/ContextBench` at `fb533c18c78a82d3f1b2047a9c1700735cf829ae`, inspected through
task packaging, streaming agent, skills and graders. Task files/container hashes are
recorded. No task answers enter training. Public default assets required no credential
during the local downloads; a changed gated model requires your own authorized HF token.

## Feasibility verdict

**A faithful constrained backbone-compression experiment is feasible. A reproduction of
the original RL-trained CLM is not currently justified.** The code now provides the
former, with the original context-management interface. Preliminary real-model language
generation and tool-call formatting worked; long-horizon context management remains
unvalidated. If the small dense model cannot handle the pilot CLM tasks, increasing the
backbone capability or obtaining the original trained checkpoint is required before
claiming CLM retention. That failure must not be hidden by reporting only perplexity.

Missing/unavailable here: a supplied trained CLM-RL checkpoint, full RL training setup,
authoritative GNN/Flow archived experiment logs, Unity credentials/association/GPU
allocation, a local CUDA device, and a working local Apptainer/Singularity runtime.
These block particular reproductions/measurements, not the source-grounded implementation.
