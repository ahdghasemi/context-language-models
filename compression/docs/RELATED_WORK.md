# Focused related-work and novelty assessment

Primary sources checked on 2026-11-11; this is a focused comparison, not an exhaustive
survey or a priority claim.

* **ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language
  Models**, Yuan et al., [paper v5](https://arxiv.org/abs/2312.05821v5),
  [official code](https://github.com/hahnyuan/ASVD4LLM). It combines activation-dependent
  scaling with layer sensitivity allocation; its KV-cache extension is separate from
  weight compression. Source revision `863042743edb4c0dcc512a45534dea07df16b140` was
  inspected, specifically `act_aware_utils.py`, `modules/svd_linear.py`, `sensitivity.py`
  and `binary_search.py`. Our separate baseline retains channel scaling with alpha=.5
  and sensitivity-ranked budget allocation, using exact SVD, actual parameter counts,
  fail-fast errors and the audited Qwen3 targets. These choices and limited training-text
  calibration differ from an official paper reproduction. Both the uniform-rank and
  adaptive variants are implemented; only the latter tests rank sensitivity allocation.

* **SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model
  Compression**, Wang et al., ICLR 2025,
  [paper](https://arxiv.org/abs/2403.07378),
  [official repository](https://github.com/AIoT-MLSys-Lab/SVD-LLM). It uses data whitening
  and sequential low-rank updates to address truncation loss. This reinforces the need
  for an activation-aware baseline and recovery controls: plain SVD alone is weak evidence
  of an advance. SVD-LLM itself is not integrated or evaluated here; its architecture/export
  path needs a separate Qwen3 audit and matching budget/protocol. That is an implementation
  scope choice, not a claimed lack of publicly available code.

* **SVD-LLM V2**, Wang et al., NAACL 2025,
  [paper](https://arxiv.org/abs/2503.12340), same official SVD-LLM repository. It combines
  nonuniform allocation based on theoretical truncation loss with loss-optimized truncation.
  This is a stronger publication-level comparator than raw SVD; it has not been integrated
  into this Qwen3/CLM protocol, and its results must not be claimed as reproduced.

* **LEMS / KFAC-SVD**, Thoma et al., 2026,
  [author project](https://lems-svd.github.io/). The current author source describes global
  rank allocation with a calibrated error model and token-wise Fisher approximations,
  including Qwen3 experiments. Its date/version and reported gains differ between indexed
  manuscript and project pages, so no exact performance number is imported here. No LEMS
  code or results were run. The initial implemented ASVD sensitivity control is reproducible,
  but is insufficient by itself to establish superiority over the latest rank-allocation
  work; adding a pinned official LEMS or SVD-LLM V2 control is required before such a claim.

* **LowRankArena: A Standardized Evaluation Platform for SVD-Based LLM Compression**,
  Shao et al., August 2026,
  [paper](https://arxiv.org/abs/2608.26389),
  [official repository](https://github.com/Zishan-Shao/lowrankarena). Its aligned comparisons
  emphasize task-version/precision/budget consistency and show that apparent method rankings
  and end-to-end gains depend on the setup. This motivates actual whole-model accounting,
  matched recovery, raw runtime measurements and explicit backend limits here. Its released
  results are not our measurements, and its vLLM path does not establish support for our
  custom factor modules. No LowRankArena experiment was run in this task.

* **Context Language Models**, supplied manuscripts and
  [official repository](https://github.com/facebookresearch/context-language-models),
  plus [ContextBench](https://github.com/RulinShao/ContextBench). Context editing and suffix
  cache reuse change system computation and memory access; they do not by themselves
  reduce backbone parameter storage. We preserve the harness and isolate the backbone
  experiment. SCR is not combined with compression in the primary results.

The hard-rank auxiliary formulation, inverse-μ quadratic penalty, alternating projection
and continuation are inherited from the user's GNN algorithm. Transfer to another neural
objective and factorized execution were already present in the supplied Flow adaptation.
Truncated matrix approximation, two-linear deployment, causal-LM fine-tuning, activation
calibration, Slurm orchestration and restart bookkeeping are established techniques.
Applying these to CLM alone does **not** establish algorithmic novelty.

A scientifically compelling contribution would require evidence that the specific
constrained procedure improves a matched size–quality–latency frontier over properly tuned
controls, and that its effect on long-horizon context editing differs meaningfully from
generic backbone degradation. A measured failure boundary, deployment-gap analysis or
demonstrated interaction with context-state errors could also be useful. Those are proposed
empirical contributions, not results obtained so far. The small model may simply lack
sufficient dense CLM competence; the reference backend may turn lower FLOPs into slower
execution. Both outcomes must remain visible.

No theorem from beamforming or Flow matching is asserted for the LM setting. The only
matrix-level inequality in the specification is elementary and explicitly conditional on
the same input; it proves neither task accuracy nor long-trajectory robustness.
