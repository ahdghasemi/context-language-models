# Diagnose cached versus full-sequence differences

The corrected Qwen3-4B BF16 export has now been observed to preserve parameter
and FP32 RoPE-buffer fingerprints with **zero reload logit error** on an A100.
Rounding only the rotary buffer reproduced a maximum logit error of 0.5.
The subsequent cached-versus-full-sequence test still failed: maximum absolute
error 0.5625, with 103 of 151,936 logits outside its existing tolerance. These are
user-supplied execution observations. They are not task-quality measurements.

The cache failure alone does not distinguish a cache-handling defect from
numerical variation in the native BF16 implementation. A full-sequence call
and a cached single-token call have different matrix/kernel shapes. PyTorch
[does not guarantee identical results for mathematically equivalent batched
and sliced operations](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html).
Transformers [documents the attention-mask and cache-position requirements
for custom cache loops](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation).
Neither statement establishes that this particular discrepancy is harmless.

## One diagnostic job

`compression/cache_audit.py` reuses an existing verified v2 export and its
pinned native checkpoint. It creates a new immutable source snapshot and
separate output directory, verifies the installed environment, strips inherited
Slurm resource overrides, and submits one GPU job. The existing screen,
checkpoint, cache files and gate are never modified. There are no downloads,
container builds, training runs, or ContextBench episodes in this diagnostic.

With the usual project environment and verified scheduler settings loaded:

```bash
# FAILED_SCREEN is the existing screen campaign directory.
# AUDIT_CAMPAIGN must be a new directory under the configured results parent.
python compression/cache_audit.py \
  --source-manifest "$FAILED_SCREEN/manifest.json" --campaign "$AUDIT_CAMPAIGN"
python compression/cache_audit.py \
  --source-manifest "$FAILED_SCREEN/manifest.json" --campaign "$AUDIT_CAMPAIGN" --submit
```

Default request: one GPU, four CPUs, 64 GiB requested host memory, one-hour
wall time. `TINY_CLM_CACHE_AUDIT_CPUS`, `_MEM`, and `_TIME` override those three
values. The GPU partition/constraint/account/QoS use the existing configuration.
FP32 model parameters alone occupy about 15 GiB; the GPU check requires at
least 32 GiB. Native and exported models are loaded sequentially. This is an
inference diagnostic, not an estimate of compression-training memory.

`cache-audit.json` records incremental diagnostics and errors. It compares:

1. Native versus exported full-sequence, implicit-cache, and explicit-cache
   outputs at the unchanged export tolerance `rtol=atol=1e-4`.
2. Cached versus full-sequence outputs **within each model**, retaining the
   existing screen tolerance `rtol=0.02, atol=0.25` as a reported measurement.
3. Implicit positions versus explicit cache positions and full attention masks.
4. Default BF16 SDPA, forced math SDPA with BF16 weights, and FP32 math with
   those same BF16-rounded weights. TF32 is disabled. Alternative modes are
   diagnostics only and do not change the serving configuration.
5. Sixteen-token greedy generations with and without caching, including
   native/export token equality separately for each generation path.

The original screen prompt is tested with a one-token and an eight-token
teacher-forced suffix. A synthetic sequence formed by repeating/truncating its
token IDs to 2048 tests eight cache updates. Higher-precision/backend controls
use the short prompt. The full logits projection is retained to match the
original screen's computation shapes. Reports include maximum error, RMSE,
argmax agreement, top-token margin, KL divergence and total variation. Token
agreement on this bounded probe is not evidence of long-horizon CLM retention.

The report cannot authorize a screen or compression run. Its completed status
means the measurements finished, not that cached and uncached computations
passed the existing screen threshold. A failed native/export comparison exits
nonzero after preserving the diagnostic. No acceptance threshold was changed.

## Verification boundary

The new CPU tests cover single/multiple cache updates, explicit positions,
FP32/BF16 native/export agreement, nonfinite results, strict export comparisons,
source checksum failure, and a real archived snapshot with a mocked scheduler.
The full suite ran from a read-only archive without Git metadata: **106 passed,
1 SDK-dependent test skipped**. Actual 4B GPU execution of this diagnostic
remains pending. Its output must be
inspected before deciding whether to fix cache handling, change a numerical
backend, or revise the cache criterion with an explicit native-model control.
