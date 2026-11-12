# Qwen3 export precision correction

Follow-up: the 4B GPU reload subsequently passed with zero logit error and
identical parameter/buffer fingerprints. A separate native-model cache audit
motivated the explicitly versioned screen criterion described in
[NATIVE_CACHE_SCREEN.md](NATIVE_CACHE_SCREEN.md). The original cache tolerance
below is retained as a diagnostic there; it is no longer an export gate.

The v1 deployment path called `model.to(dtype=torch.bfloat16)` on the entire
model. This casts floating-point buffers as well as parameters. In Qwen3 it
rounds the nonpersistent `model.rotary_emb.inv_freq` buffer, which native
Transformers construction keeps in FP32. Casting the rounded buffer back to
FP32 inside the rotary forward pass cannot recover the lost values. Identical
serialized parameter bytes therefore did not guarantee identical inference.

Source references:

- [PyTorch 2.8 Module.to](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Module.html#torch.nn.Module.to)
- [Transformers 4.57.1 Qwen3RotaryEmbedding](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3/modeling_qwen3.py)
- [Pinned Qwen3-4B-Instruct-2507 configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)

## Correction and compatibility

`tiny_clm.model.deployment_precision` preserves the original FP32 frequencies
while converting parameter dtype and device. It checks the buffer against the
official static RoPE initialization and rejects already-rounded values,
including values that were rounded and subsequently upcast. Scaled or dynamic
RoPE requires a separate audit and is rejected. Both currently configured
checkpoints use the supported static/default RoPE.

New artifacts have format `tiny-clm-factorized-v2` and inference runtime
`qwen3-fp32-rope-v2`. The v1 loader explicitly retains historical behavior and
emits a warning; existing artifacts and completed campaigns are not rewritten.
Evaluation, profiling, deployment-gap and ContextBench outputs record the
runtime. Aggregation rejects mixed policies, and earlier baseline/pilot gates
cannot authorize corrected experiments. The server health response also
identifies the runtime, since equal weight checksums alone cannot do so.

The screen now compares unique parameter bytes **and nonpersistent buffers**
before and after export. It saves audit diagnostics before raising comparison
failures. Its additional controlled forward pass rounds only the RoPE buffer
and restores it immediately, to measure this mechanism on the actual checkpoint.
The deployed artifact and subsequent cached execution retain FP32 frequencies.
The existing export tolerance (`rtol=atol=1e-4`) and cached-query tolerance
(`rtol=0.02`, `atol=0.25`) are unchanged.

No constrained-training objective, rank policy, optimizer, penalty schedule,
prompt, tool parser, or task-quality threshold was changed.

## Executed CPU evidence

A random Qwen3 fixture with 361,600 parameters, two layers, head dimension 128,
RoPE theta 5,000,000, BF16 weights and positions 4096 through 4159 reproduced the
failure mechanism under PyTorch 2.8.0+cpu and Transformers 4.57.1:

| Quantity | Measured value |
|---|---:|
| Parameter bytes changed by the old whole-model cast | No |
| Maximum rotary-frequency absolute error | 0.0008976161479949951 |
| Maximum logit absolute error after that cast | 0.0091552734375 |
| Maximum logit absolute error after restoring only the FP32 buffer | 0.0 |

The corrected full test suite ran from a read-only Git archive without a `.git`
directory: **97 passed, 1 skipped**. The skipped test requires the complete CLM
SDK environment; it was not claimed as executed in this CPU environment. The
separate random-data CPU smoke pipeline also passed. Regression coverage includes
dense and factorized FP32/BF16 round trips, tied parameters, generation, cached
execution, invalid precision states, legacy loading and incompatible result gates.

Commands, with the project environment activated and source root as the working
directory:

```bash
PYTHONPATH=clm HF_HUB_OFFLINE=1 python -m pytest -q \
  compression/tests clm/tests/test_context_string.py -p no:cacheprovider -ra
PYTHONPATH=clm HF_HUB_OFFLINE=1 python -m tiny_clm.smoke \
  --device cpu --output /tmp/tiny-clm-precision-smoke
```

The smoke output directory must be new. These tests use random weights/tokens;
they establish software behavior, not language or context-management quality.
The corrected real 4B CUDA reload subsequently passed in user-supplied output;
see the follow-up above for the cache audit and revised screen. The development
environment remains CPU-only, and no 4B ContextBench capability result is claimed.

## Interpretation of earlier results

The earlier 0.6B baseline cast the model before both sides of its reload check.
Thus its zero reload error did not establish agreement with native checkpoint
inference. Its quality and profiling measurements describe the legacy runtime;
they cannot establish the capabilities or efficiency of the corrected runtime.
Preserve those results, label them as legacy, and rerun a corrected baseline
before using them in a compression comparison. No claim is made that correcting
RoPE will by itself make either backbone pass the CLM capability gate.

Use a new campaign for the retry. Existing pinned model downloads and task
container can be reused after their normal integrity checks. A screen pass
permits a full baseline and training-memory audit, not a compression sweep.
