# Unity validation from user-supplied terminal output

The initial observations were supplied by the user while running the delivered commit
`10b85e71a6e52eb6cbfd59002f68a0f3c5db826a` on Unity. Follow-up diagnostics are identified
below. They were not executed remotely
by the assistant. They extend the earlier local-only execution report.

* Allocation: one NVIDIA A100-SXM4-80GB, four CPUs, 32 GB host RAM, two-hour limit.
  The reported partition was `gpu-highvram`, QoS `short`, with feature `a100-80g`.
  These observations do not establish access to longer QoS limits or other partitions.
* Environment: Python 3.12.3, PyTorch 2.8.0+cu128, CUDA runtime 12.8, NVIDIA driver
  595.91.07, Apptainer 1.5.5, and `/usr/bin/singularity` available.
* GPU environment check passed: one visible GPU, BF16 support, memory threshold,
  CUDA matrix multiplication/backpropagation, and exact FP32 SVD.
* `pip check` reported no broken requirements. The initially dirty tree contained
  only generated `build/`; excluding it produced a clean status. `/build/` is now
  ignored in the repository as well.
* Correctness: **43 passed**, two deprecation warnings, 31.33 seconds.
* CUDA smoke: **passed**, six optimizer updates; 20,672 original versus 14,144 deployed
  parameters; random-data validation NLL 4.1599772572517395. This uses random weights
  and data in FP32 and is not evidence of real-model compression quality.
* ContextBench clone at the pinned revision succeeded. Container construction failed
  before `%post` could run: `mount /nas->/nas ... destination /nas doesn't exist`.
  No successful task lock or CLM episode followed in the supplied output.

## Fix and its validation boundary

The first fix, `38cb910ac0f9bdf8f81fcfdae58f13fb19af9746`, precreated `/nas` in `%setup`.
The user's subsequent log confirmed that setup ran successfully, but the build then
failed at another bind destination, `/work`. That patch was insufficient.

The user's full mount diagnostic identified the source:

```text
APPTAINER_BINDPATH=/work,/nese,/project,/gypsum,/scratch,/modules,/nas,/datasets,/scratch4,/scratch3,/rstor
APPTAINER_BIND and APPTAINER_MOUNT: unset
All SINGULARITY_BIND/BINDPATH/MOUNT variables: unset
mount hostfs = no
bind path = /etc/localtime
bind path = /etc/hosts
```

Commit `4fb9e08e220e6625db5e516d5cd5c1f119f1d363` removes the directory workaround and instead clears inherited bind
and mount variables in the environment passed to container subprocesses, including
Harbor. Explicit command-line mounts and the task grader remain unchanged. Relevant
primary sources inspected were the Apptainer 1.5 user guide and its exact 1.5.5 source:

* https://apptainer.org/docs/user/1.5/build_a_container.html#bind
* https://github.com/apptainer/apptainer/blob/v1.5.5/internal/pkg/build/stage.go
* https://github.com/apptainer/apptainer/blob/v1.5.5/internal/pkg/build/util.go

Two local regression tests passed: an actual spawned child receives none of the six
bind/mount variables while keeping cache/temp/Slurm/CUDA settings and leaving the parent
unchanged; an explicit input environment is copied without adding ambient defaults.
Python compilation, recipe shell parsing and diff whitespace checks also passed.
The full ML suite was not rerun in the replacement local runtime, which lacks PyTorch.
No Apptainer runtime is available there. New total test count is 45; only the new two
tests were executed locally for that fix, and the previous 43 passed on Unity as noted
above.

## Follow-up: mount fix passed; host fakeroot library failed

The user checked out `4fb9e08e220e6625db5e516d5cd5c1f119f1d363`, explicitly loaded its
Python source, and ran `pytest -q compression/tests/test_container_env.py`: **2 passed
in 0.12 seconds**. The subsequent build reached `%post` without the missing-bind errors,
then failed before any recipe command could run:

```text
/bin/sh: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found
(required by /.singularity.d/libs/libfakeroot.so)
```

This is an independent compatibility problem. The log reports no subuid mapping and
use of the host fakeroot command. Apptainer explicitly documents the GLIBC mismatch
when the host fakeroot library needs symbols absent from the container. Its source also
uses the host fakeroot command for action execution in this mode, so a build-only
`--ignore-fakeroot-command` flag is insufficient for our runtime path. Official sources:

* https://apptainer.org/docs/user/1.5/fakeroot.html
* https://github.com/apptainer/apptainer/blob/v1.5.5/internal/pkg/runtime/launch/launcher_linux.go

### Revised base and independently checked provenance

Registry metadata and the replacement amd64 base layer were retrieved on 2026-11-12.
The compressed layer SHA-256 was checked before reading `/var/lib/dpkg/status`.
No container process was executed for this inspection.

| Property | Original | Replacement |
| --- | --- | --- |
| Official Python image | Upstream digest below | `python:3.13.15-slim-trixie` resolved to digest below |
| Python (OCI config) | 3.13.15 | 3.13.15 |
| Debian suite (OCI history) | bookworm | trixie |
| libc6 package inspected in replacement base | Not re-inspected | 2.41-12+deb13u4, amd64 |

Original index: `sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e`.
Original amd64 manifest: `sha256:2f2e5a876c71a6757f55ec57f2add0225ddaf01c802a33fcc29073943f94d907`.
Replacement index (used in the recipe):
`sha256:7c61056e61ac89e852de05f3dc6fa51a6dd2181797bceed46aa725dd7cb2cd3b`.
Replacement amd64 manifest:
`sha256:37134a49d21d2120e4c4d73bb76f8a4ab9aef31f096f7ec2ead48c2feead4332`.
Inspected base layer:
`sha256:6b37362b3da78869050b894b799ad4df04f1f3b52774087db0d81151570244c8`.

The Python source SHA in both image configurations is
`1e66a7945a48390ee4c2a4268a0e4185884059a13c4aab6d148aa208deea4a76`.
The image change is necessary for the observed host-library requirement and retains
the task Python version, but is not an exact reproduction of the upstream OS image.
Every method and both harness modes must use the same locked image. Task/grade code
is unchanged; baseline episodes are still required to validate task behavior.

The updated `%test` stops at any failure and prints libc; the Harbor-flag runtime
preflight also prints Python/libc versions. Local checks for this revision: both
environment regression tests passed again; Python compilation, recipe shell parsing
and diff whitespace checks passed. No full ML suite, Apptainer build, Harbor runtime
or Unity submission was executed by the assistant for this revision. A user retry must
finish both build and runtime preflight before reporting container preparation success.

## Subsequent Unity preparation and baseline submission

User output at `3c1195158bd00f3100c7fbd5742b94a38d11e372` confirms the complete image
build, `%test` (GLIBC 2.41, Python 3.13.15), and runtime preflight passed. Four runnable
tasks were packaged: kv-store R130/R1000, each with seeds 1101 and 1102.
The resulting SIF SHA-256 is
`382f831f3ec30ac7c893960dfbd855d6f87e7df219c51fa9a91d23cdb7e5a9cd`.
The separate baseline-input check verified the task/image/recipe hashes and model/data
hashes. Model: `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`; dataset: `Salesforce/wikitext` at
`b08601e04326c79dfdd32d625aee71d232d685c3`. These checks did not evaluate model quality.

Scheduler queries showed account `pi_aghasemi_umass_edu`, allowed QoS
`long,normal,short`, short limit four hours, long limit five days, and five-day limits
on both `cpu` and `gpu-highvram`. These are observations, not permanent cluster policy.
The user submitted `baseline-001` from a GPU VS Code allocation, using the above source
commit, one GPU/four CPUs/32 GB for GPU stages and two CPUs/8 GB for the small CPU stages.

| Stage | Job | Observed outcome |
| --- | --- | --- |
| Environment | 1467312 | Completed, exit 0:0, 24 seconds, uri-gpu001 |
| Correctness | 1467313 | Failed, exit 1:0, 2 seconds, uri-cpu074 |
| Smoke | 1467314 | Cancelled before execution |
| Original checkpoint / export / evaluate / profile / ContextBench | 1467315–1467319 | Cancelled before execution |
| Aggregate | 1467320 | Failed, exit 1:0, 1 second, uri-cpu074 |

Both failing stderr logs contain only the srun step-creation error
`Invalid generic resource (gres) specification`; their stdout is empty. Thus neither
correctness Python code nor aggregation ran. The model stages were dependency-cancelled.
This is infrastructure failure, not failed ML correctness or poor model quality.

The scripts previously exported the parent allocation's environment and left srun GPU
resource options implicit. Slurm documents `SLURM_GPUS`, `SLURM_GPUS_PER_NODE`,
`SLURM_GRES`, `SLURM_STEP_GRES`, `SLURM_TRES_PER_TASK` and related variables as input
options. Inheritance from the GPU submitting session is the working diagnosis; the
exact offending variable was not logged, so it is not claimed as directly observed.

The repair filters parent allocation/option variables at all sbatch submission sites,
preserves Slurm client connection settings and project paths, and makes CPU/GPU step
requests explicit. Step filtering preserves the NEW allocation's IDs and device mask.
The manifest records the names removed at submission. `baseline-001` and its immutable
snapshot remain unchanged; validation must use a new campaign and code snapshot.

Local validation: seven stdlib environment tests passed, including actual subprocess
boundaries and an executable stand-in for srun covering CPU and GPU requests. Those
tests do not invoke a Slurm scheduler. The combined targeted pytest run passed **14
tests**, with **2 deselected** because their aggregation paths import unavailable
PyTorch. It includes real git snapshot creation with mocked sbatch, dependency checks,
and a mocked five-stage resume checking environment filtering and its resume flag.
Both batch scripts passed shell syntax checks; Python compilation and diff whitespace
checks passed. The full ML suite and repaired Unity submission remain unrun
by the assistant. Constrained training on the real checkpoint, baseline CLM competence,
GPU profiling, and a complete Slurm campaign remain unverified.

## baseline-002: launch succeeded, environment fingerprint failed

The user fetched `132d15d15432bf6e795e7dfca229c250addb3475` and ran the three targeted
test files: **16 passed in 6.99 seconds**. Nine new jobs were submitted. Subsequent
accounting and logs showed:

| Stage | Job | Observed outcome |
| --- | --- | --- |
| Environment | 1469298 | Failed, exit 1:0, 4 seconds, uri-gpu001 |
| Correctness / smoke / model stages | 1469299–1469305 | Cancelled before execution |
| Aggregate | 1469306 | Failed, exit 1:0, 5 seconds, gypsum-gpu096 |

Both active jobs successfully created their srun steps and reached `run_stage.py`.
Their launch commands showed `--gpus=1` and `--gres=none`, respectively. Both stopped
at `Python environment changed after setup; use a new environment and campaign`.
This verifies the repaired launch path on the two reported nodes; it does not verify
ML correctness or any real-model result.

Inspection found that the fingerprint used ordinary `python -m pip freeze` while the
job appended the submitting checkout's PYTHONPATH. A local real-pip reproduction with
installed direct-URL dist-info and source egg-info produced `fixture-clm==0.1` for
ordinary freeze versus `fixture-clm @ file:///fixture/original-install` for isolated
freeze, with no installed package change. The Unity diff has not yet been supplied;
we cannot claim this is its exact discrepancy without checking the original hash.

The repair isolates package enumeration, removes mutable inherited Python source paths
from batch jobs, and verifies dependencies before sbatch as well as at job execution.
Existing locks are never refreshed; mismatch with the original freeze or campaign
record remains fatal and includes a redacted diff. Setup uses isolated enumeration for
new environments. The model and task container are unaffected.

Local regression run: **19 passed, 2 deselected** (the two aggregation tests import
unavailable PyTorch). Five new real-pip tests cover source-metadata shadowing, a real
version change represented by replacement dist-info, modified freeze/READY records,
and diagnostic URL redaction. Workflow tests verify a rejected environment submits no
jobs and creates no campaign. The first synthetic upgrade fixture changed METADATA
without renaming its versioned dist-info directory; pip continued reporting its old
directory version, so that test failed. The fixture was corrected to represent an
installed upgrade, then the targeted suite passed. These tests bootstrap an isolated
temporary venv using the installed pip code, without network access or ensurepip.
Python compilation, shell syntax and diff checks passed. No Unity validation of this
fingerprint repair or new baseline submission was executed by the assistant.

## baseline-003: completed workflow, zero task reward

The following measurements were supplied in the user's terminal output, not produced
by assistant access to Unity. Commit: `e3181bd8586278ef83b88b0123efb3648da8c79b`.
The original installed-package fingerprint verified unchanged:
`f050b1957056a34c40df4f892cbe2c34bae39af1652b084515e329fd7167dccb`.
All nine jobs (1472969–1472977) completed with exit 0:0. Environment, correctness and
smoke took 11 seconds, 1:35 and 25 seconds. ContextBench took 55:06. Slurm completion
means outputs were produced, not that the capability gate passed.

The gate was false. Its manifest hash was
`30d6f095933170bebfcc72e1657b88bb21f743b8829e086f66d717018ac91e91`.
The run `original-keep0.8-seed0` was the uncompressed original checkpoint, with zero
training tokens and zero projection time. The inherited keep0.8 run label does not
mean any compression took place: whole_keep=1.0.

| Measured property | User-reported result |
| --- | --- |
| Parameters, original and exported | 596,049,920 |
| Serialized weight bytes | 1,192,135,112 |
| Validation NLL / perplexity | 3.2950888499617577 / 26.97981111859551 |
| Export NLL delta / reload maximum logit error | 0.0 / 0.0 |
| CLM / append-only mean reward | 0.0 / 0.0 |
| Artifact SHA-256 | ca051e9da0b09e6c14792278cefdaddd3e52f24507c0b7fb4e12b9575e14b2d3 |

Reported hardware: NVIDIA A100-SXM4-80GB. The reference-backend profile produced:

| Prompt tokens | Median prefill seconds | Decode tokens/s | Peak CUDA allocated bytes |
| --- | --- | --- | --- |
| 128 | 0.02805823998642154 | 37.21851447113786 | 1,255,424,000 |
| 512 | 0.02834520599571988 | 37.55204807776526 | 1,415,990,272 |
| 2048 | 0.03374105098191649 | 37.5838932537975 | 2,062,580,736 |

These are uncompressed fixed-workload measurements, not speedups or task latency.
Peak allocation includes weights, activations, workspaces and cache; it is not an
isolated weight-memory figure. The full profile sample arrays have not been supplied.

Every episode had status=complete, grader_valid=1, error=null and reward=0:

| Mode | Level / seed | Model calls | Parsed tool calls | Ops delivered / total | Answers / 24 |
| --- | --- | --- | --- | --- | --- |
| CLM | R1000 / 1101 | 128 | 0 | 1 / 34 | 0 |
| CLM | R1000 / 1102 | 115 | 0 | 1 / 34 | 0 |
| CLM | R130 / 1101 | 116 | 0 | 1 / 26 | 0 |
| CLM | R130 / 1102 | 128 | 0 | 1 / 26 | 0 |
| Append-only | R1000 / 1101 | 124 | 22 | 4 / 34 | 0 |
| Append-only | R1000 / 1102 | 128 | 17 | 1 / 34 | 0 |
| Append-only | R130 / 1101 | 128 | 17 | 1 / 26 | 0 |
| Append-only | R130 / 1102 | 85 | 84 | 26 / 26 | 24 |

All matched-answer counts were zero, including the append-only episode that reached
all operations and attempted 24 answers. The CLM failures occur before meaningful
stream progress. In the inspected original harness, `_extract_tool_call` requires a
structured bash call; a response without one adds a reminder and continues.
`ClmStream._is_successful_submit` advances the stream only after the tool result.
The upstream append-only agent at the pinned ContextBench revision also uses structured
bash calls, with a shorter system prompt and its own reminder. The server counted
parsed calls, not necessarily valid/executed commands. Thus a zero CLM count is a
protocol/capability diagnostic, but does not alone distinguish model instruction
failure from malformed output or client/server handling. Saved replies are needed.

The benchmark budget uses o200k_base text counts; server context lengths use the
Qwen tokenizer and chat template. Values above 32,768 in the latter are not, by
themselves, a violation of the benchmark's stated counter. Do not silently change
tokenizer calibration, increase caps, weaken the gate, or substitute a larger model
and pool it with this run. Any changed protocol requires a separate recorded baseline.
No real-model constrained method has run; this result does not assess its efficacy.

Added a standard-library, read-only inspector for the two upstream transcript layouts.
Four local tests passed for early/late excerpts, campaign byte preservation, missing
transcripts and explicit truncation. No Unity execution of this inspector has yet been
observed. The actual model replies and detailed profiler samples remain uninspected.

### Saved replies supplied after the first baseline report

The user ran the inspector at `22f06e96a3b54faf23ba0e69884bd1a798db43a9` and supplied
its output. The sampled replies establish the following, without inferring unseen
turns or changing the original scores:

* CLM R1000/1101: 128 server finish reasons were `stop`; n_bash=0 and
  n_ctx_syncs_real=0. The stream recorded `max_steps`. Early replies wrote a Python
  here-document after the ordinary text `bash`, with no structured tool call. The
  harness repeatedly replied that no tool call was present. The last replies reduced
  to an unfinished here-document header. No requested offloading code was executed.
* CLM R130/1101: 116 finish reasons were `stop`; n_bash=0 and n_ctx_syncs_real=0.
  The stream recorded `budget_wall`. The model claimed the first batch was already
  offloaded and generated references to subsequent batch labels despite only one
  operation being delivered. These are unsupported generated statements, not observed
  context-management actions. The repeated no-tool-call turns consumed the budget.
* Append-only R130/1102: 84 structured calls reached execution. An early call tried
  to execute `SET K00000 = ...` as a shell command and returned exit 127. A late answer
  marker was emitted without shell quoting and produced a syntax error, exit 2. The
  following readiness command succeeded. The stream recorded `stream_exhausted`.
  Thus all 26 operations being delivered and 24 answers counted by the grader do not
  mean correct answers or successful shell execution. All matched answers were zero.

The observed failure includes output-format and task-execution problems before useful
CLM memory management. It cannot support a claim about compression or long-horizon
retention. Do not convert arbitrary generated shell text into executed calls: that
would change the original harness's protocol rather than validate this baseline.
Neither a longer step cap nor a weaker threshold addresses the observed behavior.

The common server parsed some append-only tool calls, but CLM and append-only use
different clients (LiteLLM versus OpenAI). The saved text alone cannot establish
whether CLM's request contained the tool schema. Added a no-model loopback probe of
the original CLM query method with its actual installed client, checking both the
request and the structured response. Model generation and task behavior are outside
this probe's scope. No model, decoding setting, prompt, parser or gate was changed.

Local fixture/inspector tests: **6 passed, 1 skipped**. The skipped test is the real
CLM SDK round trip: this local runtime lacks LiteLLM and Harbor. A local attempt to
install the pinned LiteLLM dependencies did not complete and was interrupted; no
user environment was modified. The fixture tests exercise an actual localhost HTTP
exchange, reject dropped tools and text-only replies, and preserve saved transcripts.
Python compilation and diff checks passed. The full SDK probe still requires execution
in the user's already installed Unity environment; no new GPU run was performed.

### Unity client transport result and next diagnostic

The user subsequently ran `compression/check_client_protocol.py` at
`78e579dd0091837f53bbf5d92751596e8cb955dc` on Unity. It reported
`CLM_CLIENT_PROTOCOL_PASSED`: request tools, decoding settings, structured response
and original action extraction all passed. Versions: LiteLLM 1.109.0, OpenAI 3.49.0,
Harbor 0.16.1. This is actual user-supplied execution evidence for the synthetic
round trip; no model generation or task execution occurs in that probe. The previous
paragraph's untested status for this probe is superseded by this output.

Added a separate [two-episode 4B screen](BACKBONE_SCREEN.md), preserving the original
protocol and reusing the locked task container. It changes the backbone explicitly,
performs no training and cannot satisfy baseline/pilot gates. No results from that
model are available yet. Baseline-003 remains a valid zero-reward negative result.

Follow-up local validation (2026-11-12): **44 passed, 1 skipped, 2 deselected**
for the screen, container environment, Slurm environment, installed-environment
fingerprint, workflow, client fixture and transcript inspector tests. The real SDK
probe is skipped locally (its Unity pass is recorded above). Two existing aggregate
tests were deselected because this runtime has no PyTorch. No Torch/Transformers
model execution is claimed for the new screen. Python compilation, shell syntax and
Git whitespace checks passed. The screen's first Unity job runs the full inexpensive
suite, including the tests unavailable here, before downloading or running the 4B model.
