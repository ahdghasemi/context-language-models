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

Real-model/data preparation completion has not yet been shown in the user's output.
Constrained training on the real checkpoint, baseline CLM competence, production GPU
profiling, and complete Slurm campaigns remain unverified.
