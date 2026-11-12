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

The current fix removes the directory workaround and instead clears inherited bind
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
No Apptainer runtime is available there, so full build/runtime validation still requires
the user's Unity retry. Do not report this as a completed container build until its
output and runtime check succeed. New total test count is 45; only the new two tests
were executed locally for this fix, and the previous 43 passed on Unity as noted above.

Real-model/data preparation completion has not yet been shown in the user's output.
Constrained training on the real checkpoint, baseline CLM competence, production GPU
profiling, and complete Slurm campaigns remain unverified.
