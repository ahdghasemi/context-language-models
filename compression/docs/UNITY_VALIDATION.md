# Unity validation from user-supplied terminal output

These observations were supplied by the user while running the delivered commit
`10b85e71a6e52eb6cbfd59002f68a0f3c5db826a` on Unity. They were not executed remotely
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

The recipe adds a guarded `%setup` command that creates only the `/nas` directory
under Apptainer's supplied build root. The existing pinned base image, package setup,
runtime containment flags and task grader remain unchanged. The source for this
approach is Apptainer's documented build bind-point requirement:
https://apptainer.org/docs/user/1.5/build_a_container.html#bind

Local shell checks cover the recipe sections, successful creation under a temporary
build root, and rejection of missing/root-directory build-root variables. No Apptainer
runtime is available in the assistant's environment, so a full corrected build still
requires the user's Unity retry. The fix must not be reported as a completed container
build until its output and runtime check succeed.

Real-model/data preparation completion has not yet been shown in the user's output.
Constrained training on the real checkpoint, baseline CLM competence, production GPU
profiling, and complete Slurm campaigns remain unverified.
