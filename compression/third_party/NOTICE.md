# Third-party provenance

`compression/contextbench.def` is adapted from
`RulinShao/ContextBench/environment/contextbench.def` at commit
`fb533c18c78a82d3f1b2047a9c1700735cf829ae`, distributed under Apache-2.0.
The original license is included as `ContextBench-LICENSE`.
Changes pin the Python dependency versions, require asciinema installation, add a test
section and simplify the build commands. The server-venv layout and Python 3.13.15
version are retained. The base image changes from the upstream Debian bookworm image to
the digest-pinned official Python 3.13.15 slim-trixie image to satisfy Unity's host
fakeroot GLIBC_2.38 dependency at build and task runtime. This is a task-environment
adaptation, not an exact reproduction of the upstream container; see
`compression/docs/UNITY_VALIDATION.md` for provenance and validation limits.
The ContextBench repository is downloaded separately by the workflow;
its authorship and original task/skill/grading code are preserved.

The ASVD baseline is an independent implementation of the published algorithm, with
source/revision and explicit differences documented in `docs/RELATED_WORK.md`.
Supplied private GNN/Flow code and manuscripts are not redistributed.
The original CLM repository's license and attribution are unchanged.
