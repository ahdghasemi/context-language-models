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

`compression/audit_kv_responses.py` uses the ANSWER-block regular expression
from `contextbench/graders/kv.py` at the same pinned Apache-2.0 revision. Its
surrounding code is a separate read-only diagnostic, not a modified benchmark
grader. The original license is included as `ContextBench-LICENSE`.

`compression/patches/contextbench-quoted-heredoc.patch` contains proposed changes to
`contextbench/stream.py` and `tests/test_stream.py` at the same pinned Apache-2.0
revision. It makes here-document detection aware of shell quoting and adds
regression tests. Modified files carry change notices; the original license is
`ContextBench-LICENSE`. No workflow automatically applies this diagnostic patch.

The ASVD baseline is an independent implementation of the published algorithm, with
source/revision and explicit differences documented in `docs/RELATED_WORK.md`.
Supplied private GNN/Flow code and manuscripts are not redistributed.
The original CLM repository's license and attribution are unchanged.
