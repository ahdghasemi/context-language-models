# KV command failures and a bounded stream-parser correction

This is a CPU diagnosis and a proposed external patch, not a successful CLM
baseline. No compression, model inference, task rescoring or container modification
is performed. All existing protocols and gates remain unchanged.

## Distinguish retrieval from stream advancement

The pinned KV generator emits unquoted lines with spaces around the equals sign,
for example `SET K00000 = alpha beta #fixture`. Two shell patterns can silently
discard the entire value:

| Pattern | Behavior on that synthetic line |
|---|---|
| `grep -o '="[^"]*"'` | Requires quotes immediately after `=`; no match. |
| `grep -o 'SET K00000 =[^ ]*'` | Matches only through `=` because the next character is a space; `cut` returns an empty value. |

In a pipeline without `pipefail`, a succeeding final `cut` can leave exit status
zero even when an earlier `grep` fails. An `echo READY_FOR_NEXT_OP` can therefore
advance the task after empty retrieval. A recognized answer block may itself be
empty; recognized answers and successful shell exit codes are not correct answers.
Restricting retrieval to the first batch also cannot retrieve keys stored only in
later batches. These are agent command errors, not export or compression errors.

Local Bash checks on a temporary synthetic file returned exit zero for both bad
patterns: the quoted-value pipeline returned no text; the space-stopping pattern
followed by `echo` returned a blank line and the readiness marker. A plain
`grep ... | cut ...` control recovered the fixture value. This diagnoses shell
semantics only; it neither repairs model commands nor establishes full-episode
quality or the contents of any saved task sandbox.

## Separate harness defect

In pinned `contextbench/stream.py`, `_strip_heredoc_bodies` searches each physical
line with `_HEREDOC_OPEN.findall(line)` without checking shell quote state. Thus a
quoted `<<<ANSWER` can be mistaken for a here-document opener. If a `printf` format
contains literal newlines, the detector discards the remainder of the command,
including a real final readiness statement. Bash executes it, but the protocol
returns a misleading non-standalone-marker explanation.

Minimal reproducer, where the newlines inside the quoted format are **literal**:

```bash
printf '<<<ANSWER key=K00000>>>
%s
<<<ANSWER END>>>
' value; echo READY_FOR_NEXT_OP
```

The proposed [patch](../patches/contextbench-quoted-heredoc.patch) recognizes
here-document operators outside quotes, carries quote state between physical
lines, and excludes here-strings, comments and escaped operators. It preserves
the existing identifier-delimiter grammar and statement matcher. Real here-doc
data and quoted marker text must still not advance the stream. Both readiness
and submission checks use this helper.

**This is not a general Bash parser.** Nested command substitutions remain a
known limitation of `_split_unquoted`, explicitly covered by a strict expected-
failure test. The narrow correction is not ready to replace the full shell
recognition contract without further validation. It does not fix empty retrieval,
missing answer blocks, or incorrect file selection.

## Source and CPU validation

Source: `RulinShao/ContextBench` revision
`fb533c18c78a82d3f1b2047a9c1700735cf829ae` (Apache-2.0).

- [Generator](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/tasks/kv_store/generate.py)
- [Turn protocol](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/contextbench/stream.py)
- [Original protocol tests](https://github.com/RulinShao/ContextBench/blob/fb533c18c78a82d3f1b2047a9c1700735cf829ae/tests/test_stream.py)

Original file SHA256 values, verified against their pinned Git blobs:

```text
contextbench/stream.py  60b0d05b58e2f2a9ecbd0f7b8d25c5572b386120681e51ae6586b31c6a057056
tests/test_stream.py   508a863147c07e721c9f99bb7111caf3b4199486b92b24753af7d3ae28eed9f9
```

Executed locally on CPU: the 20 added non-xfail regressions produced 10 failures
and 10 passes against the original source. The isolated patched source passed
all 37 original tests plus all 20 new cases. The additional strict expected failure
records the remaining nested-substitution defect. No Unity model episode was run
with this patch, and no quality improvement is claimed.

To reproduce, set `TINY_CLM_ENV_FILE` to your existing environment configuration.
The following copies only the two public source files to
a temporary directory, verifies their hashes, applies the patch there, and runs
CPU tests using the existing environment. It never edits the locked checkout or
campaign and submits no Slurm jobs. This optional check is not another GPU stage.

```bash
(
  set -euo pipefail
  source "${TINY_CLM_ENV_FILE:?Set the path to your existing environment configuration}"
  cd "$TINY_CLM_ROOT"
  TINY_CLM_CB_SOURCE="$TINY_CLM_CONTEXTBENCH/fb533c18c78a82d3f1b2047a9c1700735cf829ae"
  TINY_CLM_PARSER_CHECK=$(mktemp -d "${TMPDIR:-/tmp}/tiny-clm-parser.XXXXXX")
  mkdir -p "$TINY_CLM_PARSER_CHECK/contextbench" "$TINY_CLM_PARSER_CHECK/tests"
  cp "$TINY_CLM_CB_SOURCE/contextbench/stream.py" "$TINY_CLM_PARSER_CHECK/contextbench/"
  cp "$TINY_CLM_CB_SOURCE/tests/test_stream.py" "$TINY_CLM_PARSER_CHECK/tests/"
  touch "$TINY_CLM_PARSER_CHECK/contextbench/__init__.py"
  cd "$TINY_CLM_PARSER_CHECK"
  sha256sum --check <<'SHA'
60b0d05b58e2f2a9ecbd0f7b8d25c5572b386120681e51ae6586b31c6a057056  contextbench/stream.py
508a863147c07e721c9f99bb7111caf3b4199486b92b24753af7d3ae28eed9f9  tests/test_stream.py
SHA
  patch --batch --fuzz=0 -p1 < "$TINY_CLM_ROOT/compression/patches/contextbench-quoted-heredoc.patch"
  PYTHONPATH="$TINY_CLM_PARSER_CHECK" "$TINY_CLM_VENV/bin/python" \
    -m pytest -q --import-mode=importlib tests/test_stream.py
  printf 'Isolated CPU check: %s\n' "$TINY_CLM_PARSER_CHECK"
)
```

Do not apply this patch to a source tree referenced by completed or queued runs.
Adopting any corrected parser requires a new recorded source identity, new task
lock where affected, and fresh matched dense/control evaluations. Scores from the
original protocol remain original. A corrected parser alone does not establish
a capable backbone; recurrent command errors require a separately validated
backbone or explicit behavior training before compression comparisons are useful.
