"""Verify installed dependencies independently of the selected source checkout."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess


def frozen_packages(python):
    # -I ignores PYTHONPATH, the working directory and user site packages. Without
    # it, a checkout's build-generated egg-info can shadow installed dist-info.
    return subprocess.check_output([str(python), "-I", "-m", "pip", "freeze"])


def safe_diff(before, after):
    lines = difflib.unified_diff(before.decode().splitlines(), after.decode().splitlines(),
                                fromfile="recorded freeze.txt", tofile="isolated installed packages",
                                lineterm="")
    # Direct requirements may carry credentials/query tokens. Do not log them.
    return "\n".join(re.sub(r"(?:git\+)?https?://\S+", "<remote URL redacted>", line)
                     for line in lines)


def verify_environment(venv, expected=None):
    """Keep existing hashes authoritative; never refresh or normalize a lock."""
    root = Path(venv)
    ready = json.loads((root/"READY.json").read_text())
    if expected is not None and ready != expected:
        raise RuntimeError("READY.json differs from the campaign's recorded environment")
    recorded = (root/"freeze.txt").read_bytes()
    if hashlib.sha256(recorded).hexdigest() != ready["freeze_sha256"]:
        raise RuntimeError("freeze.txt differs from READY.json; preserve both for diagnosis")
    observed = frozen_packages(root/"bin/python")
    actual = hashlib.sha256(observed).hexdigest()
    if actual != ready["freeze_sha256"]:
        raise RuntimeError(
            "Installed environment does not match its recorded freeze. No lock was changed.\n"
            f"Expected SHA256: {ready['freeze_sha256']}\nActual SHA256: {actual}\n"
            + safe_diff(recorded, observed))
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True)
    args = parser.parse_args()
    ready = verify_environment(args.venv)
    print(json.dumps({"verified": True, "check": "python -I -m pip freeze",
                      "freeze_sha256": ready["freeze_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
