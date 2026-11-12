"""Environment for explicitly mounted ContextBench containers."""
from __future__ import annotations

import os


INHERITED_MOUNTS = tuple(
    f"{prefix}_{suffix}"
    for prefix in ("APPTAINER", "SINGULARITY")
    for suffix in ("BIND", "BINDPATH", "MOUNT")
)


def container_environment(base=None):
    """Keep caches/runtime settings; exclude implicit filesystem mounts.

    Unity's module exports APPTAINER_BINDPATH. Apptainer forwards BINDPATH
    and MOUNT into the writable build's nested exec, where absent destination
    directories cause failure. Harbor likewise inherits these variables.
    Our containers use explicit command-line mounts, which remain unchanged.
    Never mutate the caller's environment or the cluster configuration.
    """
    result = dict(os.environ if base is None else base)
    for key in INHERITED_MOUNTS:
        result.pop(key, None)
    return result
