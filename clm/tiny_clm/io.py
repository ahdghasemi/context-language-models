from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


@contextlib.contextmanager
def atomic_target(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        yield Path(tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    with atomic_target(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_checkpoint(path, state):
    with atomic_target(path) as tmp:
        torch.save(state, tmp)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def metadata():
    import importlib.metadata
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = os.environ.get("TINY_CLM_COMMIT", "unknown"), None
    packages = {}
    for name in ["torch", "transformers", "numpy", "datasets", "safetensors", "harbor"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"commit": os.environ.get("TINY_CLM_COMMIT", commit), "dirty": dirty,
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_runtime": torch.version.cuda,
            "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_total_memory": [torch.cuda.get_device_properties(i).total_memory
                                 for i in range(torch.cuda.device_count())],
            "slurm": {k: os.environ[k] for k in ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                     "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST"] if k in os.environ},
            "timestamp_unix": time.time()}


def memory():
    import resource
    return {"host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "cuda_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
            "cuda_peak_allocated": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "cuda_peak_reserved": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}


@contextlib.contextmanager
def exclusive(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
