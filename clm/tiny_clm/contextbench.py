"""Pinned original ContextBench + original CLM adapter; no substitute benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from .io import digest, exclusive, file_sha, metadata, read_json, write_json


def tree_hash(root):
    return digest({str(p.relative_to(root)): file_sha(p)
                   for p in sorted(Path(root).rglob("*")) if p.is_file()})


def check_source(source, revision):
    current = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if current != revision or dirty:
        raise ValueError("ContextBench source changed; use a clean checkout at the pinned revision")


def verify_lock(lock, cb, *, runnable=True):
    if lock["request_hash"] != digest(cb) or (runnable and not lock["runnable"]):
        raise ValueError("prepared tasks differ from the request or lack a container")
    check_source(lock["source"], lock["revision"])
    if lock["runnable"] and file_sha(lock["image"]) != lock["image_sha256"]:
        raise ValueError("task container changed")
    for task in lock["tasks"]:
        if tree_hash(Path(task["path"])) != task["sha256"]:
            raise ValueError("prepared task changed: "+task["path"])


def prepare(config, source, destination, recipe, *, skip_container=False):
    """skip_container is ONLY for task-builder tests; such tasks are not runnable."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive(destination/".prepare.lock"):
        cb = config["contextbench"]
        lock_path = destination/"contextbench-lock.json"
        if lock_path.exists():
            lock = read_json(lock_path)
            verify_lock(lock, cb, runnable=not skip_container)
            if lock["recipe_sha256"] != file_sha(recipe) or lock["source"] != str(source):
                raise ValueError("preparation inputs changed; choose a new destination")
            return lock
        if (destination/"tasks").exists():
            raise ValueError("partial preparation exists without a lock; preserve it for diagnosis and use a new destination")
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/RulinShao/ContextBench.git", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", cb["revision"]], check=True)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != cb["revision"] or subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"], text=True):
            raise ValueError("ContextBench source must be clean and at the pinned revision; use a new source directory")
        sys.path.insert(0, str(source))
        import yaml
        official = yaml.safe_load((source/"contextbench/configs/paper.yaml").read_text())
        subset = {k: v for k, v in official.items() if k != "families"}
        subset["families"] = {}
        for task in cb["tasks"]:
            family, level = task["family"], task["level"]
            definition = official["families"][family]
            entries = [x for x in definition["levels"] if x["name"] == level]
            if len(entries) != 1:
                raise ValueError(f"unknown ContextBench level {family}/{level}")
            subset["families"].setdefault(family, {"fixed": definition["fixed"], "levels": []})["levels"].extend(entries)
        # Keep official feasibility budgets. Altering these requires rebuilding tasks and auditing pressure.
        if cb["budget"] != 32768 or cb["reserve"] != 2048:
            raise ValueError("packaged paper levels require 32768/2048 budgets")
        subset_path = destination/"levels.yaml"
        subset_path.write_text(yaml.safe_dump(subset))
        image = destination/"contextbench.sif"
        if not skip_container and not image.exists():
            if not shutil.which("apptainer"):
                raise RuntimeError("load apptainer/latest on Unity before container preparation")
            tmp = destination/"contextbench.building.sif"
            subprocess.run(["apptainer", "build", "--fakeroot", str(tmp), str(recipe)], check=True)
            os.replace(tmp, image)
        if not skip_container:
            if not shutil.which("singularity"):
                raise RuntimeError("Harbor requires the singularity compatibility command")
            # Exercise the actual Harbor isolation flags before any GPU array.
            subprocess.run(["singularity", "exec", "--writable-tmpfs", "--fakeroot",
                "--containall", "--pid", "--no-mount", "home,tmp,bind-paths", str(image),
                "/opt/harbor-server/bin/python3", "-c", "import fastapi,uvicorn; print('container runtime check passed')"], check=True)
        # Pinned source via PYTHONPATH, never editable pip installs into a live job environment.
        env = dict(os.environ, PYTHONPATH=str(source)+os.pathsep+os.environ.get("PYTHONPATH", ""))
        tasks = destination/"tasks"
        if not tasks.exists():
            subprocess.run([sys.executable, "-m", "contextbench.cli", "package", "--all",
                "--config", str(subset_path), "--seeds", ",".join(map(str, cb["seeds"])),
                "--out-dir", str(tasks), "--image", str(image),
                "--agent-timeout-sec", str(cb["episode_timeout"])], env=env, check=True)
        names = sorted(str(p.parent) for p in tasks.glob("*/task.toml"))
        if len(names) != len(cb["tasks"])*len(cb["seeds"]):
            raise ValueError("task count differs from configured episodes")
        manifest = {"request_hash": digest(cb), "source": str(source), "revision": revision,
                    "image": str(image), "image_sha256": None if skip_container else file_sha(image),
                    "tasks": [{"path": p, "sha256": tree_hash(Path(p))} for p in names],
                    "runnable": not skip_container, "recipe_sha256": file_sha(recipe)}
        write_json(destination/"contextbench-lock.json", manifest)
        return manifest


def wait_server(process, url, expected_sha, deadline=180):
    end = time.monotonic()+deadline
    while time.monotonic() < end:
        if process.poll() is not None:
            raise RuntimeError("reference server exited; inspect its log")
        try:
            with urllib.request.urlopen(url+"/health", timeout=2) as response:
                health = json.load(response)
            if health.get("weight_sha256") != expected_sha:
                raise RuntimeError("port belongs to a different model server")
            return
        except OSError:
            time.sleep(.2)
    raise TimeoutError("reference server did not become ready")


def run(artifact, config, lock_path, output, device="cuda"):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    cb, lock = config["contextbench"], read_json(lock_path)
    verify_lock(lock, cb)
    if not shutil.which("singularity"):
        raise RuntimeError("Harbor needs the singularity compatibility executable from Apptainer")
    spec = read_json(Path(artifact)/"compression.json")
    env = dict(os.environ, OPENAI_API_KEY="EMPTY",
               PYTHONPATH=lock["source"]+os.pathsep+os.environ.get("PYTHONPATH", ""))
    episodes = []
    for mode in cb["modes"]:
        agent = {"clm": "clm_contextbench:ClmStream",
                 "append_only": "contextbench.harbor.agents.base:ContextBenchAgent"}[mode]
        for ti, task in enumerate(lock["tasks"]):
            path = Path(task["path"])
            if tree_hash(path) != task["sha256"]:
                raise ValueError(f"prepared task changed: {path}")
            name = mode+"--"+path.name
            record_path = out/(name+".json")
            if record_path.exists():
                record = read_json(record_path)
                if record["weight_sha256"] != spec["weight_sha256"] or record["request_hash"] != digest(cb):
                    raise ValueError("existing episode belongs to another experiment")
                episodes.append(record)
                continue
            # Port selection has a bounded race; readiness checks model identity.
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            api = f"http://127.0.0.1:{port}"
            server_log = out/(name+".server.log")
            usage_path = out/(name+".requests.jsonl")
            if usage_path.exists() or (out/"trials"/name).exists():
                raise FileExistsError("partial episode exists without a final record; archive its logs and trial directory before retrying: "+name)
            with server_log.open("w") as logfile:
                server = subprocess.Popen([sys.executable, "-m", "tiny_clm.cli", "serve",
                    "--artifact", str(artifact), "--device", device, "--port", str(port),
                    "--log", str(usage_path), "--seed", str(10000+ti)], stdout=logfile, stderr=subprocess.STDOUT, env=env)
                start = time.perf_counter()
                try:
                    wait_server(server, api, spec["weight_sha256"])
                    kwargs = {"api_base": api+"/v1", "context_budget_tokens": cb["budget"],
                        "context_budget_reserve_tokens": cb["reserve"], "max_tokens": cb["max_tokens"],
                        "temperature": cb["temperature"], "top_p": cb["top_p"], "max_steps": cb["max_steps"]}
                    if mode == "clm":
                        kwargs.update(enable_thinking=cb["enable_thinking"], cost_metric="usd",
                                      skills_dir=str(Path(lock["source"])/"clm_exp/skills"),
                                      lm_call_cap=cb["max_steps"])
                    else:
                        kwargs["chat_template_kwargs"] = {"enable_thinking": cb["enable_thinking"]}
                    command = [str(Path(sys.executable).with_name("harbor")), "trials", "start",
                        "-p", str(path), "-a", agent, "-m", "openai/tiny-clm", "-e", "singularity",
                        "--trial-name", name, "--trials-dir", str(out/"trials"),
                        "--agent-timeout", str(cb["episode_timeout"])]
                    for key, value in kwargs.items():
                        command.extend(["--agent-kwarg", key+"="+(json.dumps(value) if not isinstance(value, str) else value)])
                    start = time.perf_counter()
                    with (out/(name+".harbor.log")).open("w") as trial_log:
                        process = subprocess.run(command, env=env, stdout=trial_log, stderr=subprocess.STDOUT,
                                                 timeout=cb["episode_timeout"]+600)
                    result_path = out/"trials"/name/"result.json"
                    result = read_json(result_path) if result_path.exists() else {}
                    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
                    requests = [json.loads(x) for x in usage_path.read_text().splitlines()] if usage_path.exists() else []
                    error = result.get("exception_info")
                    # Scored low/zero reward is a valid negative result. Infrastructure/agent exceptions are failures.
                    status = "complete" if process.returncode == 0 and not error and rewards.get("grader_valid") == 1 else "failed"
                    record = {"episode": path.name, "mode": mode, "status": status, "rewards": rewards,
                        "error": error, "returncode": process.returncode, "elapsed_seconds": time.perf_counter()-start,
                        "agent_execution": result.get("agent_execution"), "request_hash": digest(cb),
                        "prompt_tokens": sum(r["prompt_tokens"] for r in requests),
                        "completion_tokens": sum(r["completion_tokens"] for r in requests),
                        "tool_calls": sum(r["tool_calls"] for r in requests), "model_calls": len(requests),
                        "context_lengths": [r["prompt_tokens"] for r in requests],
                        "weight_sha256": spec["weight_sha256"], "task_sha256": task["sha256"]}
                    write_json(record_path, record)
                    episodes.append(record)
                except Exception as exc:
                    record = {"episode": path.name, "mode": mode, "status": "failed", "rewards": {},
                              "error": {"type": type(exc).__name__, "message": str(exc)},
                              "request_hash": digest(cb), "weight_sha256": spec["weight_sha256"],
                              "task_sha256": task["sha256"], "elapsed_seconds": time.perf_counter()-start}
                    write_json(record_path, record)
                    episodes.append(record)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait()
    result = {"episodes": episodes, "metadata": metadata(), "contextbench": cb,
              "container_sha256": lock["image_sha256"], "source_revision": lock["revision"],
              "all_complete": all(r["status"] == "complete" for r in episodes),
              "scope": "original CLM harness and ContextBench append-only agent; substituted Qwen3 backbone and stated decoding"}
    write_json(out/"summary.json", result)
    if not result["all_complete"]:
        raise RuntimeError("ContextBench has failed episodes; see summary.json and logs")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--source", required=True)
    a.add_argument("--destination", required=True)
    a.add_argument("--recipe", required=True)
    a.add_argument("--config", required=True)
    a = sub.add_parser("run")
    a.add_argument("--config", required=True)
    a.add_argument("--artifact", required=True)
    a.add_argument("--lock", required=True)
    a.add_argument("--output", required=True)
    a.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = read_json(args.config)
    if args.command == "prepare":
        prepare(config, args.source, args.destination, args.recipe)
    else:
        run(args.artifact, config, args.lock, args.output, args.device)
