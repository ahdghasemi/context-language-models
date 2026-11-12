from __future__ import annotations

import contextlib
import math
import signal
import time
from pathlib import Path

import torch

from .data import TokenStream
from .io import (digest, exclusive, memory, metadata, read_json, restore_rng, rng_state,
                 save_checkpoint, seed_all, write_json)
from .lowrank import Constraint, accounting, mu_schedule, replace, select

METHODS = {"original", "dense_matched", "svd", "svd_recovery",
           "asvd_uniform", "asvd_uniform_recovery", "asvd_adaptive", "constrained"}


def validate(config):
    if config["method"] not in METHODS:
        raise ValueError("unknown method")
    t = config["training"]
    for k in ["stage_steps", "batch_size", "accumulation", "eval_every", "checkpoint_every", "validation_batches"]:
        if type(t[k]) is not int or t[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if t["precision"] not in {"float32", "bfloat16"}:
        raise ValueError("use FP32 or BF16 autocast with FP32 master weights")
    if not (math.isfinite(t["lr"]) and t["lr"] > 0):
        raise ValueError("invalid learning rate")
    if t.get("optimizer_policy", "gnn_persistent") not in {"gnn_persistent", "flow_reset"}:
        raise ValueError("optimizer_policy must be gnn_persistent or flow_reset")
    if t.get("adam_foreach") is not None and type(t["adam_foreach"]) is not bool:
        raise ValueError("adam_foreach must be null or a boolean")
    mu_schedule(**config["penalty"])
    constant = config.get("ablation_constant_mu")
    if constant is not None and not (math.isfinite(constant) and constant > 0):
        raise ValueError("constant-mu ablation requires a positive finite value")
    if t.get("projection_every", 1) != 1:
        raise ValueError("faithful method refreshes proxies at every optimizer update")


def autocast_for(model, precision):
    device = next(model.parameters()).device.type
    if precision == "bfloat16":
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 training requires a supported CUDA device; use float32 on CPU")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def nll(model, data, batches=None, batch_size=1, precision="float32"):
    model.eval()
    limit = len(data) if batches is None else min(len(data), batches*batch_size)
    if limit == 0:
        raise ValueError("empty evaluation data")
    total, tokens = 0., 0
    for start in range(0, limit, batch_size):
        x = torch.tensor(data[start:min(start+batch_size, limit)].copy(), dtype=torch.long,
                         device=next(model.parameters()).device)
        with autocast_for(model, precision):
            loss = model(input_ids=x, labels=x, use_cache=False).loss
        count = x.shape[0]*(x.shape[1]-1)
        total += loss.item()*count
        tokens += count
    value = total/tokens
    if not math.isfinite(value):
        raise FloatingPointError("non-finite evaluation NLL")
    return {"nll": value, "perplexity": math.exp(value) if value < 700 else None,
            "scored_tokens": tokens, "blocks": limit}


@torch.no_grad()
def calibrate_asvd(model, ranks, data, batches, alpha=0.5):
    """ASVD's channel scaling with fixed, matched ranks; no rank sensitivity search."""
    sums, counts, handles = {}, {}, []
    for name in ranks:
        layer = model.get_submodule(name)
        sums[name] = torch.zeros(layer.in_features, device=layer.weight.device)
        counts[name] = 0

        def hook(module, args, output, key=name):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            sums[key].add_(x.abs().sum(0))
            counts[key] += x.shape[0]

        handles.append(layer.register_forward_hook(hook))
    try:
        model.eval()
        for block in data[:batches]:
            x = torch.tensor(block.copy(), dtype=torch.long, device=next(model.parameters()).device)[None]
            model(x, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return {n: (sums[n]/counts[n]).pow(alpha)+1e-6 for n in ranks}


def train(model, train_data, val_data, config, run_dir, *, resume=False, stop_after=None):
    """One GPU, full fine-tuning, exact projection per optimizer update.

    stop_after is a software-test interruption at an optimizer boundary. It is
    not part of experiment configuration or a model selection parameter.
    """
    validate(config)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive(root/".run.lock"):
        return _train(model, train_data, val_data, config, root, resume, stop_after)


def _train(model, train_data, val_data, cfg, root, resume, stop_after):
    start_time = time.perf_counter()
    checkpoint = root/"latest.pt"
    if (checkpoint.exists() or (root/"config.json").exists()) and not resume:
        raise FileExistsError("run exists; use --resume or a new run directory")
    if resume and not checkpoint.exists():
        raise FileNotFoundError("resume requested but latest.pt is missing")
    seed_all(cfg["seed"])
    t, method = cfg["training"], cfg["method"]
    seed_meta = metadata()
    if resume and seed_meta["commit"] != read_json(root/"metadata.json")["commit"]:
        raise ValueError("resume requires the same immutable code revision")
    if not resume:
        write_json(root/"config.json", cfg)
        write_json(root/"metadata.json", seed_meta)
    ranks = select(model, cfg["selection"])
    footprint = accounting(model, ranks)
    write_json(root/"target_budget.json", footprint)
    if method in {"original", "dense_matched"}:
        ranks = {}
    scales = None
    if method.startswith("asvd") and not resume:
        scales = calibrate_asvd(model, ranks, train_data, cfg.get("calibration_batches", 16))
        if method == "asvd_adaptive":
            from .asvd import sensitivity_ranks
            ranks, report = sensitivity_ranks(model, ranks, scales, train_data,
                cfg.get("calibration_batches", 16), cfg["selection"]["keep"])
            write_json(root/"asvd_sensitivity.json", report)
    factor_training = method in {"svd_recovery", "asvd_uniform_recovery", "svd", "asvd_uniform", "asvd_adaptive"}
    if factor_training:
        replace(model, ranks, initialize=not resume, scales=scales)
    write_json(root/"footprint.json", accounting(model, ranks))
    scales = None
    model.config.use_cache = False
    if t.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    constraint = Constraint(model, ranks) if method == "constrained" else None
    sampler = TokenStream(train_data, cfg["seed"])
    stages = mu_schedule(**cfg["penalty"])
    if cfg.get("ablation_constant_mu") is not None:
        stages = [cfg["ablation_constant_mu"]]*len(stages)
    if method in {"original", "svd", "asvd_uniform", "asvd_adaptive"}:
        stages = []
    stage, step, total_steps, tokens, best_score, best = 0, 0, 0, 0, float("inf"), None
    opt = sched = None
    wall, projection_seconds, updates_seconds = 0., 0., 0.
    interrupted = [False]
    old_handlers = {}
    initialization_seconds = time.perf_counter()-start_time

    def new_optimizer():
        # Original GNN: retain Adam moments across best-weight restoration.
        # Flow resets are an explicit, separately labeled ablation. No weight decay.
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], betas=(0.9, 0.999),
                                     foreach=t.get("adam_foreach"))
        if t.get("optimizer_policy", "gnn_persistent") == "gnn_persistent":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=t["stage_steps"],
                                                       gamma=math.sqrt(cfg["penalty"]["gamma"]))
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["stage_steps"])
        return optimizer, scheduler

    if resume:
        # Only load our own trusted run checkpoints (Python RNG state is pickled).
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config_hash"] != digest(cfg):
            raise ValueError("resume configuration/input hashes differ")
        if state["complete"]:
            raise ValueError("training already complete; use export/evaluate")
        model.load_state_dict(state["model"], strict=True)
        stage, step, total_steps, tokens = (state[k] for k in ["stage", "step", "total_steps", "tokens"])
        best_score, best = state["best_score"], state["best_model"]
        wall, projection_seconds, updates_seconds = (state[k] for k in ["elapsed_seconds", "projection_seconds", "updates_seconds"])
        initialization_seconds += state.get("initialization_seconds", 0.)
        sampler.load_state_dict(state["sampler"])
        if constraint is not None:
            constraint.load_state_dict(state["constraint"])
        if state["optimizer"] is not None:
            opt, sched = new_optimizer()
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        del state
        # A process may have logged steps beyond the last atomic checkpoint.
        log = root/"training.jsonl"
        if log.exists():
            import json
            from .io import atomic_target
            committed = [line for line in log.read_text().splitlines()
                         if json.loads(line)["global_step"] <= total_steps]
            with atomic_target(log) as tmp:
                tmp.write_text("\n".join(committed)+("\n" if committed else ""))

    def save(complete=False):
        elapsed = wall + time.perf_counter()-start_time
        save_checkpoint(checkpoint, {
            "format": 1, "config": cfg, "config_hash": digest(cfg),
            "model": model.state_dict(), "ranks": ranks, "factorized": factor_training,
            "optimizer": None if opt is None else opt.state_dict(),
            "scheduler": None if sched is None else sched.state_dict(),
            "constraint": None if constraint is None else constraint.state_dict(),
            "rng": rng_state(), "sampler": sampler.state_dict(),
            "stage": stage, "step": step, "mu": stages[stage] if stage < len(stages) else None,
            "total_steps": total_steps, "tokens": tokens, "best_score": best_score,
            "best_model": best, "elapsed_seconds": elapsed, "projection_seconds": projection_seconds,
            "updates_seconds": updates_seconds, "initialization_seconds": initialization_seconds, "complete": complete})
        write_json(root/"status.json", {"status": "complete" if complete else "incomplete",
            "stage": stage, "step": step, "total_steps": total_steps, "training_tokens": tokens,
            "elapsed_seconds": elapsed, "projection_seconds": projection_seconds,
            "updates_seconds": updates_seconds, "initialization_seconds": initialization_seconds, "memory": memory()})

    for sig in [signal.SIGTERM, signal.SIGUSR1]:
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    try:
        if not resume:
            save()  # permits restart even if the first expensive update is killed
        while stage < len(stages):
            mu = stages[stage]
            if opt is None:
                opt, sched = new_optimizer()
            while step < t["stage_steps"]:
                model.train()
                before = time.perf_counter()
                if constraint is not None:
                    constraint.refresh()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                projection_seconds += time.perf_counter()-before
                before = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                task_loss = 0.
                for _ in range(t["accumulation"]):
                    x = sampler.next(t["batch_size"], next(model.parameters()).device)
                    with autocast_for(model, t["precision"]):
                        loss = model(input_ids=x, labels=x, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    (loss/t["accumulation"]).backward()
                    task_loss += loss.item()/t["accumulation"]
                    tokens += x.shape[0]*(x.shape[1]-1)
                # Add penalty once, not once per accumulated microbatch.
                penalty = 0.
                if constraint is not None:
                    p = constraint.penalty(mu)
                    if not torch.isfinite(p):
                        raise FloatingPointError("non-finite coupling penalty")
                    p.backward()
                    penalty = p.item()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("non-finite gradient")
                opt.step()
                sched.step()
                step += 1
                total_steps += 1
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                updates_seconds += time.perf_counter()-before
                record = {"stage": stage, "step": step, "global_step": total_steps, "mu": mu,
                          "train_nll": task_loss, "penalty": penalty, "training_tokens": tokens,
                          "optimizer_policy": t.get("optimizer_policy", "gnn_persistent"),
                          "learning_rate_next": opt.param_groups[0]["lr"]}
                if step % t["eval_every"] == 0 or step == t["stage_steps"]:
                    metric = nll(model, val_data, t["validation_batches"], precision=t["precision"])
                    score = metric["nll"]
                    if constraint is not None:
                        before = time.perf_counter()
                        constraint.refresh()
                        score += constraint.penalty(mu).item()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        projection_seconds += time.perf_counter()-before
                    record.update(validation=metric, selection_objective=score)
                    if score < best_score:
                        best_score = score
                        best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                import json
                with open(root/"training.jsonl", "a") as f:
                    f.write(json.dumps(record, allow_nan=False)+"\n")
                print(json.dumps(record), flush=True)
                if interrupted[0] or (stop_after is not None and total_steps >= stop_after):
                    save()
                    return False
                if step % t["checkpoint_every"] == 0:
                    save()
            if best is None:
                raise RuntimeError("no finite stage validation checkpoint")
            model.load_state_dict(best, strict=True)
            # The GNN source retains moments/scheduler even when restoring earlier weights.
            stage, step, best, best_score = stage+1, 0, None, float("inf")
            if t.get("optimizer_policy", "gnn_persistent") == "flow_reset" or stage == len(stages):
                opt = sched = None
            save()
        save(complete=True)
        return True
    except BaseException as exc:
        write_json(root/"failure.json", {"exception": type(exc).__name__, "message": str(exc),
                   "stage": stage, "step": step, "memory": memory()})
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
