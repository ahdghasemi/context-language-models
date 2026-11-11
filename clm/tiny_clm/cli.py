from __future__ import annotations

import argparse
import json
import os
import sys

from .io import metadata, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Constrained Qwen3 compression and reference deployment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", required=True)
    p.add_argument("--destination", required=True)
    p = sub.add_parser("environment")
    p.add_argument("--output", required=True)
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--min-vram-gib", type=float, default=0)
    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p = sub.add_parser("export")
    p.add_argument("--run", required=True)
    p.add_argument("--device", default="cuda")
    for name in ("evaluate", "profile", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--artifact", required=True)
        p.add_argument("--device", default="cuda")
        if name != "serve":
            p.add_argument("--output", required=True)
        if name == "evaluate":
            p.add_argument("--split", choices=["validation", "test"], default="validation")
            p.add_argument("--selection-record")
        elif name == "profile":
            p.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
            p.add_argument("--decode-tokens", type=int, default=32)
            p.add_argument("--repeats", type=int, default=7)
            p.add_argument("--batch-size", type=int, default=1)
        else:
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--log", required=True)
            p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if args.command == "environment":
        info = metadata()
        write_json(args.output, info)
        print(json.dumps(info, indent=2))
        if args.require_cuda:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("this workflow requires exactly one visible CUDA GPU")
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("default mixed precision requires BF16 support")
            if torch.cuda.get_device_properties(0).total_memory < args.min_vram_gib*2**30:
                raise RuntimeError("GPU is below requested pilot memory floor")
            # Exercise GEMM, backprop and exact FP32 SVD on the assigned device.
            x = torch.randn(64, 32, device="cuda", requires_grad=True)
            (x@x.T).square().mean().backward()
            torch.linalg.svd(x.detach(), full_matrices=False)
    elif args.command == "prepare":
        from .assets import prepare
        result = prepare(read_json(args.config), args.destination)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        from .assets import resolve
        from .data import load_blocks
        from .model import dense_model
        from .train import train
        if args.device.startswith("cuda") and not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("GPU training must run in a Slurm allocation")
        config, assets = resolve(read_json(args.config), args.assets)
        model = dense_model(assets["model_path"], device=args.device)
        train_data = load_blocks(assets["files"]["train"]["path"], model.config.vocab_size)
        val_data = load_blocks(assets["files"]["validation"]["path"], model.config.vocab_size)
        complete = train(model, train_data, val_data, config, args.run, resume=args.resume)
        if not complete:
            sys.exit(75)  # intentional incomplete job: afterok/aftercorr must not release export
    elif args.command == "export":
        from .evaluate import export_run
        export_run(args.run, args.device)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args.artifact, args.output, args.split, args.selection_record, args.device)
    elif args.command == "profile":
        from .evaluate import profile
        profile(args.artifact, args.output, args.device, args.lengths, args.decode_tokens,
                args.repeats, batch_size=args.batch_size)
    elif args.command == "serve":
        from .serve import serve
        serve(args.artifact, args.device, args.port, args.log, args.seed)


if __name__ == "__main__":
    main()
