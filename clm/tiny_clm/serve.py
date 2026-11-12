"""Single-worker reference server for dense AND factorized Qwen3 models.

No weight reconstruction, cross-request prefix reuse, suffix reuse, tensor
parallelism, speculative decoding, or production-throughput claims.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import threading
import time
import uuid

import torch

from .io import digest, seed_all
from .model import load_export, tokenizer_from
from .provenance import LEGACY_INFERENCE_RUNTIME


def parse_response(text):
    reasoning = None
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()
    calls = []
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL))
    for match in matches:
        try:
            value = json.loads(match[1])
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                raise ValueError("invalid tool name")
            arguments = value["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be an object")
        except (ValueError, KeyError, TypeError):
            # Keep malformed output visible to harness; never invent a valid action.
            return {"role": "assistant", "content": text, "reasoning_content": reasoning}, "stop"
        calls.append({"id": "call_"+uuid.uuid4().hex[:16], "type": "function",
                      "function": {"name": value["name"], "arguments": json.dumps(arguments)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL).strip()
    message = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message, "tool_calls" if calls else "stop"


class Engine:
    def __init__(self, artifact, device="cpu", log=None, seed=0):
        self.model, self.spec = load_export(artifact, device)
        self.tokenizer = tokenizer_from(artifact)
        self.device = device
        self.lock = threading.Lock()
        self.log = Path(log) if log else None
        if self.log:
            self.log.parent.mkdir(parents=True, exist_ok=True)
        seed_all(seed)

    @torch.no_grad()
    def complete(self, req):
        for key in ("response_format", "logprobs", "stop", "logit_bias"):
            if req.get(key):
                raise ValueError(f"unsupported reference-server parameter: {key}")
        if req.get("stream", False) or req.get("n", 1) != 1:
            raise ValueError("reference backend supports non-streaming n=1 only")
        if req.get("model") not in (None, "tiny-clm"):
            raise ValueError("served model name is tiny-clm")
        choice = req.get("tool_choice", "auto")
        if choice not in ("auto", "none"):
            raise ValueError("tool_choice must be auto or none")
        tools = None if choice == "none" else req.get("tools")
        messages = [dict(m) for m in req["messages"]]
        for message in messages:
            if isinstance(message.get("content"), list):
                if any(x.get("type") != "text" for x in message["content"]):
                    raise ValueError("text-only backbone")
                message["content"] = "".join(x["text"] for x in message["content"])
            for tc in message.get("tool_calls", []):
                tc = dict(tc)
                # The tokenizer template accepts arguments as a dictionary.
                if isinstance(tc["function"].get("arguments"), str):
                    tc["function"] = dict(tc["function"], arguments=json.loads(tc["function"]["arguments"]))
                message["tool_calls"] = [tc if c["id"] == tc["id"] else c for c in message["tool_calls"]]
        template = req.get("chat_template_kwargs", {})
        if set(template) - {"enable_thinking"}:
            raise ValueError("unsupported chat_template_kwargs")
        ids = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
            add_generation_prompt=True, enable_thinking=template.get("enable_thinking", False),
            return_tensors="pt").to(self.device)
        cap = int(req.get("max_completion_tokens", req.get("max_tokens", 512)))
        if cap < 1 or ids.shape[1]+cap > self.model.config.max_position_embeddings:
            raise ValueError("maximum context length exceeded by prompt plus requested output")
        temperature = float(req.get("temperature", 0.0))
        top_p = float(req.get("top_p", 1.0))
        if temperature < 0 or not 0 < top_p <= 1:
            raise ValueError("invalid decoding parameters")
        generation = {"max_new_tokens": cap, "do_sample": temperature > 0,
                      "use_cache": True, "pad_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            generation.update(temperature=temperature, top_p=top_p)
        with self.lock:
            if req.get("seed") is not None:
                seed_all(int(req["seed"]))
            start = time.perf_counter()
            output = self.model.generate(ids, attention_mask=torch.ones_like(ids), **generation)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
        new = output[0, ids.shape[1]:]
        decoded = self.tokenizer.decode(new, skip_special_tokens=True)
        message, finish = parse_response(decoded)
        if len(new) == cap and finish != "tool_calls":
            finish = "length"
        usage = {"prompt_tokens": ids.shape[1], "completion_tokens": len(new),
                 "total_tokens": ids.shape[1]+len(new)}
        if self.log:
            with self.log.open("a") as f:
                f.write(json.dumps({**usage, "elapsed_seconds": elapsed, "request_hash": digest(req),
                    "tool_calls": len(message.get("tool_calls", [])), "finish_reason": finish,
                    "weight_sha256": self.spec["weight_sha256"]})+"\n")
        return {"id": "chatcmpl-"+uuid.uuid4().hex, "object": "chat.completion",
                "created": int(time.time()), "model": "tiny-clm", "usage": usage,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}]}


def create_app(engine):
    from fastapi import FastAPI, HTTPException
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok", "weight_sha256": engine.spec["weight_sha256"],
                "inference_runtime": engine.spec.get("inference_runtime", LEGACY_INFERENCE_RUNTIME)}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "tiny-clm", "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    def complete(request: dict):
        try:
            return engine.complete(request)
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return app


def serve(artifact, device, port, log, seed=0):
    import uvicorn
    # Localhost only: Harbor's host-side agent calls the model, not the container.
    uvicorn.run(create_app(Engine(artifact, device, log, seed)), host="127.0.0.1", port=port, workers=1)
