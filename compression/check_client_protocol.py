#!/usr/bin/env python3
"""Check the real CLM client against a synthetic localhost server; no model or GPU."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import json
import os
import threading
from types import SimpleNamespace


@contextmanager
def capture_server():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            result = {"id": "chatcmpl-protocol-fixture", "object": "chat.completion",
                "created": 0, "model": "tiny-clm", "choices": [{"index": 0,
                "finish_reason": "tool_calls", "message": {"role": "assistant", "content": None,
                "tool_calls": [{"id": "call_protocol_fixture", "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"echo PROTOCOL_FIXTURE"}'}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            payload = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def verify_exchange(received, message, tools, messages):
    if len(received) != 1:
        raise AssertionError(f"Expected one request, observed {len(received)}")
    body = received[0]
    expected = {"model": "tiny-clm", "messages": messages, "tools": tools,
                "max_tokens": 1024, "temperature": 0.0, "top_p": 1.0,
                "chat_template_kwargs": {"enable_thinking": False}}
    for key, value in expected.items():
        if body.get(key) != value:
            raise AssertionError(f"CLM client did not preserve {key}: {body.get(key)!r}")
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "bash":
        raise AssertionError("Synthetic structured bash reply was not preserved")
    if json.loads(calls[0]["function"]["arguments"]) != {"command": "echo PROTOCOL_FIXTURE"}:
        raise AssertionError("Synthetic command arguments were changed")


def main():
    # The only completion endpoint is the synthetic loopback server. No credentials
    # or model download are required; the command text in its reply is never executed.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    from clm_harness.clm_agent.harness import ClmAgent, _extract_tool_call
    from clm_harness.utils.tool_schemas import BASH_TOOL

    messages = [{"role": "system", "content": "Protocol fixture; return one bash tool call."},
                {"role": "user", "content": "Return the fixed fixture command."}]
    with capture_server() as (url, received):
        # Invoke the original method without constructing a task container or context
        # manager. These are precisely the attributes it uses in the primary setup.
        agent = SimpleNamespace(api_base=url, max_tokens=1024, temperature=0.0,
                                top_p=1.0, enable_thinking=False, send_chat_template_kwargs=True)
        response = asyncio.run(ClmAgent._query_with_retry(agent, "openai/tiny-clm", messages))
        message = response.choices[0].message.model_dump()
        verify_exchange(received, message, [BASH_TOOL], messages)
        action = _extract_tool_call(message)
        if action.get("type") != "command" or action.get("command") != "echo PROTOCOL_FIXTURE":
            raise AssertionError("Original CLM action extractor rejected the structured fixture")
    print("CLM_CLIENT_PROTOCOL_PASSED")
    print(json.dumps({"packages": {name: importlib.metadata.version(name)
                                  for name in ("litellm", "openai", "harbor")},
        "request_tools_preserved": True, "decoding_settings_preserved": True,
        "structured_response_preserved": True, "original_action_extractor_passed": True,
        "scope": "synthetic client round trip only; no model inference, command execution or task-quality claim"}, indent=2))


if __name__ == "__main__":
    main()
