import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys
import urllib.request

import pytest

spec = importlib.util.spec_from_file_location("check_client_protocol",
    Path(__file__).resolve().parents[1]/"check_client_protocol.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def fixture():
    return {"model": "tiny-clm", "messages": [{"role": "user", "content": "fixture"}],
            "tools": [{"type": "function", "function": {"name": "bash"}}],
            "max_tokens": 1024, "temperature": 0.0, "top_p": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}}


def test_real_loopback_round_trip_and_dropped_tools_rejected():
    body = fixture()
    with probe.capture_server() as (url, received):
        request = urllib.request.Request(url+"/chat/completions", json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=5) as response:
            reply = json.load(response)["choices"][0]["message"]
        probe.verify_exchange(received, reply, body["tools"], body["messages"])
        del received[0]["tools"]
        with pytest.raises(AssertionError, match="preserve tools"):
            probe.verify_exchange(received, reply, body["tools"], body["messages"])


def test_text_only_reply_is_not_accepted_as_executed_action():
    body = fixture()
    with pytest.raises(AssertionError, match="reply was not preserved"):
        probe.verify_exchange([body], {"role": "assistant", "content": "bash\necho PROTOCOL_FIXTURE"},
                              body["tools"], body["messages"])


def test_installed_clm_client_round_trip():
    if any(importlib.util.find_spec(name) is None for name in ("litellm", "harbor")):
        pytest.skip("Full CLM SDK check requires the installed harness environment")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(root/"clm")+os.pathsep+os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, str(root/"compression/check_client_protocol.py")],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "CLM_CLIENT_PROTOCOL_PASSED" in result.stdout
