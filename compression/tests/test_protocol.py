import pytest
from tiny_clm.serve import parse_response
from clm_harness.context_utils.context_string import parse_back, render_editable


def test_native_qwen_tool_parse_with_reasoning():
    message, finish = parse_response('<think>check</think><tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>')
    assert finish == "tool_calls"
    assert message["reasoning_content"] == "check"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"command": "pwd"}'


@pytest.mark.parametrize("text", ['<tool_call>not json</tool_call>', '<tool_call>{"name": "bash"}</tool_call>'])
def test_invalid_calls_are_not_repaired(text):
    message, finish = parse_response(text)
    assert finish == "stop" and "tool_calls" not in message


def test_original_context_file_contract_preserved():
    messages = [{"role":"system","content":"system"}, {"role":"user","content":"task"},
                {"role":"user","content":"state A"}, {"role":"assistant","content":"observed"}]
    rendered = render_editable(messages)
    edited = rendered.replace("state A", "state B")
    restored = parse_back(edited, messages[:2])
    assert restored[:2] == messages[:2]
    assert "state B" in str(restored) and "state A" not in str(restored)


def test_reference_http_server_preserves_tools_and_usage(tmp_path):
    from fastapi.testclient import TestClient
    from tiny_clm.serve import Engine, create_app
    from tiny_clm.model import export_model
    from tiny_clm.smoke import prepare_tiny
    from tiny_clm.lowrank import replace, select
    from tiny_clm.model import tokenizer_from
    model, cfg, _ = prepare_tiny(tmp_path/"assets")
    ranks = select(model, cfg["selection"])
    replace(model, ranks)
    export_model(model, tmp_path/"export", ranks, tokenizer=tokenizer_from(cfg["assets"]["model_path"]), dtype="float32")
    client = TestClient(create_app(Engine(tmp_path/"export")))
    assert client.get("/health").status_code == 200
    request = {"model":"tiny-clm", "messages":[{"role":"user","content":"w4 w5"}], "max_tokens":2,"temperature":0}
    response = client.post("/v1/chat/completions", json=request)
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens"] == 2
    assert response.json()["usage"]["completion_tokens"] == 2
    assert client.post("/v1/chat/completions", json={**request,"stream":True}).status_code == 400
    assert client.post("/v1/chat/completions", json={**request,"max_tokens":1000}).status_code == 400
