"""Opt-in inference-side history formatting; never promote prose to an action."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .history_audit import PLAIN_CALL, call_key, digest
from .protocol_screen import sha

CONTROL = "plain-history-control-v1"
NATIVE = "verified-native-history-v1"


def identity(identifier):
    if identifier not in (CONTROL, NATIVE):
        raise ValueError("unknown history protocol")
    return {"id": identifier, "placement": "inference_messages_only",
            "verification": "matching_previously_observed_structured_bash_arguments"}


def rewrite(messages, known, *, enabled):
    """Same terminal-line envelope as the replay; preserve unknown/mutated prose.

    `known` contains command hashes observed in structured historical calls in
    this trial. Nothing is recovered from offloaded files, omitted context, or
    gold answers. Structured fields and the editable mirror remain unchanged.
    """
    result, changed, unverified = copy.deepcopy(messages), [], 0
    for index, message in enumerate(result):
        if index < 2 or message.get("role") != "assistant" or message.get("tool_calls"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        def replace(match):
            nonlocal unverified
            try:
                args = json.loads(match[1])
                key = call_key({"name": "bash", "arguments": args})
            except (ValueError, TypeError, KeyError):
                unverified += 1
                return match[0]
            if key not in known:
                unverified += 1
                return match[0]
            if not enabled:
                return match[0]
            changed.append({"message_index": index, "command_sha256": key})
            return "<tool_call>\n"+json.dumps({"name": "bash", "arguments": args}, ensure_ascii=False)+"\n</tool_call>"
        message["content"] = PLAIN_CALL.sub(replace, content)
    return result, changed, unverified


class HistoryFormatMixin:
    history_protocol_id = CONTROL

    async def run(self, instruction, environment, context):
        self._history_known = set()
        self._history_requests = 0
        self._history_path = Path(self.logs_dir)/"history_format.jsonl"
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("x"):
            pass
        return await super().run(instruction, environment, context)

    async def _query_with_retry(self, model, messages):
        # These are prior parsed calls in the actual episode, not newly inferred
        # commands. Retain only fingerprints, never restore removed content.
        for message in messages[2:]:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    try:
                        self._history_known.add(call_key(call["function"]))
                    except (ValueError, TypeError, KeyError):
                        continue
        outgoing, changed, unverified = rewrite(messages, self._history_known,
            enabled=self.history_protocol_id == NATIVE)
        self._history_requests += 1
        record = {"history_protocol": identity(self.history_protocol_id),
            "request": self._history_requests, "before_sha256": digest(messages),
            "after_sha256": digest(outgoing), "protected_before_sha256": digest(messages[:2]),
            "protected_after_sha256": digest(outgoing[:2]), "changed": changed,
            "unverified_plain_calls": unverified,
            "noncontent_fields_equal": [{k: v for k, v in m.items() if k != "content"} for m in messages] ==
                                       [{k: v for k, v in m.items() if k != "content"} for m in outgoing]}
        with self._history_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True)+"\n")
        return await super()._query_with_retry(model, outgoing)


class NativeHistoryMixin(HistoryFormatMixin):
    history_protocol_id = NATIVE


def audit_log(path, identifier):
    protocol = identity(identifier)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("missing history application requests")
    for index, row in enumerate(rows, 1):
        hashes = [row.get(k) for k in ("before_sha256", "after_sha256", "protected_before_sha256", "protected_after_sha256")]
        if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes):
            raise ValueError("missing or invalid history message fingerprints")
        if (row.get("request") != index or row.get("history_protocol") != protocol or
                row.get("noncontent_fields_equal") is not True or
                not isinstance(row.get("changed"), list) or not isinstance(row.get("unverified_plain_calls"), int) or
                row["unverified_plain_calls"] < 0 or
                row.get("protected_before_sha256") != row.get("protected_after_sha256") or
                (identifier == CONTROL and (row["changed"] or row["before_sha256"] != row["after_sha256"])) or
                (not row["changed"] and row["before_sha256"] != row["after_sha256"])):
            raise ValueError("inconsistent history application log")
    return {"history_protocol": protocol, "requests": len(rows),
            "changed_historical_calls": sum(len(r["changed"]) for r in rows),
            "unverified_plain_calls": sum(r["unverified_plain_calls"] for r in rows),
            "sha256": sha(path)}
