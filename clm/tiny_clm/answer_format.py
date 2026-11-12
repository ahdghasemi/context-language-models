"""A single declared prompt intervention; no command repair or answer synthesis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT_ID = "kv-answer-format-v1"
FORMAT_TEXT = """## GET answer-format reminder

For every GET, the retrieved value must be printed inside this exact structure,
with KEY replaced by the queried key and VALUE replaced by the retrieved value:

<<<ANSWER key=KEY>>>
VALUE
<<<ANSWER END>>>

A bare value is not an answer block. Print the complete block before running
echo READY_FOR_NEXT_OP. If you use shell variables named key and value, the
printing statement is:

printf '<<<ANSWER key=%s>>>\\n%s\\n<<<ANSWER END>>>\\n' "$key" "$value"

Describing an intention to print the block does not emit it.
"""

TOOL_FORMAT_ID = "kv-answer-and-tool-format-v1"
TOOL_FORMAT_TEXT = """## Bash tool-call format reminder

To execute an action, use the supplied bash function tool. After any required
THOUGHT, serialize exactly one tool call in this form:

<tool_call>
{"name":"bash","arguments":{"command":"pwd"}}
</tool_call>

The example command pwd only illustrates the calling format. Replace it with
the command you intend to execute, using a valid JSON string for command.
Include the complete JSON object and closing </tool_call> tag.
A Markdown code block, a bare shell command, or text such as
bash {"command":"pwd"} does not invoke the tool.
Wait for the tool result before treating an action as executed.
"""


def protocol_text(identifier=FORMAT_ID):
    if identifier == FORMAT_ID:
        return FORMAT_TEXT
    if identifier == TOOL_FORMAT_ID:
        return FORMAT_TEXT+"\n"+TOOL_FORMAT_TEXT
    raise ValueError("unknown diagnostic prompt protocol")


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def protocol_identity(identifier=None):
    if identifier is None:
        return None
    return {"id": identifier, "suffix_sha256": text_sha(protocol_text(identifier)),
            "placement": "instruction_suffix_once"}


def add_reminder(instruction, identifier=FORMAT_ID):
    if not instruction.lstrip().startswith("# Key-Value Store"):
        raise ValueError("answer-format diagnostic supports only KV tasks")
    if "## GET answer-format reminder" in instruction or "## Bash tool-call format reminder" in instruction:
        raise ValueError("answer-format reminder was already applied")
    return instruction.rstrip()+"\n\n"+protocol_text(identifier)


class FormatReminderMixin:
    """Only wrap initial instructions; delegate every model/tool operation."""

    diagnostic_protocol_id = FORMAT_ID

    async def run(self, instruction, environment, context):
        modified = add_reminder(instruction, self.diagnostic_protocol_id)
        path = Path(self.logs_dir)/"answer_format_protocol.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        application = {"diagnostic_protocol": protocol_identity(self.diagnostic_protocol_id),
                       "original_instruction_sha256": text_sha(instruction),
                       "modified_instruction_sha256": text_sha(modified)}
        # A new trial is required; never overwrite prior intervention evidence.
        with path.open("x") as handle:
            json.dump(application, handle, indent=2)
        return await super().run(modified, environment, context)


class ToolFormatReminderMixin(FormatReminderMixin):
    diagnostic_protocol_id = TOOL_FORMAT_ID


def application_valid(application, identity):
    return bool(isinstance(application, dict) and application.get("diagnostic_protocol") == identity
        and all(isinstance(application.get(k), str) and len(application[k]) == 64
                and all(c in "0123456789abcdef" for c in application[k])
                for k in ("original_instruction_sha256", "modified_instruction_sha256"))
        and application["original_instruction_sha256"] != application["modified_instruction_sha256"])


def agent_path(mode, identifier=None):
    protocol_identity(identifier)  # Unknown variants must fail, never fall back.
    if identifier is None:
        return {"clm": "clm_contextbench:ClmStream",
                "append_only": "contextbench.harbor.agents.base:ContextBenchAgent"}[mode]
    if identifier == TOOL_FORMAT_ID:
        return {"clm": "tiny_clm.contextbench_protocol:ToolFormatReminderClm",
                "append_only": "tiny_clm.contextbench_protocol:ToolFormatReminderAppendOnly"}[mode]
    return {"clm": "tiny_clm.contextbench_protocol:FormatReminderClm",
            "append_only": "tiny_clm.contextbench_protocol:FormatReminderAppendOnly"}[mode]
