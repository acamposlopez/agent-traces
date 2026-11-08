"""Recognizer (not an adapter) for the streaming output of the Claude Agent SDK / ``claude -p
--output-format stream-json``.

That format resembles Cursor's stream-json, so it must be recognised and reported as unsupported instead
of being mistaken for another format. Signatures follow the message types of the official Agent SDK
reference (code.claude.com/docs/en/agent-sdk/typescript and /python): assistant/user messages with ``uuid``,
``session_id`` and ``parent_tool_use_id``; ``result`` with ``num_turns`` / ``total_cost_usd``; ``system`` init
with ``tools`` plus SDK-only fields (``claude_code_version``, ``slash_commands``, ``output_style``). Nothing is
parsed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_traces.adapters.base import FormatRecognizer, Signal, Source, Vocabulary, text_value

_D = Source.DOCUMENTED


class ClaudeAgentSdkStream(FormatRecognizer):
    name = "claude-agent-sdk-stream"
    display_name = "Claude Agent SDK stream-json output"
    supported = False
    vocabulary = Vocabulary(
        record_types={"assistant": _D, "user": _D, "result": _D, "system": _D},
        discriminators={
            name: _D
            for name in ("init", "compact_boundary", "informational", "success", "error_max_turns",
                         "error_during_execution", "error_max_budget_usd", "error_max_structured_output_retries")
        },
        fields={
            name: _D
            for name in (
                "type", "subtype", "uuid", "session_id", "parent_tool_use_id", "message", "error", "agent_id",
                "timestamp", "isSynthetic", "tool_use_result", "priority", "origin", "num_turns", "total_cost_usd",
                "duration_ms", "duration_api_ms", "is_error", "result", "stop_reason", "usage", "modelUsage",
                "permission_denials", "errors", "structured_output", "terminal_reason", "tools", "mcp_servers",
                "model", "cwd", "apiKeySource", "permissionMode", "claude_code_version", "slash_commands",
                "output_style", "skills", "plugins", "agents", "betas", "effort", "capabilities", "compact_metadata",
                "content", "level",
            )
        },
    )

    def sniff(self, record: Mapping[str, Any]) -> Signal:
        rtype = text_value(record.get("type"))
        if not (isinstance(record.get("session_id"), str) and isinstance(record.get("uuid"), str)):
            return Signal.NONE
        message = record.get("message")
        if rtype in ("assistant", "user") and "parent_tool_use_id" in record and isinstance(message, dict):
            return Signal.STRONG
        if rtype == "result" and ("num_turns" in record or "total_cost_usd" in record) and "is_error" in record:
            return Signal.STRONG
        if (
            rtype == "system"
            and record.get("subtype") == "init"
            and isinstance(record.get("tools"), list)
            and any(key in record for key in ("claude_code_version", "slash_commands", "output_style"))
        ):
            return Signal.STRONG
        return Signal.NONE
