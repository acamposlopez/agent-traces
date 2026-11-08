"""Registry of source adapters and known unsupported formats.

Adding a format = writing an adapter module and listing it in DEFAULT_ADAPTERS. Formats that are only
recognised (so that detection reports them instead of mistaking them) go in KNOWN_UNSUPPORTED.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_traces.adapters.base import FormatRecognizer, Signal, SourceAdapter
from agent_traces.adapters.claude_agent_sdk import ClaudeAgentSdkStream
from agent_traces.adapters.claude_code import ClaudeCodeAdapter
from agent_traces.adapters.cursor_cli import CursorCliAdapter

DEFAULT_ADAPTERS: tuple[SourceAdapter, ...] = (ClaudeCodeAdapter(), CursorCliAdapter())
KNOWN_UNSUPPORTED: tuple[FormatRecognizer, ...] = (ClaudeAgentSdkStream(),)


def get_adapter(name: str, adapters: Sequence[SourceAdapter] = DEFAULT_ADAPTERS) -> SourceAdapter:
    for adapter in adapters:
        if adapter.name == name:
            return adapter
    raise KeyError(name)


__all__ = ["DEFAULT_ADAPTERS", "KNOWN_UNSUPPORTED", "FormatRecognizer", "Signal", "SourceAdapter", "get_adapter"]
