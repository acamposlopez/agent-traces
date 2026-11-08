"""Reviewed structural vocabulary of Claude Code transcripts, with provenance (consulted 2026-11-07).

- documented (D): official Anthropic documentation. The transcript format itself is officially "internal to
  Claude Code and changes between versions" (code.claude.com/docs/en/sessions), so only names documented
  elsewhere qualify: the Messages API (platform.claude.com/docs/en/api/messages: message, content blocks,
  usage, stop reasons) and the official tool list (code.claude.com/docs/en/tools-reference; the Task -> Agent
  rename in docs/en/sub-agents).
- observed (O): public third-party analyses of real transcripts, i.e. experimental observation, not a
  contract: claude-code-viewer (conversation-schema), claude-code-transcripts (types.rs), claude-code-log
  (models.py), claude-session-types, mekara, lin-guanguo's schema notes, supercode-harness, agent-top-core.

Only names listed here may appear in schema-discovery output; any other name becomes an opaque local code.
Every entry is a structural name of the format; nothing here comes from a real session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_traces.adapters.base import Source, Vocabulary
from agent_traces.model import MessageOrigin, ToolCategory

D, O = Source.DOCUMENTED, Source.OBSERVED


@dataclass(frozen=True)
class MetadataSpec:
    metadata_class: str
    expected: frozenset[str] = frozenset()  # observed structural fields required to call a record verified
    discriminator: tuple[str, ...] | None = None  # path of a value that refines the type (vocabulary-gated)


def _spec(metadata_class: str, *expected: str, discriminator: tuple[str, ...] | None = None) -> MetadataSpec:
    return MetadataSpec(metadata_class, frozenset(expected), discriminator)


# Metadata record types (all observed): identified, content not interpreted, never exportable by default.
# An empty field set means the type is known by name only and is never reported as "verified".
METADATA_SPECS: Mapping[str, MetadataSpec] = {
    "summary": _spec("title", "summary", "leafUuid"),
    "ai-title": _spec("title", "aiTitle"),
    "custom-title": _spec("title", "customTitle"),
    "agent-name": _spec("title", "agentName"),
    "tag": _spec("title"),
    "last-prompt": _spec("prompt", "lastPrompt"),
    "file-history-snapshot": _spec("file_history", "messageId", "snapshot", "isSnapshotUpdate"),
    "file-history-delta": _spec("file_history", "messageId", "trackingPath"),
    "queue-operation": _spec("queue", "operation", "timestamp", discriminator=("operation",)),
    "attachment": _spec("attachment", "attachment", discriminator=("attachment", "type")),
    "progress": _spec("progress", "data", discriminator=("data", "type")),
    "permission-mode": _spec("config", "permissionMode"),
    "mode": _spec("config", "mode"),
    "agent-color": _spec("config"),
    "agent-setting": _spec("config"),
    "pr-link": _spec("link", "prNumber", "prUrl"),
    "frame-link": _spec("link", "frameUrl"),
    "worktree-state": _spec("workspace", "worktreeSession"),
    "fork-context-ref": _spec("relation", "parentSessionId", "parentLastUuid"),
    "started": _spec("lifecycle"),
    "result": _spec("lifecycle"),
    "task-summary": _spec("summary"),
    "cost-state": _spec("cost", "totalCostUSD"),
    "content-replacement": _spec("internal"),
    "attribution-snapshot": _spec("internal"),
    "marble-origami-commit": _spec("internal"),
    "marble-origami-snapshot": _spec("internal"),
    "speculation-accept": _spec("internal"),
    "atis-latch": _spec("internal"),
    "bridge-session": _spec("internal"),
    "artifact-autoreact-ledger": _spec("internal"),
    "artifact-comment-monitor": _spec("internal"),
}

# Subtypes of "system" records interpreted as operational metadata (api_error is an ErrorEvent instead).
SYSTEM_SPECS: Mapping[str, MetadataSpec] = {
    "compact_boundary": _spec("compaction", "compactMetadata"),
    "microcompact_boundary": _spec("compaction"),
    "local_command": _spec("command", "content"),
    "stop_hook_summary": _spec("hooks", "hookCount"),
    "turn_duration": _spec("timing", "durationMs"),
    "away_summary": _spec("summary", "content"),
    "informational": _spec("notice", "content"),
    "bridge_status": _spec("bridge", "url"),
    "scheduled_task_fire": _spec("scheduling", "cronKind"),
    "permission_retry": _spec("permissions"),
    "agents_killed": _spec("agents"),
    "init": _spec("lifecycle"),
}

PROGRESS_TYPES = ("hook_progress", "bash_progress", "agent_progress", "query_update", "search_results_received",
                  "waiting_for_task")
ATTACHMENT_TYPES = (
    "hook_success", "hook_non_blocking_error", "hook_blocking_error", "hook_cancelled", "hook_additional_context",
    "hook_permission_decision", "hook_stopped_continuation", "hook_system_message", "file", "edited_text_file",
    "directory", "compact_file_reference", "nested_memory", "plan_mode", "plan_mode_exit", "plan_file_reference",
    "auto_mode", "auto_mode_exit", "skill_listing", "dynamic_skill", "invoked_skills", "task_reminder",
    "todo_reminder", "diagnostics", "date_change", "deferred_tools_delta", "mcp_instructions_delta",
    "agent_listing_delta", "command_permissions", "ultrathink_effort", "queued_command", "total_tokens_reminder",
    "read_truncation_notice", "goal_status", "task_status", "companion_intro",
)
QUEUE_OPERATIONS = ("enqueue", "dequeue", "remove", "popAll")

# Harness markup at the start of a user-role text (observed in public transcript parsers). The first match
# decides the origin; text without any marker is reported as "prompt".
ORIGIN_MARKERS: tuple[tuple[str, MessageOrigin], ...] = (
    ("<command-name>", MessageOrigin.COMMAND),
    ("<command-message>", MessageOrigin.COMMAND),
    ("<command-args>", MessageOrigin.COMMAND),
    ("<bash-input>", MessageOrigin.COMMAND),
    ("<local-command-stdout>", MessageOrigin.COMMAND_OUTPUT),
    ("<local-command-stderr>", MessageOrigin.COMMAND_OUTPUT),
    ("<bash-stdout>", MessageOrigin.COMMAND_OUTPUT),
    ("<bash-stderr>", MessageOrigin.COMMAND_OUTPUT),
    ("<local-command-caveat>", MessageOrigin.INJECTED),
    ("<system-reminder>", MessageOrigin.INJECTED),
    ("[Request interrupted by user", MessageOrigin.INTERRUPTION),
)

# Built-in tools: current official list (tools-reference), plus older names still found in transcripts
# (Task before its rename to Agent, documented in sub-agents; the rest observed by public parsers).
KNOWN_TOOLS: Mapping[str, ToolCategory] = {
    # documented
    "Agent": ToolCategory.SUBAGENT,
    "Artifact": ToolCategory.OTHER,
    "AskUserQuestion": ToolCategory.USER_INTERACTION,
    "Bash": ToolCategory.SHELL,
    "CronCreate": ToolCategory.OTHER,
    "CronDelete": ToolCategory.OTHER,
    "CronList": ToolCategory.OTHER,
    "Edit": ToolCategory.FILE_WRITE,
    "EndConversation": ToolCategory.OTHER,
    "EnterPlanMode": ToolCategory.PLANNING,
    "EnterWorktree": ToolCategory.OTHER,
    "ExitPlanMode": ToolCategory.PLANNING,
    "ExitWorktree": ToolCategory.OTHER,
    "Glob": ToolCategory.SEARCH,
    "Grep": ToolCategory.SEARCH,
    "ListAgents": ToolCategory.SUBAGENT,
    "ListMcpResourcesTool": ToolCategory.MCP,
    "LSP": ToolCategory.SEARCH,
    "Monitor": ToolCategory.SHELL,
    "NotebookEdit": ToolCategory.FILE_WRITE,
    "PowerShell": ToolCategory.SHELL,
    "PushNotification": ToolCategory.USER_INTERACTION,
    "Read": ToolCategory.FILE_READ,
    "ReadMcpResourceTool": ToolCategory.MCP,
    "RemoteTrigger": ToolCategory.OTHER,
    "ReportFindings": ToolCategory.OTHER,
    "ScheduleWakeup": ToolCategory.OTHER,
    "SendFeedback": ToolCategory.USER_INTERACTION,
    "SendMessage": ToolCategory.SUBAGENT,
    "SendUserFile": ToolCategory.USER_INTERACTION,
    "ShareOnboardingGuide": ToolCategory.USER_INTERACTION,
    "Skill": ToolCategory.OTHER,
    "SubagentHandback": ToolCategory.SUBAGENT,
    "TaskCreate": ToolCategory.PLANNING,
    "TaskGet": ToolCategory.PLANNING,
    "TaskList": ToolCategory.PLANNING,
    "TaskOutput": ToolCategory.PLANNING,
    "TaskStop": ToolCategory.PLANNING,
    "TaskUpdate": ToolCategory.PLANNING,
    "TodoWrite": ToolCategory.PLANNING,
    "ToolSearch": ToolCategory.OTHER,
    "WaitForMcpServers": ToolCategory.MCP,
    "WebFetch": ToolCategory.WEB,
    "WebSearch": ToolCategory.WEB,
    "Workflow": ToolCategory.SUBAGENT,
    "Write": ToolCategory.FILE_WRITE,
    "Task": ToolCategory.SUBAGENT,  # pre-2.1.63 name of Agent (documented rename)
    # observed in older transcripts
    "MultiEdit": ToolCategory.FILE_WRITE,
    "BashOutput": ToolCategory.SHELL,
    "KillShell": ToolCategory.SHELL,
    "KillBash": ToolCategory.SHELL,
    "LS": ToolCategory.SEARCH,
    "NotebookRead": ToolCategory.FILE_READ,
    "SlashCommand": ToolCategory.OTHER,
}


def _names(source: Source, *names: str) -> dict[str, Source]:
    return {name: source for name in names}


VOCABULARY = Vocabulary(
    record_types={**_names(O, "user", "assistant", "system"), **_names(O, *METADATA_SPECS)},
    discriminators={
        **_names(O, *SYSTEM_SPECS, "api_error"),
        **_names(O, *QUEUE_OPERATIONS, *PROGRESS_TYPES, *ATTACHMENT_TYPES),
    },
    discriminator_paths=(("subtype",), ("operation",), ("data", "type"), ("attachment", "type")),
    fields={
        # Messages API (documented): message, content blocks, usage.
        **_names(
            D, "id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "stop_details", "usage",
            "container", "diagnostics", "text", "citations", "thinking", "signature", "data", "name", "input",
            "caller", "tool_use_id", "is_error", "source", "file_id", "input_tokens", "output_tokens",
            "cache_creation_input_tokens", "cache_read_input_tokens", "cache_creation", "ephemeral_5m_input_tokens",
            "ephemeral_1h_input_tokens", "server_tool_use", "web_search_requests", "web_fetch_requests",
            "service_tier", "inference_geo", "speed", "output_tokens_details", "thinking_tokens",
        ),
        # Record envelope (observed), including the snake_case session_id written by newer clients.
        **_names(
            O, "uuid", "parentUuid", "logicalParentUuid", "isSidechain", "sessionId", "session_id", "timestamp",
            "userType", "entrypoint", "cwd", "version", "gitBranch", "slug", "agentId", "teamName", "promptId",
            "isMeta", "forkedFrom", "messageUuid", "sessionKind", "message", "subtype", "level", "toolUseResult",
            "sourceToolAssistantUUID", "sourceToolUseID", "permissionMode", "origin", "isCompactSummary",
            "isVisibleInTranscriptOnly", "imagePasteIds", "thinkingMetadata", "todos", "requestId",
            "isApiErrorMessage", "error", "apiErrorStatus", "quotaLimits",
        ),
        # Metadata records and system subtypes (observed).
        **_names(
            O, "operation", "messageId", "snapshot", "isSnapshotUpdate", "trackedFileBackups", "backupFileName",
            "backupTime", "realParentDir", "summary", "leafUuid", "toolUseID", "parentToolUseID", "attachment",
            "lastPrompt", "customTitle", "aiTitle", "agentName", "mode", "snapshotMessageId", "trackingPath",
            "backup", "worktreeSession", "originalCwd", "worktreePath", "worktreeName", "worktreeBranch",
            "prNumber", "prUrl", "prRepository", "path", "frameUrl", "title", "artifactCount", "parentSessionId",
            "parentLastUuid", "contextLength", "totalCostUSD", "hasUnknownModelCost", "compactMetadata", "trigger",
            "preTokens", "postTokens", "durationMs", "preservedSegment", "preservedMessages", "hookCount",
            "hookInfos", "hookErrors", "preventedContinuation", "stopReason", "hasOutput", "messageCount", "status",
            "headers", "requestID", "retryInMs", "retryAttempt", "maxRetries", "url", "cronKind",
            # progress data and attachment payloads
            "hookEvent", "hookName", "command", "output", "fullOutput", "elapsedTimeSeconds", "totalLines",
            "prompt", "query", "taskDescription", "taskType", "normalizedMessages", "serverName", "toolName",
            "elapsedTimeMs", "taskId", "stdout", "stderr", "exitCode", "commandMode", "source_uuid",
        ),
    },
    block_types=_names(
        D, "text", "thinking", "redacted_thinking", "tool_use", "tool_result", "image", "document", "server_tool_use",
        "web_search_tool_result", "web_fetch_tool_result", "code_execution_tool_result",
        "bash_code_execution_tool_result", "text_editor_code_execution_tool_result", "tool_search_tool_result",
        "container_upload", "search_result", "tool_reference",
    ),
    containers=frozenset(
        {"message", "data", "attachment", "snapshot", "compactMetadata", "error", "backup", "worktreeSession",
         "forkedFrom", "usage"}
    ),
)
