"""Claude Code metadata records: identified structurally, never over-claimed, never exportable by default."""

from __future__ import annotations

import json

from agent_traces import cli
from agent_traces.model import ExportPolicy, MetadataStructure, Notice, export_policy
from conftest import fixture, report_of, trace_of

PATH = "claude-code/metadata-records.jsonl"


def test_metadata_types_classes_and_structure():
    notices = [e for e in trace_of(PATH).events if isinstance(e, Notice)]
    name_only = sorted(e.notice_type for e in notices if e.structure is MetadataStructure.NAME_ONLY)
    assert name_only == ["attachment/other", "system/microcompact_boundary", "task-summary"]
    by_type = {e.notice_type: e.metadata_class for e in notices}
    assert by_type["progress/hook_progress"] == by_type["progress/bash_progress"] == "progress"
    assert by_type["attachment/hook_additional_context"] == by_type["attachment/queued_command"] == "attachment"
    assert by_type["last-prompt"] == "prompt" and by_type["permission-mode"] == by_type["mode"] == "config"
    assert {export_policy(e) for e in notices} == {ExportPolicy.NOT_BY_DEFAULT}


def test_metadata_coverage_and_origins():
    report = report_of(PATH)
    coverage = report.coverage
    assert [coverage[k]["records"] for k in ("semantic", "metadata_verified", "metadata_name_only", "unknown")] == [
        9, 20, 3, 1]
    assert coverage["structurally_recognized_percent"] == 97.0
    assert report.counts["user_messages_by_origin"] == {
        "prompt": 1, "command": 1, "command_output": 1, "interruption": 1, "injected": 2, "unknown": 0,
    }
    assert report.records["export_policy"] == {"after_scrubbing": 7, "not_by_default": 26, "never": 1}
    assert report.fields["out_of_order"] == {"conversation": 0, "metadata": 1}


def test_discovery_describes_metadata_with_vocabulary_names_only(capsys):
    code = cli.main(["inspect", str(fixture(PATH)), "--discover-schema", "--output", "json"])
    discovery = json.loads(capsys.readouterr().out)["discovery"]
    assert code == 0
    groups = {(g["category"], g["type"], (g["discriminator"] or {}).get("value")): g for g in discovery["groups"]}
    hook = groups[("metadata", "attachment", "hook_additional_context")]
    assert sorted(hook["nested"]["attachment"]) == ["content", "hookEvent", "hookName", "type"]
    unknown = groups[("unknown", "T1", None)]
    assert "session_id" in unknown["fields"] and "K1" in unknown["fields"]  # known field shown, unknown coded
    future = groups[("metadata", "attachment", "S1")]  # attachment type outside the vocabulary
    assert future["records"] == 1
    assert "canary" not in json.dumps(discovery).lower()
