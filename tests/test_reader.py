"""Incremental reader: limits, malformed input and safe error messages."""

from __future__ import annotations

import hashlib
import os

import pytest

from agent_traces.model import UnrecognizedReason
from agent_traces.reader import InputError, Limits, ReadStats, iter_records

LIMITS = Limits(max_line_bytes=1024)


def read(path, limits=LIMITS, compute_hash=True):
    stats = ReadStats()
    records = list(iter_records(path, limits, stats, compute_hash=compute_hash))
    return records, stats


def test_reads_objects_and_skips_blank_lines(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"a": 1}\n\n   \n{"b": 2}\r\n')
    records, stats = read(path)
    assert [(r.line, r.data) for r in records] == [(1, {"a": 1}), (4, {"b": 2})]
    assert stats.records == 2 and stats.blank_lines == 2
    assert stats.complete and stats.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_problem_lines_are_classified_without_keeping_their_bytes(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"ok": 1}\n{broken\n[1, 2]\n"text"\n{"x": NaN}\n\xff\xfe{}\n' + b"[" * 50_000 + b"\n")
    records, stats = read(path, Limits(max_line_bytes=100_000))
    problems = [r.problem for r in records]
    assert problems == [
        None,
        UnrecognizedReason.MALFORMED_JSON,
        UnrecognizedReason.NOT_AN_OBJECT,
        UnrecognizedReason.NOT_AN_OBJECT,
        UnrecognizedReason.MALFORMED_JSON,
        UnrecognizedReason.INVALID_ENCODING,
        UnrecognizedReason.MALFORMED_JSON,  # deep nesting: RecursionError is contained
    ]
    assert all(r.data is None for r in records[1:])
    assert stats.problems == {"malformed_json": 3, "not_an_object": 2, "invalid_encoding": 1}


def test_oversized_line_is_skipped_and_reading_continues(tmp_path):
    path = tmp_path / "t.jsonl"
    big = b'{"blob": "' + b"x" * 5000 + b'"}\n'
    path.write_bytes(b'{"a": 1}\n' + big + b'{"b": 2}\n')
    records, stats = read(path)
    assert [(r.line, r.problem) for r in records] == [
        (1, None),
        (2, UnrecognizedReason.OVERSIZED),
        (3, None),
    ]
    assert records[2].data == {"b": 2}
    assert stats.bytes_read == path.stat().st_size
    assert stats.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_line_exactly_at_the_limit_is_accepted(tmp_path):
    payload = b'{"k": "' + b"y" * (1024 - 10) + b'"}'
    assert len(payload) == 1024 - 1
    path = tmp_path / "t.jsonl"
    path.write_bytes(payload + b"\n")
    records, _ = read(path)
    assert records[0].problem is None


def test_unterminated_unparseable_tail_is_not_a_record(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": "cut')
    records, stats = read(path)
    assert [r.data for r in records] == [{"a": 1}]
    assert stats.records == 1 and stats.incomplete_tail_bytes == len(b'{"b": "cut')
    assert not stats.ends_with_newline and stats.problems == {}


def test_unterminated_but_valid_final_record_is_kept(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": 2}')
    records, stats = read(path)
    assert [r.data for r in records] == [{"a": 1}, {"b": 2}] and stats.incomplete_tail_bytes == 0


def test_terminated_malformed_final_line_is_still_malformed(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"a": 1}\n{"b": "cut\n')
    records, stats = read(path)
    assert records[-1].problem is UnrecognizedReason.MALFORMED_JSON and stats.incomplete_tail_bytes == 0


def test_active_file_is_read_as_a_snapshot(tmp_path):
    """Bytes appended while reading (an active session) are not read; the growth is reported."""
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"n": 1}\n{"n": 2}\n{"n": 3}\n')
    stats = ReadStats()
    records = iter_records(path, LIMITS, stats)
    first = next(records)
    with path.open("ab") as handle:
        handle.write(b'{"n": 4}\n{"n": 5, "half": "wri')
    rest = list(records)
    assert [r.data["n"] for r in [first, *rest]] == [1, 2, 3]
    assert stats.snapshot_bytes == 27 and stats.bytes_read == 27
    assert stats.grew_during_read and not stats.shrank_during_read and stats.complete


def test_file_truncated_while_reading_is_reported(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_bytes(b'{"n": 1}\n' + b'{"n": 2}\n' * 2000)
    stats = ReadStats()
    records = iter_records(path, LIMITS, stats)
    next(records)
    with path.open("r+b") as handle:
        handle.truncate(9)
    list(records)
    assert stats.shrank_during_read and not stats.complete


def test_read_only_access_is_enough(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"a": 1}\n')
    path.chmod(0o400)
    before = path.stat()
    records, _ = read(path)
    assert records[0].data == {"a": 1}
    after = path.stat()
    assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo"])
def test_input_errors_never_echo_the_path(tmp_path, kind):
    secret_dir = tmp_path / "CANARY-private-folder"
    secret_dir.mkdir()
    target = secret_dir / "session.jsonl"
    if kind == "directory":
        target.mkdir()
    elif kind == "fifo":
        os.mkfifo(target)
    with pytest.raises(InputError) as error:
        list(iter_records(target, LIMITS))
    assert "canary" not in str(error.value).lower()
    assert error.value.__cause__ is None


def test_limits_are_validated():
    with pytest.raises(ValueError):
        Limits(max_line_bytes=10)
    with pytest.raises(ValueError):
        Limits(sniff_records=0)
