"""Incremental, read-only JSONL reader with memory limits.

Only one line is held in memory at a time and never more than ``Limits.max_line_bytes`` of it. Problems
(malformed JSON, invalid UTF-8, oversized or non-object lines) become ``RawRecord.problem`` codes; the
offending bytes are discarded and never reach exceptions or messages. Nothing is ever written.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import IO, Any

from agent_traces.model import UnrecognizedReason

DEFAULT_MAX_LINE_BYTES = 16 * 1024 * 1024
MIN_LINE_BYTES = 1024
MAX_LINE_BYTES = 256 * 1024 * 1024  # upper bound for --max-line-bytes: one line must fit in memory
_SKIP_CHUNK = 1024 * 1024


class InputError(Exception):
    """The input cannot be read. Messages are fixed strings: they never include paths or content."""


@dataclass(frozen=True, slots=True)
class Limits:
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES
    sniff_records: int = 50  # records detection examines once some format has strong evidence
    sniff_max_records: int = 1000  # records detection may examine while no format has strong evidence

    def __post_init__(self) -> None:
        if not MIN_LINE_BYTES <= self.max_line_bytes <= MAX_LINE_BYTES:
            raise ValueError("max_line_bytes out of range")
        if not 1 <= self.sniff_records <= self.sniff_max_records:
            raise ValueError("sniff limits out of range")


@dataclass(slots=True)
class RawRecord:
    line: int
    data: dict[str, Any] | None = field(default=None, repr=False)
    problem: UnrecognizedReason | None = None


@dataclass(slots=True)
class ReadStats:
    records: int = 0  # non-blank lines, including the ones that could not be parsed
    blank_lines: int = 0
    bytes_read: int = 0
    problems: dict[str, int] = field(default_factory=dict)
    ends_with_newline: bool = True
    snapshot_bytes: int = 0  # file size when it was opened: only these bytes are read
    grew_during_read: bool = False  # an active session kept writing; later bytes were not read
    shrank_during_read: bool = False  # the file was truncated or replaced while it was read
    incomplete_tail_bytes: int = 0  # unterminated, unparseable final fragment (being written, or cut)
    sha256: str = ""  # only computed on request
    complete: bool = False  # the whole snapshot was read


def _reject_constant(name: str) -> Any:
    raise ValueError("non-standard JSON constant")


def open_readonly(path: str | os.PathLike[str]) -> IO[bytes]:
    """Open a regular file read-only without blocking on FIFOs or devices."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise InputError("input file not found") from None
    except IsADirectoryError:
        raise InputError("input is not a regular file") from None
    except PermissionError:
        raise InputError("input file is not readable (permission denied)") from None
    except (OSError, ValueError):
        raise InputError("input file cannot be opened") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise InputError("input is not a regular file")
        return os.fdopen(fd, "rb", buffering=_SKIP_CHUNK)
    except InputError:
        os.close(fd)
        raise
    except OSError:
        os.close(fd)
        raise InputError("input file cannot be opened") from None


def _readline(handle: IO[bytes], size: int) -> bytes:
    try:
        return handle.readline(size)
    except OSError:
        raise InputError("input file could not be read") from None


def iter_records(
    path: str | os.PathLike[str],
    limits: Limits,
    stats: ReadStats | None = None,
    *,
    compute_hash: bool = False,
) -> Iterator[RawRecord]:
    """Yield the records of the file as it was when opened (a snapshot), one line at a time.

    A session that is still being written may grow while it is read: bytes appended after opening are
    ignored (``grew_during_read``), and an unterminated final fragment that does not parse is reported as
    an incomplete tail instead of a record.
    """
    stats = stats if stats is not None else ReadStats()
    digest = hashlib.sha256() if compute_hash else None
    line_no = 0
    with open_readonly(path) as handle:
        remaining = stats.snapshot_bytes = _size(handle)
        while remaining > 0:
            chunk = _readline(handle, min(limits.max_line_bytes + 1, remaining))
            if not chunk:
                stats.shrank_during_read = True
                break
            remaining -= len(chunk)
            line_no += 1
            if digest is not None:
                digest.update(chunk)
            stats.bytes_read += len(chunk)
            stats.ends_with_newline = chunk.endswith(b"\n")
            if len(chunk) > limits.max_line_bytes and not stats.ends_with_newline:
                # Oversized line: consume the rest (within the snapshot) without keeping it.
                while remaining > 0:
                    rest = _readline(handle, min(_SKIP_CHUNK, remaining))
                    if not rest:
                        stats.shrank_during_read = True
                        break
                    remaining -= len(rest)
                    if digest is not None:
                        digest.update(rest)
                    stats.bytes_read += len(rest)
                    stats.ends_with_newline = rest.endswith(b"\n")
                    if stats.ends_with_newline:
                        break
                record = RawRecord(line_no, problem=UnrecognizedReason.OVERSIZED)
            else:
                record = _parse_line(chunk, line_no)
                if record is None:
                    stats.blank_lines += 1
                    continue
                if record.problem is not None and remaining == 0 and not stats.ends_with_newline:
                    stats.incomplete_tail_bytes = len(chunk)  # still being written, or cut: not a record
                    break
            stats.records += 1
            if record.problem is not None:
                stats.problems[record.problem.value] = stats.problems.get(record.problem.value, 0) + 1
            yield record
        final_size = _size(handle)
        stats.grew_during_read = final_size > stats.snapshot_bytes
        if final_size < stats.snapshot_bytes:
            stats.shrank_during_read = True  # truncated or replaced: what was read may no longer match it
    if digest is not None:
        stats.sha256 = digest.hexdigest()
    stats.complete = not stats.shrank_during_read


def _size(handle: IO[bytes]) -> int:
    try:
        return os.fstat(handle.fileno()).st_size
    except OSError:
        raise InputError("input file could not be read") from None


def _parse_line(chunk: bytes, line_no: int) -> RawRecord | None:
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        return RawRecord(line_no, problem=UnrecognizedReason.INVALID_ENCODING)
    if line_no == 1:
        text = text.removeprefix("\ufeff")
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return RawRecord(line_no, problem=UnrecognizedReason.MALFORMED_JSON)
    if not isinstance(value, dict):
        return RawRecord(line_no, problem=UnrecognizedReason.NOT_AN_OBJECT)
    return RawRecord(line_no, data=value)
