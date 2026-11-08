"""Command-line entry point.

Messages never include paths, field values or exception texts: a trace may contain secrets anywhere, and
the arguments themselves may contain private paths. Exit codes: 0 report produced, 1 input error, 2 usage
error, 3 format not determinable, 4 internal error, 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections.abc import Sequence
from typing import NoReturn

from agent_traces import __version__
from agent_traces.adapters import DEFAULT_ADAPTERS, SourceAdapter
from agent_traces.detection import DetectionStatus
from agent_traces.engine import AUTO, FormatError, run
from agent_traces.processors import DiscoveryProcessor, InspectionProcessor, render_discovery, render_text
from agent_traces.reader import DEFAULT_MAX_LINE_BYTES, MAX_LINE_BYTES, MIN_LINE_BYTES, InputError, Limits

EXIT_OK, EXIT_INPUT, EXIT_USAGE, EXIT_FORMAT, EXIT_INTERNAL, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130


class _ArgumentParser(argparse.ArgumentParser):
    """argparse repeats offending values (paths included) in its errors; this parser never does."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {_redact_usage_error(message)}\n")


def _redact_usage_error(message: str) -> str:
    if message.startswith("unrecognized arguments"):
        return "unrecognized arguments (values not shown)"
    message = re.sub(r"invalid choice: .*? \(choose from", "invalid choice (choose from", message, flags=re.DOTALL)
    return re.sub(r"(invalid \S+ value): .*", r"\1", message, flags=re.DOTALL)


def _line_bytes(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if not MIN_LINE_BYTES <= number <= MAX_LINE_BYTES:
        raise argparse.ArgumentTypeError(f"must be between {MIN_LINE_BYTES} and {MAX_LINE_BYTES}")
    return number


def build_parser(adapters: Sequence[SourceAdapter] = DEFAULT_ADAPTERS) -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="agent-traces",
        description="Local, offline tools for AI agent traces. Nothing is uploaded or written to disk.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    inspect = commands.add_parser(
        "inspect",
        help="structural report of a trace file (never shows its content)",
        description=(
            "Read a trace file (read-only, incrementally) and report its structure and integrity: format, "
            "records, messages, tool calls and results, errors, unknown records and limitations. "
            "Prompts, responses, arguments, outputs, paths and identifiers are never printed."
        ),
    )
    inspect.add_argument("file", metavar="FILE", help="trace file (JSONL / NDJSON)")
    inspect.add_argument(
        "--format",
        default=AUTO,
        choices=[AUTO, *(adapter.name for adapter in adapters)],
        help="source format (default: auto, which refuses to guess when the evidence is ambiguous)",
    )
    inspect.add_argument("--output", choices=["text", "json"], default="text", help="report format")
    inspect.add_argument(
        "--max-line-bytes",
        type=_line_bytes,
        default=DEFAULT_MAX_LINE_BYTES,
        help=f"skip records longer than this (default: {DEFAULT_MAX_LINE_BYTES}, max: {MAX_LINE_BYTES})",
    )
    inspect.add_argument(
        "--include-file-hash",
        action="store_true",
        help="add the SHA-256 of the input file to the report (omitted by default)",
    )
    inspect.add_argument(
        "--include-models",
        action="store_true",
        help="show model identifiers that match a known safe pattern (omitted by default)",
    )
    inspect.add_argument(
        "--include-agent-version",
        action="store_true",
        help="show agent version numbers that match a strict numeric pattern (omitted by default)",
    )
    inspect.add_argument(
        "--discover-schema",
        action="store_true",
        help="add a structural description of the records: only reviewed vocabulary names, opaque codes for "
        "anything else, never values (off by default)",
    )
    return parser


def main(argv: list[str] | None = None, *, adapters: Sequence[SourceAdapter] = DEFAULT_ADAPTERS) -> int:
    parser = build_parser(adapters)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return EXIT_OK
    try:
        code, output = _inspect(args, adapters)
        if output:
            sys.stdout.write(output)
            sys.stdout.flush()
        return code
    except BrokenPipeError:
        # Output closed early (e.g. piped to `head`): exit quietly.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return EXIT_OK
    except KeyboardInterrupt:
        _error("interrupted")
        return EXIT_INTERRUPTED


def _inspect(args: argparse.Namespace, adapters: Sequence[SourceAdapter]) -> tuple[int, str]:
    names = ", ".join(adapter.name for adapter in adapters)
    try:
        processors: list = [
            InspectionProcessor(
                include_file_hash=args.include_file_hash,
                include_models=args.include_models,
                include_agent_version=args.include_agent_version,
            )
        ]
        if args.discover_schema:
            processors.append(DiscoveryProcessor())
        results = run(
            args.file,
            processors,
            fmt=args.format,
            adapters=adapters,
            limits=Limits(max_line_bytes=args.max_line_bytes),
            compute_hash=args.include_file_hash,
        )
        report, discovery = results["inspection"], results.get("discovery")
        if args.output == "json":
            payload = report.to_dict()
            if discovery is not None:
                payload["discovery"] = discovery
            output = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
        else:
            output = render_text(report) + (("\n" + render_discovery(discovery)) if discovery is not None else "")
    except InputError as error:
        _error(str(error))  # fixed messages, never the path
        return EXIT_INPUT, ""
    except FormatError as error:
        detection = error.detection
        if detection.status is DetectionStatus.CONTRADICTORY:
            _error(
                "contradictory evidence: records match the signatures of several formats "
                f"({', '.join(detection.matched)}); the file may mix formats."
            )
        elif detection.status is DetectionStatus.UNSUPPORTED and detection.unsupported is not None:
            _error(f"the records match a known but unsupported format: {detection.unsupported.display_name}.")
        else:
            _error(
                f"insufficient evidence: no supported format signature in {detection.records_examined} record(s)."
            )
            if detection.likely:
                _error(f"weak evidence suggests: {', '.join(detection.likely)} (not enough to decide).")
        _error(f"if you are sure, specify it with --format ({names}).")
        return EXIT_FORMAT, ""
    except Exception as error:  # noqa: BLE001 - last resort: report the location only, never the message
        frame = traceback.extract_tb(error.__traceback__)[-1]
        _error(
            f"internal error {type(error).__name__} at {os.path.basename(frame.filename)}:{frame.lineno} "
            "(details suppressed so that no trace content is exposed)."
        )
        return EXIT_INTERNAL, ""
    return EXIT_OK, output


def _error(message: str) -> None:
    print(f"agent-traces: error: {message}", file=sys.stderr)
