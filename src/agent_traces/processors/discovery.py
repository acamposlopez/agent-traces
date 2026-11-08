"""Schema discovery (opt-in): describe the structure of records without revealing names or values.

- Record types, field names, discriminator values (subtype-like fields) and content-block types are shown
  only when they belong to the active adapter's reviewed vocabulary. A name is never trusted because of how
  it looks: anything outside the vocabulary becomes an opaque local code (T1, K1, S1, B1...) assigned in
  order of first appearance in this run. Codes are not hashes; they cannot be reversed or compared across
  runs, and the internal digests behind them use a random per-run key that is never revealed.
- Values are never shown: only JSON value types ("string", "object"...) and counts.
- Records are grouped by what the parser made of them (semantic / metadata / unknown / skipped), record type
  and discriminator. Nothing is stored beyond these aggregates, and every structure is bounded.
"""

from __future__ import annotations

import hashlib
import secrets
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agent_traces.adapters.base import Source, Vocabulary
from agent_traces.engine import RunContext
from agent_traces.model import Event, RecordCategory
from agent_traces.processors.classification import RecordClassifier
from agent_traces.reader import RawRecord

MAX_CODES = 64  # opaque codes per namespace; beyond, everything shares the overflow code ("T*", "K*"...)
MAX_GROUPS = 256
MAX_KEYS_PER_OBJECT = 256
_PREFIX = {"type": "T", "field": "K", "value": "S", "block": "B"}
_CATEGORY_ORDER = {
    RecordCategory.UNKNOWN.value: 0,
    RecordCategory.METADATA.value: 1,
    RecordCategory.SEMANTIC.value: 2,
    RecordCategory.SKIPPED.value: 3,
}
NOTE = (
    "Structure only: names outside the reviewed vocabulary are replaced by local codes assigned in order of "
    "appearance; values are never shown."
)


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "other"


@dataclass(slots=True)
class _Group:
    records: int = 0
    fields: dict[str, Counter[str]] = field(default_factory=dict)
    nested: dict[str, dict[str, Counter[str]]] = field(default_factory=dict)
    blocks: Counter[str] = field(default_factory=Counter)
    truncated_fields: int = 0


class DiscoveryProcessor:
    name = "discovery"

    def __init__(self, *, max_codes: int = MAX_CODES, max_groups: int = MAX_GROUPS) -> None:
        self.max_codes = max_codes
        self.max_groups = max_groups
        self._reset(Vocabulary())

    def _reset(self, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary
        self._key = secrets.token_bytes(16)
        self._codes: dict[str, dict[bytes, str]] = {namespace: {} for namespace in _PREFIX}
        self._groups: dict[tuple[str, str, Any], _Group] = {}
        self._used: set[str] = set()  # vocabulary labels that appear in the output
        self.truncated = False
        self.overflow_records = 0
        self._classifier = RecordClassifier(self._add)

    # Processor and RecordObserver protocols ---------------------------------------------------------------

    def start(self, context: RunContext) -> None:
        self._reset(context.adapter.vocabulary if context.adapter else Vocabulary())

    def observe(self, record: RawRecord) -> None:
        self._classifier.observe(record.line, self._describe(record))

    def handle(self, event: Event) -> None:
        self._classifier.event(event)

    def finish(self, context: RunContext) -> dict[str, Any]:
        self._classifier.finish()
        return self._result()

    # Description of one record (labels only; nothing from the record is kept) ------------------------------

    def _describe(self, record: RawRecord) -> tuple[Any, ...]:
        data = record.data
        if data is None:
            reason = record.problem.value if record.problem else "unparseable"
            return ("(unparseable line)", ("problem", reason), (), (), ())
        vocab = self.vocabulary
        type_label = self._value_label(data.get("type"), vocab.record_types, "type")
        discriminator = None
        for path in vocab.discriminator_paths:
            value = _resolve(data, path)
            if value is not None:
                discriminator = (".".join(path), self._value_label(value, vocab.discriminators, "value"))
                break
        type_value = data.get("type")
        unknown_type = not (isinstance(type_value, str) and type_value in vocab.record_types)
        fields, nested = [], []
        for count, (key, value) in enumerate(data.items()):
            if count >= MAX_KEYS_PER_OBJECT:
                fields.append(("(more fields)", "other"))
                break
            label = self._field_label(key)
            fields.append((label, json_type(value)))
            # Describe one level deeper for known containers and for anything unknown (still labels only).
            if isinstance(value, dict) and (key in vocab.containers or key not in vocab.fields or unknown_type):
                nested.append((label, self._object_fields(value)))
        blocks = self._blocks(data)
        return (type_label, discriminator, tuple(fields), tuple(nested), blocks)

    def _object_fields(self, value: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        entries = []
        for count, (key, item) in enumerate(value.items()):
            if count >= MAX_KEYS_PER_OBJECT:
                entries.append(("(more fields)", "other"))
                break
            entries.append((self._field_label(key), json_type(item)))
        return tuple(entries)

    def _blocks(self, data: dict[str, Any]) -> tuple[str, ...]:
        message = data.get("message")
        content = data.get("content")
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            content = message["content"]
        if not isinstance(content, list):
            return ()
        labels = []
        for block in content[:MAX_KEYS_PER_OBJECT]:
            if isinstance(block, dict):
                labels.append(self._value_label(block.get("type"), self.vocabulary.block_types, "block"))
        return tuple(labels)

    def _value_label(self, value: Any, known: Any, namespace: str) -> str:
        if value is None:
            return "(absent)"
        if not isinstance(value, str):
            return f"({json_type(value)})"
        if value in known:
            self._used.add(value)
            return value
        return self._opaque(namespace, value)

    def _field_label(self, key: str) -> str:
        if key in self.vocabulary.fields:
            self._used.add(key)
            return key
        return self._opaque("field", key)

    def _opaque(self, namespace: str, value: str) -> str:
        table = self._codes[namespace]
        digest = hashlib.blake2b(value.encode("utf-8", "surrogatepass"), digest_size=16, key=self._key).digest()
        code = table.get(digest)
        if code is None:
            if len(table) >= self.max_codes:
                self.truncated = True
                return f"{_PREFIX[namespace]}*"
            code = table[digest] = f"{_PREFIX[namespace]}{len(table) + 1}"
        return code

    # Aggregation ------------------------------------------------------------------------------------------

    def _add(self, line: int, structure: tuple[Any, ...], category: RecordCategory) -> None:
        type_label, discriminator, fields, nested, blocks = structure
        key = (category.value, type_label, discriminator)
        group = self._groups.get(key)
        if group is None:
            if len(self._groups) >= self.max_groups:
                self.truncated = True
                self.overflow_records += 1
                return
            group = self._groups[key] = _Group()
        group.records += 1
        for label, kind in fields:
            group.fields.setdefault(label, Counter())[kind] += 1
        for container, entries in nested:
            target = group.nested.setdefault(container, {})
            for label, kind in entries:
                target.setdefault(label, Counter())[kind] += 1
        group.blocks.update(blocks)

    def _result(self) -> dict[str, Any]:
        ordered = sorted(
            self._groups.items(), key=lambda item: (_CATEGORY_ORDER.get(item[0][0], 9), -item[1].records, item[0][1])
        )
        groups = []
        for (category, type_label, discriminator), group in ordered:
            groups.append(
                {
                    "category": category,
                    "type": type_label,
                    "discriminator": {"path": discriminator[0], "value": discriminator[1]} if discriminator else None,
                    "records": group.records,
                    "fields": _field_stats(group.fields),
                    "nested": {container: _field_stats(stats) for container, stats in sorted(group.nested.items())},
                    "blocks": dict(sorted(group.blocks.items())),
                }
            )
        sources: dict[str, list[str]] = {Source.DOCUMENTED.value: [], Source.OBSERVED.value: []}
        for label in sorted(self._used):
            source = self.vocabulary.source_of(label)
            if source is not None:
                sources[source.value].append(label)
        return {
            "note": NOTE,
            "groups": groups,
            "opaque_codes": {
                "record_types": len(self._codes["type"]),
                "fields": len(self._codes["field"]),
                "discriminator_values": len(self._codes["value"]),
                "block_types": len(self._codes["block"]),
            },
            "vocabulary_sources": sources,
            "truncated": self.truncated or self._classifier.overflowed > 0,
            "records_not_grouped": self.overflow_records,
        }


def _resolve(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _field_stats(stats: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
    return {
        label: {"records": sum(kinds.values()), "types": dict(sorted(kinds.items()))}
        for label, kinds in sorted(stats.items())
    }


def render_discovery(result: dict[str, Any]) -> str:
    """ASCII text rendering of the discovery result."""
    lines = ["Schema discovery", f"  {result['note']}"]
    for group in result["groups"]:
        disc = group["discriminator"]
        title = group["type"] + (f" {disc['path']}={disc['value']}" if disc else "")
        lines.append(f"  [{group['category']}] {title}: {group['records']} records")
        lines.append(f"      fields: {_describe_fields(group['fields'], group['records'])}")
        for container, stats in group["nested"].items():
            lines.append(f"      {container}.{{{_describe_fields(stats, None)}}}")
        if group["blocks"]:
            lines.append("      content blocks: " + ", ".join(f"{label} {n}" for label, n in group["blocks"].items()))
    codes = result["opaque_codes"]
    lines.append(
        f"  opaque codes (names withheld): {codes['record_types']} record types, {codes['fields']} fields, "
        f"{codes['discriminator_values']} discriminator values, {codes['block_types']} block types"
    )
    sources = result["vocabulary_sources"]
    lines.append(f"  vocabulary used: {len(sources['documented'])} documented names, {len(sources['observed'])} "
                 "observed names (public analyses)")
    if result["truncated"]:
        lines.append(f"  truncated: limits reached ({result['records_not_grouped']} records not grouped)")
    return "\n".join(lines) + "\n"


def _describe_fields(stats: dict[str, dict[str, Any]], records: int | None) -> str:
    parts = []
    for label, info in stats.items():
        kinds = "|".join(info["types"])
        share = "" if records is None or info["records"] == records else f" x{info['records']}"
        parts.append(f"{label}:{kinds}{share}")
    return ", ".join(parts) if parts else "none"
