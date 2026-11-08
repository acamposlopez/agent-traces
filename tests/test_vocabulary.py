"""The reviewed vocabularies are explicit, internally consistent and carry a provenance for every name."""

from __future__ import annotations

import pytest

from agent_traces.adapters import DEFAULT_ADAPTERS, KNOWN_UNSUPPORTED
from agent_traces.adapters.base import Source
from agent_traces.adapters.claude_code_vocabulary import METADATA_SPECS, SYSTEM_SPECS, VOCABULARY

RECOGNIZERS = [*DEFAULT_ADAPTERS, *KNOWN_UNSUPPORTED]


@pytest.mark.parametrize("recognizer", RECOGNIZERS, ids=lambda r: r.name)
def test_vocabulary_is_consistent(recognizer):
    vocab = recognizer.vocabulary
    tables = (vocab.record_types, vocab.discriminators, vocab.fields, vocab.block_types)
    for table in tables:
        assert all(isinstance(name, str) and name for name in table)
        assert set(table.values()) <= set(Source)
    assert vocab.containers <= set(vocab.fields), "every container must be a reviewed field name"
    for path in vocab.discriminator_paths:
        assert set(path) <= set(vocab.fields), "discriminator paths must use reviewed field names"


def test_claude_code_metadata_specs_use_the_vocabulary():
    assert set(METADATA_SPECS) <= set(VOCABULARY.record_types)
    assert set(SYSTEM_SPECS) <= set(VOCABULARY.discriminators)
    for spec in (*METADATA_SPECS.values(), *SYSTEM_SPECS.values()):
        assert spec.expected <= set(VOCABULARY.fields)
        if spec.discriminator is not None:
            assert spec.discriminator in VOCABULARY.discriminator_paths


def test_adapter_allowlists_are_part_of_the_vocabulary():
    for adapter in DEFAULT_ADAPTERS:
        assert set(adapter.known_record_types) <= set(adapter.vocabulary.record_types)
