# Agent Traces — Quality Inspection for Coding-Agent Data

> This experimental tool inspects recorded structure. It does not scrub sensitive content or independently verify task outcomes.

`agent-traces inspect` reads one coding-agent trace file (a Claude Code session transcript, or the
`stream-json` output of the Cursor Agent CLI) and reports its **structure**: how many records parse, how
messages, tool calls and results relate to each other, whether its last record is complete and, where the
format defines one, a terminal marker is present, and how much of it the tool understands. It prints counts
and fixed codes; the default report shows none of the recorded text, and explicit options reveal only specific
metadata (section 3).

**Why it matters for reviewing agent data.** Traces are examined as data for evaluation and debugging, and a
trace often says more than it establishes. A tool result with no explicit status is a protocol default, not a
confirmed success. A call whose result never appears is a gap in the record, not proof that nothing ran. An
assistant message that announces success is a message. The inspector keeps those distinctions visible.

**Run an example.** Declared requirement: Python 3.12 or newer. Tested only on Linux with Python 3.12. No
runtime dependencies. Not on PyPI.

```bash
python3 -m venv .venv && . .venv/bin/activate   # from a checkout of this repository
pip install .                                    # may download the build backend; see section 6
agent-traces inspect examples/case-a-complete-task.jsonl
```

<details>
<summary>Report produced by the installed package for this example (abridged: the <em>Records</em> and <em>Limitations</em> sections are omitted here; the complete report is in <a href="CASE_STUDY.md">CASE_STUDY.md</a>)</summary>

```
agent-traces inspect: structural report (trace content is never shown)

Source
  format   claude-code (auto-detected, 9 records examined)
  adapter  Claude Code session transcript (JSONL), adapter 0.1.0, canonical schema 0.1.0
  input    6747 bytes, 9 records, 0 blank lines

Session
  session id      present (1 distinct, values not shown)
  agent version   present (1 distinct, not shown; use --include-agent-version)
  model           present (1 distinct, not shown; use --include-models)
  terminal event  not applicable

Coverage (per record)
  semantic    9 (100.0%) interpreted conversational records
  metadata    0 (0.0%) verified structure, 0 (0.0%) name only (content not interpreted)
  unknown     0 (0.0%) not interpretable
  skipped     0 (0.0%) documented duplicates
  recognized  100.0% structurally, 100.0% semantically

Counts
  records        9
  events         9
  messages       3 (assistant 2, user 1)
  user messages  prompt 1 (by origin)
  content kinds  text 3
  reasoning      0 blocks (content never stored)
  tool calls     3
  tool results   3 (success_confirmed 1, success_protocol_default 1, error_confirmed 1)
  errors         1 (tool 1, run failed 0)

Tools
  categories       file_write 1, shell 2
  known tools      Bash 2, Edit 1
  names not shown  0 distinct (not in the allowlist)

Relations
  call <-> result  3 paired, 0 calls without result, 0 results without call
  parent links     8 resolved, 0 unresolved
  subagents        0 subagent calls, 0 subagent refs, 0 agent ids, 0 sidechain events

Integrity: ok
```

</details>

In the report, `success_confirmed` and `error_confirmed` mean that the record carries an explicit success or
error indicator; `success_protocol_default` means it carries none and the documented default applies. None of
them is an independent verification of what happened (section 2).

**What it does not show.**

- Whether the task succeeded: the tool does not analyse what messages say (user text is only matched against
  fixed harness markers to classify its origin; assistant text is counted, not read), and no tool-result
  status is an independent verification of what happened (section 2).
- What was said or done: prompts, responses, tool arguments, outputs, paths and identifiers are reduced to
  counts.
- That a trace is safe to share: nothing is scrubbed, and the opt-in flags reveal metadata (section 3).

Contents: [1. What the report contains](#1-what-the-report-contains) · [2. Tool-result statuses](#2-tool-result-statuses) ·
[3. Privacy and processing behaviour](#3-privacy-and-processing-behaviour) · [4. Case study](#4-case-study) ·
[5. Formats and variants checked](#5-formats-and-variants-checked) · [6. Installation and tests](#6-installation-and-tests) ·
[7. Limitations](#7-limitations) · [Maintainer](#maintainer) · [License](#license)

## 1. What the report contains

One command, `inspect`, reads the file incrementally and read-only (a bounded detection pass over the first
records, then one full pass) and prints a text or JSON report with:

- the detected format and the number of records examined to decide it;
- coverage per record: interpreted conversation, identified metadata (verified structure or name only),
  unknown, skipped duplicates;
- counts of messages (user messages by origin: prompt with no harness marker, harness command, command
  output, interruption, injected context), tool calls and tool results, the latter split by status
  (section 2);
- relation totals: calls paired with results, calls without a result, results without a call, parent links
  resolved or unresolved, subagent references;
- integrity: `ok`, `warnings`, `degraded`, `unusable` or `empty`, with fixed issue codes such as
  `calls_without_result`, `malformed_json` or `incomplete_final_record`;
- the adapter's own limitations, printed with every report.

What the model keeps and what the report shows: while it runs, the canonical model keeps record order, native
identifiers and relations in memory so that pairing and parent links can be checked; the report then shows
aggregated counts and relation totals only. It does not print a sequence, a timeline or any per-record listing,
and it does not reconstruct what happened. Assistant messages are counted per `message.id` (several records
with the same id are one logical message); a record whose content is only `tool_use` or `tool_result` blocks
yields tool events, not a message. `--discover-schema` adds field names and JSON types per record type, using
only a reviewed vocabulary with opaque local codes for anything else. The `export policy` row is a per-event
default label; nothing is exported, and the tool includes no export or scrubbing feature.

Exit codes: `0` report produced (also help and version), `1` input error, `2` usage error, `3` format not
determinable, `4` internal error, `130` interrupted. See `agent-traces inspect --help` for the options.

## 2. Tool-result statuses

Every tool result is reported in exactly one of four groups:

| Status | Meaning in the record | What it is not |
|---|---|---|
| `success_confirmed` | the record carries an explicit success indicator (`is_error: false` in Claude Code, `result.success` in Cursor) | an independent verification that the action worked |
| `error_confirmed` | the record carries an explicit error indicator (`is_error: true`) | proof of what failed, or why |
| `success_protocol_default` | the record carries no indicator; the documented protocol default (success) applies | a confirmed success |
| `unknown` | the record uses an undocumented shape or an uninterpretable indicator | an error |

None of these statuses is an independent verification of what happened. They describe the indicator present
in the record, and nothing else.

## 3. Privacy and processing behaviour

Intended behaviour, as checked by the test suite (every report, error message and `repr` is compared against
the `CANARY` markers of the fixtures and against a closed vocabulary, and a Python audit-hook run records file
writes, deletions, socket and process events while inspecting every fixture):

- The default report contains numbers, fixed codes and names from the adapter's reviewed allowlist (built-in
  tool names, record and metadata types). Prompts, responses, tool arguments, outputs, paths, identifiers and
  unknown field or tool names are reduced to counts. The content of thinking blocks is discarded while
  parsing.
- Opt-in flags reveal metadata that the default report omits: `--include-file-hash` prints the SHA-256 of the
  exact file; `--include-models` and `--include-agent-version` print model identifiers and client versions
  that match a conservative pattern, which narrows down provider, model and time. Even the default allowlisted
  names describe how a session was shaped.
- Processing uses no network and writes no files: the tool opens the input read-only and, if the output pipe
  is closed early, `/dev/null`. Installing from source is a separate matter: `pip install .` may download the
  build backend (section 6).
- The whole record is held in memory while it is parsed. Nothing is scrubbed, and nothing here establishes
  that a trace can be shared.

## 4. Case study

[`CASE_STUDY.md`](CASE_STUDY.md), *What can a coding-agent trace actually establish?*, walks through three
small synthetic traces in `examples/`, each shown with the report the installed package produced, and
separates what was detected automatically, what the write-up explains by hand and what remains unverified:

- **A.** every call has a recorded result, with three different statuses; why that is still not evidence that
  the task was done well;
- **B.** a call whose result never appears, next to a successful terminal marker; the diagnosis the inspector
  gives, without inventing a cause;
- **C.** a structurally valid session whose only content is a claim of success; what the tool does not
  analyse.

`tests/test_case_study.py` pins the documented counts (from the JSON report) to the example files.

## 5. Formats and variants checked

Everything was built and tested with the synthetic fixtures in `tests/fixtures/` (hand-written, fictitious
values marked `CANARY`). The Claude Code adapter was additionally checked, by structure only, against one real
session of the maintainer's own; it is not part of this repository and nothing from it was copied here. Cursor
support follows the published reference and has **not** been validated on real Cursor output.

<details>
<summary>Formats and the variants the fixtures exercise</summary>

| Format | Status | Exercised by the fixtures |
|---|---|---|
| `claude-code` (session transcript JSONL) | Supported | conversation records (`user`, `assistant`, `system`); `tool_use` / `tool_result` with `is_error` true, false and absent; thinking blocks (content discarded); a reviewed vocabulary of 32 metadata record types and 12 `system` subtypes identified by name (14 types and 6 subtypes appear in the `claude-code/` fixtures); subagent files; harness markers in user messages; `session_id` next to `sessionId` |
| `cursor-cli` (`--output-format stream-json`) | Supported | `system/init`, `user`, `assistant`, `tool_call` `started`/`completed` for `readToolCall`, `writeToolCall` and `function`; `result`; `--stream-partial-output` deltas and duplicate flushes; a failed run that ends without `result` |
| Claude Agent SDK `stream-json` | Recognised and refused | so that it is not mistaken for Cursor output |

</details>

Sources: Cursor's `--output-format stream-json` reference (https://cursor.com/docs/cli/reference/output-format);
for Claude Code, Anthropic's sessions documentation (https://code.claude.com/docs/en/sessions), the Messages API
and tools reference, and the public transcript parsers listed with their provenance in
`src/agent_traces/adapters/claude_code_vocabulary.py`.

## 6. Installation and tests

Declared: `requires-python >= 3.12`, no runtime dependencies, build backend `setuptools >= 77`. Tested: Linux
with Python 3.12 only; no other platform or version has been tried.

`pip install .` builds the package in an isolated environment and downloads `setuptools` from PyPI unless pip
is pointed at a local copy (`pip install --no-index --find-links <dir> .` with a `setuptools` wheel in `<dir>`
works without network access). `pip install -e ".[dev]"` adds pytest; `pytest` runs the suite, which covers
both adapters, detection, the reader's limits, relation fidelity against the source JSON, robustness against
mutated and hostile records, bounded memory, privacy (section 3), metamorphic properties (renaming identifiers
or changing every fictitious secret leaves the report unchanged) and the case-study counts.

Verified for this version, offline, on Linux with Python 3.12: `python -m build` with build isolation and
local wheels produces the sdist and, from it, the wheel; `pip install --no-index --find-links <dir> .` installs
from the source tree; the wheel installs into a clean virtual environment; the installed CLI runs from outside
the source tree; the test suite passes against the installed package. The sdist contains the tests, their
synthetic fixtures, the examples, the case study and the license.

An isolated development container is also provided (`Dockerfile`, `compose.yaml`; Python 3.12, no network
after the image is built, read-only root filesystem, non-root user): `docker compose up -d --build`, then
`docker compose exec agent-traces-dev pytest`. Building the image downloads the dependencies; the image has
not been rebuilt from this exact tree as part of the verification above.

## 7. Limitations

**Compatibility.** The Claude Code transcript format is internal and changes between versions; records outside
the reviewed vocabulary are counted as unknown and never shown. Cursor support covers the documented events only
(`readToolCall`, `writeToolCall` and `result.success`; any other result shape is `unknown`, not an error).
Subagent transcripts are inspected file by file. The origin of a user-role message comes from flags and harness
markers: `prompt` means no marker was found, not that a person typed it. Forcing `--format` disables detection
and a wrong format yields meaningless counts. Only Linux with Python 3.12 has been tested.

**Evidence.** The report establishes structure, not what happened. A tool result without an indicator is a
protocol default, not a confirmed success, and is reported separately. A call without a recorded result is
reported as such; the file does not say why. A terminal marker means that the client wrote one, not that
everything that happened was recorded. Assistant text is not analysed semantically: a message that claims
success is counted as a message. Detection refuses ambiguous or mixed files instead of guessing, within the
sample of records it examines (the report prints how many); records of another format that appear later in
the file are counted as unknown, not refused. Relation checks use fixed-capacity digests; beyond the capacity
the report says so and counts become approximate.

## Maintainer

Alberto Campos

- GitHub: https://github.com/acamposlopez
- LinkedIn: https://www.linkedin.com/in/albertocamposlopez/

## License

Licensed under the Apache License, Version 2.0; see [`LICENSE`](LICENSE). Claude Code is a product of
Anthropic and Cursor is a product of Anysphere; this project is independent and is not affiliated with or
endorsed by either. Experimental, version 0.1.0; the report format may change.
