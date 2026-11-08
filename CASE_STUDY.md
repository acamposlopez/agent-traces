# What can a coding-agent trace actually establish?

Three small, fully synthetic traces in `examples/`, each run through `agent-traces inspect`. The reports below
are the ones the **installed package** produced (version 0.1.0, built with `python -m build` and installed from
the wheel into a clean virtual environment, run from a directory outside the source tree). They are reproduced
verbatim except for the trailing *Limitations* block, which every report prints and which appears once per
format in the appendix. `tests/test_case_study.py` pins the documented counts (from the JSON report, not the text
layout of the reports pasted here) to the example files, so if the examples or those counts change, the suite
says so. The `export policy` line in every report is a per-event default label only; this version exports
nothing and performs no scrubbing.

Two words are used carefully throughout. A tool result is **confirmed** (`success_confirmed`,
`error_confirmed`) when the record itself carries an explicit success or error indicator, and a
**protocol default** (`success_protocol_default`) when the record has no indicator and the documented default
applies; neither is an independent verification of what happened. And **complete** is only ever used in a
narrow, stated sense (every record parses, a terminal marker is present, every call has a recorded result);
whether a recording captures everything that actually happened cannot be shown from the file.

The report shows aggregated counts and relation totals. The canonical model keeps record order and native
identifiers in memory while it runs, but the report never prints a sequence or a timeline.

Every case separates three things:

- **Detected automatically** — what the inspector reports from the recorded structure.
- **Explained by hand** — what the author of the example knows because the example was written on purpose; the
  inspector does not analyse the meaning of any message (user-role text is only matched against fixed harness
  markers to classify its origin, and assistant text is parsed only to count its blocks, never analysed), so it
  would not know it.
- **Not established** — what neither the recording nor the tool can tell you.

To reproduce:

```bash
pip install ".[dev]"                            # from a checkout; see README.md
agent-traces inspect examples/case-a-complete-task.jsonl
agent-traces inspect examples/case-b-missing-result.jsonl
agent-traces inspect examples/case-c-unverified-claim.jsonl
pytest tests/test_case_study.py                  # the documented counts, checked
```

## Case A — a complete record of a task

**Purpose.** Show what the inspector can establish from a well-formed recording of a short task: one prompt,
three tool calls, three results with three different kinds of status, one chain of parent links, as counts and
relation totals. "Complete" here means that every call has a recorded result and the parent chain is unbroken,
not that the file records everything that happened. And show where the evidence stops.

**Fixture** (`examples/case-a-complete-task.jsonl`, Claude Code transcript format, 9 records, written by hand):

| Line | Record | Content |
|---|---|---|
| 1 | `user` | prompt: add a `--dry-run` flag to the sync command and make the test suite pass |
| 2 | `assistant` | `tool_use` **Bash** `pytest -q` |
| 3 | `user` | `tool_result` for line 2, **`is_error: true`**: 1 failed, 3 passed (unrecognized argument) |
| 4 | `assistant` | text: the parser does not define the flag yet, it will be added |
| 5 | `assistant` | `tool_use` **Edit** on `sync.py` (same `message.id` as line 4) |
| 6 | `user` | `tool_result` for line 5, **`is_error: false`**: file updated |
| 7 | `assistant` | `tool_use` **Bash** `pytest -q` |
| 8 | `user` | `tool_result` for line 7, **no `is_error` field**: "4 passed in 0.21s" |
| 9 | `assistant` | text: done, the flag is defined and all 4 tests pass |

Every record from line 2 on carries `parentUuid` = the `uuid` of the previous line.

**Command.** `agent-traces inspect examples/case-a-complete-task.jsonl`

**Output obtained from the installed package:**

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

Records
  interpreted events  9
  unrecognized        none
  metadata types      none
  metadata classes    none
  session markers     none
  normalization       none
  export policy       after_scrubbing 9 (events; nothing is exported yet)
  timestamps          9 present, 0 absent, 0 unparseable
  out of order        0 conversation, 0 metadata

Integrity: ok
```

**Interpretation.**

*Detected automatically.* All 9 records parse and are interpreted. There is 1 user prompt and 2 assistant
messages: the inspector counts assistant messages from text blocks, one per distinct `message.id`, which here
means line 4 and line 9. Lines 2, 5 and 7 contain only tool calls and are counted as calls, not messages (line 5
shares the `message.id` of line 4, so even a second text block there would not add a message). The 3 calls are
each paired with exactly one result and
the parent links form one unbroken chain of 8 resolved references. The 3 results have **three different
statuses**, and the report keeps them apart on purpose: an error stated by the record (`error_confirmed`), a
success stated by the record (`success_confirmed`), and a success that is only the protocol's default because the
record has no `is_error` field at all (`success_protocol_default`). Integrity is `ok`: no call lacks a result,
no parent reference is unresolved, and nothing is duplicated, malformed or out of order.

*Explained by hand.* The story — a failing test, a one-line fix, a passing re-run — is visible to a human reading
the file. The inspector never analyses that text: it reports that there was one error result, one file-writing
call and two shell calls, not what the error was about or that the second run passed; and the report does not
show in which order they happened.

*Not established.* That the software is correct: "4 passed" is text inside a tool result, counted and never
evaluated. That the fix was the right one, or the only change made. That the third call really succeeded: its
status is a default, and the report says so. That nothing else happened: a file can parse perfectly and still
omit actions. A parseable, internally consistent recording with every call answered is evidence that the session
was recorded coherently, **not independent evidence that the task was done well**.

## Case B — an incomplete record

**Purpose.** Show the diagnosis the inspector gives when a recorded action has no recorded outcome, and the
limitation that introduces, without inventing a cause.

**Fixture** (`examples/case-b-missing-result.jsonl`, Cursor Agent CLI `stream-json` format, 6 records):

| Line | Record | Content |
|---|---|---|
| 1 | `system` / `init` | session start |
| 2 | `user` | prompt: run the database migration for staging |
| 3 | `assistant` | text: running the migration now |
| 4 | `tool_call` / `started` | a `function` tool call with a shell command |
| 5 | `assistant` | text: the migration finished without errors |
| 6 | `result` / `success` | terminal event of the run |

There is no `tool_call` / `completed` event for the call started on line 4.

**Command.** `agent-traces inspect examples/case-b-missing-result.jsonl`

**Output obtained from the installed package:**

```
agent-traces inspect: structural report (trace content is never shown)

Source
  format   cursor-cli (auto-detected, 6 records examined)
  adapter  Cursor Agent CLI (stream-json), adapter 0.1.0, canonical schema 0.1.0
  input    1269 bytes, 6 records, 0 blank lines

Session
  session id      present (1 distinct, values not shown)
  agent version   absent
  model           present (1 distinct, not shown; use --include-models)
  terminal event  completed

Coverage (per record)
  semantic    6 (100.0%) interpreted conversational records
  metadata    0 (0.0%) verified structure, 0 (0.0%) name only (content not interpreted)
  unknown     0 (0.0%) not interpretable
  skipped     0 (0.0%) documented duplicates
  recognized  100.0% structurally, 100.0% semantically

Counts
  records        6
  events         6
  messages       3 (assistant 2, user 1)
  user messages  prompt 1 (by origin)
  content kinds  text 3
  reasoning      0 blocks (content never stored)
  tool calls     1
  tool results   0 (none)
  errors         0 (tool 0, run failed 0)

Tools
  categories       unknown 1
  known tools      none
  names not shown  1 distinct (not in the allowlist)

Relations
  call <-> result  0 paired, 1 calls without result, 0 results without call
  parent links     0 resolved, 0 unresolved
  subagents        0 subagent calls, 0 subagent refs, 0 agent ids, 0 sidechain events

Records
  interpreted events  6
  unrecognized        none
  metadata types      none
  metadata classes    none
  session markers     end 1, start 1
  normalization       none
  export policy       after_scrubbing 6 (events; nothing is exported yet)
  timestamps          0 present, 6 absent, 0 unparseable
  out of order        0 conversation, 0 metadata

Integrity: warnings
  [warning] calls_without_result: 1
```

**Interpretation.**

*Detected automatically.* Every record parses, and the documented terminal marker is present: the file starts
with the `init` event (counted as the start marker) and ends with a `result` whose `subtype` is `success` and
`is_error` is false, which is what the report calls `terminal event completed`. That is a marker written by the
client, not completeness of what happened. One tool call has no result:
`calls_without_result: 1`, reported as a warning that moves the integrity status from `ok` to `warnings`. The
tool is a `function` call, which the Cursor reference does not document in detail, so its name is withheld
(`names not shown 1`) and its category is `unknown`.

*Explained by hand.* The author left the `completed` event out deliberately, and wrote an assistant message that
asserts the migration finished. The inspector counts that message (`assistant 2`); it does not relate it to the
missing result.

*Not established.* Why the result is missing. A missing result in this file does **not** show that the command
never ran: the tool may have run and its completion may not have been recorded, an event may have been dropped
from the stream, or the client may not emit completions for this tool type. Nor does `terminal event completed`
show that the migration succeeded or that everything that happened was recorded; it only says that the client
wrote a successful terminal marker. The recording supports exactly one conclusion: for this call, the outcome is
not on record.

## Case C — a claim of success without a check

**Purpose.** Show that a structurally flawless session can contain a claim that nothing in the recording
supports, and be clear about what the tool does and does not analyse.

**Fixture** (`examples/case-c-unverified-claim.jsonl`, Claude Code transcript format, 2 records):

| Line | Record | Content |
|---|---|---|
| 1 | `user` | prompt: deploy the fix to staging and confirm that the health endpoint responds |
| 2 | `assistant` | text: the fix has been deployed and the health endpoint confirmed to respond with 200 OK |

No tool call, no tool result, nothing between the request and the answer.

**Command.** `agent-traces inspect examples/case-c-unverified-claim.jsonl`

**Output obtained from the installed package:**

```
agent-traces inspect: structural report (trace content is never shown)

Source
  format   claude-code (auto-detected, 2 records examined)
  adapter  Claude Code session transcript (JSONL), adapter 0.1.0, canonical schema 0.1.0
  input    1230 bytes, 2 records, 0 blank lines

Session
  session id      present (1 distinct, values not shown)
  agent version   present (1 distinct, not shown; use --include-agent-version)
  model           present (1 distinct, not shown; use --include-models)
  terminal event  not applicable

Coverage (per record)
  semantic    2 (100.0%) interpreted conversational records
  metadata    0 (0.0%) verified structure, 0 (0.0%) name only (content not interpreted)
  unknown     0 (0.0%) not interpretable
  skipped     0 (0.0%) documented duplicates
  recognized  100.0% structurally, 100.0% semantically

Counts
  records        2
  events         2
  messages       2 (assistant 1, user 1)
  user messages  prompt 1 (by origin)
  content kinds  text 2
  reasoning      0 blocks (content never stored)
  tool calls     0
  tool results   0 (none)
  errors         0 (tool 0, run failed 0)

Tools
  categories       none
  known tools      none
  names not shown  0 distinct (not in the allowlist)

Relations
  call <-> result  0 paired, 0 calls without result, 0 results without call
  parent links     1 resolved, 0 unresolved
  subagents        0 subagent calls, 0 subagent refs, 0 agent ids, 0 sidechain events

Records
  interpreted events  2
  unrecognized        none
  metadata types      none
  metadata classes    none
  session markers     none
  normalization       none
  export policy       after_scrubbing 2 (events; nothing is exported yet)
  timestamps          2 present, 0 absent, 0 unparseable
  out of order        0 conversation, 0 metadata

Integrity: ok
```

**Interpretation.**

*Detected automatically.* Two records, both interpreted, one parent link resolved, no tool activity of any kind,
integrity `ok`. That status means the file is well-formed and internally consistent. It says nothing about
whether the task was performed.

*Explained by hand.* The assistant's message claims a deployment and a verification. The recording contains no
action that could have produced either: no shell command, no HTTP check, no file change. The inspector does not
analyse the meaning of messages (assistant text is counted, never analysed), so it does not flag this; it counts
one assistant message and stops there. There
is no detector for unsupported claims in this tool, and this case study does not pretend otherwise.

*Not established.* Whether anything was deployed, whether the endpoint responds, whether the claim is false. An
action could have happened outside the recording, or not at all. Settling that requires evidence independent of
the trace (deployment logs, a health check of your own), never the trace alone.

## Summary

| Case | Structural integrity | What the recording establishes | What it does not |
|---|---|---|---|
| A | `ok` | 1 prompt, 3 calls each with a recorded result, one chain of 8 parent links, one explicit error, one explicit success, one default success | that the fix is correct or complete; that the default success really succeeded; that nothing else happened |
| B | `warnings` (`calls_without_result: 1`) | every record parses and a successful terminal marker is present; one started call has no recorded outcome | why the outcome is missing; whether the command ran or succeeded; whether the recording is complete |
| C | `ok` | two well-formed messages and nothing else | anything about the claimed deployment or verification |

The common thread: the inspector tells you what the recording contains and what it is missing, with counts and
fixed codes. It does not analyse the meaning of the text, does not judge outcomes, and does not turn a tidy file
into proof.

## Appendix — the *Limitations* block printed with every report

For `claude-code` reports:

```
Limitations
  - Content is never shown: prompts, responses, tool arguments and outputs, paths, identifiers and unknown field or tool names are reduced to counts.
  - Model identifiers, agent versions and the file hash are omitted unless explicitly requested.
  - Metadata records are identified, not interpreted: 'verified' means their documented or observed structural fields are present, not that their content is understood.
  - claude-code: the transcript format is internal to Claude Code and changes between versions; this adapter follows public documentation and community analyses and was checked, by structure only, against one real session.
  - claude-code: metadata records (snapshots, titles, queue operations, attachments, progress, modes...) are identified but their content is not interpreted; attachment, last-prompt and similar records may hold prompts, injected context or paths and are not exportable by default.
  - claude-code: recent versions store subagent transcripts in separate files; each file is inspected on its own and links to subagents are reported as references only.
  - claude-code: tool results without is_error are counted as successful by protocol default (Anthropic Messages API), never as confirmed successes.
  - claude-code: assistant records flagged isApiErrorMessage are counted as API errors (interpretation based on the field name).
  - claude-code: the origin of user-role messages (prompt, command, command output, interruption, injected) comes from source flags and harness markers; 'prompt' means no marker was found, not that a person typed it.
```

For `cursor-cli` reports (case B also ends with `1 distinct tool name(s) are not in the adapter allowlist and are
not shown.`):

```
Limitations
  - Content is never shown: prompts, responses, tool arguments and outputs, paths, identifiers and unknown field or tool names are reduced to counts.
  - Model identifiers, agent versions and the file hash are omitted unless explicitly requested.
  - Metadata records are identified, not interpreted: 'verified' means their documented or observed structural fields are present, not that their content is understood.
  - cursor-cli: only the documented stream-json events are interpreted; the Cursor editor's local session storage is a different format and is not supported.
  - cursor-cli: only readToolCall and writeToolCall are documented tool types; other tool types and 'function' tools are counted in category 'unknown' and their names are not shown.
  - cursor-cli: only successful tool results (result.success) are documented; any other result shape is reported with outcome 'unknown', not as an error.
  - cursor-cli: print mode suppresses thinking events, so reasoning never appears in this format.
  - cursor-cli: timestamps only exist on --stream-partial-output deltas.
```
