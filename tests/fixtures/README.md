# Synthetic fixtures

Every file in this directory was written by hand for the tests. **They are not real samples and have not
been verified against real traces**: they reproduce the structure described in the public sources listed in
the project README (official Cursor documentation; official documentation and public analyses of Claude
Code) with invented values.

- Every fictitious sensitive value (prompts, secrets, paths, e-mail addresses, branches, unknown names)
  contains the marker `CANARY`. `tests/test_privacy.py` checks that it never appears in reports, errors or
  `repr` output.
- `claude-code/`:
  - `metadata-records`: metadata records with the structures described by public sources (progress,
    attachment, last-prompt, titles, modes, file-history-*, queue-operation, cost-state and system
    subtypes), user messages with harness markers, and `session_id` next to `sessionId` as recent clients
    write it;
  - `simple`: a simple session;
  - `tool-calls`: tool calls, equivalent to `cursor-cli/tool-calls`;
  - `errors-recovery`: errors and recovery;
  - `secrets`: fictitious secrets, MCP or unknown tools, unsafe model and version values;
  - `subagent-main` and `subagent-agent`: main session and subagent transcript.
- `cursor-cli/`:
  - `simple`: a simple session;
  - `tool-calls`: tool calls;
  - `failed-run`: a result with an undocumented shape and a stream cut off without `result`;
  - `sensitive-data`: fictitious sensitive data, `--stream-partial-output` deltas and unknown types.
- `edge-cases/`:
  - empty or blank files;
  - invalid JSON and a truncated last line (`invalid-json.jsonl` deliberately has no final newline);
  - unknown fields, blocks, subtypes and types;
  - incomplete records;
  - an ambiguous mix of formats;
  - an unknown format;
  - an approximation of the Claude Agent SDK `stream-json` output (only to check that it is not mistaken for Cursor);
  - `discovery-hostile`: types, keys, subtypes and blocks with malicious or innocent-looking names outside
    the vocabulary, to check that `--discover-schema` reveals none of them.

Never copy real traces here.
