# DS41-DSML reference coverage matrix

Status: draft for CTR revision R3 (the report folds this in whole as its
reference section). Prepared by ds4-dfm-rs work session, 2026-10-18.

This document answers one question: for every DSML requirement this project
has written down, which case exercises it, and what did that case actually
measure? Its rows come from the two maintained documents and nowhere else:

- `docs/ds41-dsml-conformance.md` (CTR R2) — requirements R-01..R-18 (clause
  6.3), nonconformities NC-01..NC-08 (clause 8), phases P0..P5 (clause 9);
- `docs/ds41-dsml-port-inventory.md` — gaps G1..G9 (section 4).

A row whose requirement has no case stays OPEN and blocks any "all covered"
claim. The leg's own findings — differences the oracle leg measured that
neither source lists yet — are rows O-01..O-04 in part D; they are findings,
not requirements, and each names whether it is a defect, a delta between the
two references, or a decision the operator owes.

## 1. The oracle

| Field | Value |
| --- | --- |
| File | `tests/ds41_ref/oracle/encoding.py` (vendored in this repo) |
| sha256 | `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2` |
| Provenance | operator-supplied, read 2026-10-18 from `/data/ds4/test/encoding.py`; the vendored copy is byte-identical, and that external tree is no longer consulted |
| What it is | the official DeepSeek-V4.1 text/vision encoding implementation: the implementation that writes the prompt the model was trained on and asserts the completion shape it produces |
| Driver | `tests/ds41_ref/ref_drive.py` — `encode CASE MODE` prints the prompt bytes, `parse COMPLETION MODE` prints the parsed message; a hash other than the pinned one aborts the run |
| Inputs | `tests/ds41_ref/cases/*.json` (encode cases, the oracle's own case format), `tests/ds41_ref/decode/*.txt` (completion bytes, built from the oracle's templates by `tests/ds41_ref/mk_decode_input.py`) |
| Fixtures | `crates/ds4-server/tests/fixtures/ds41-ref/` — `<case>.prompt.txt` (oracle prompt bytes), `<case>.calls.json` (oracle parse), `<case>.txt` (the completion the oracle parsed). Regenerate with `bash tests/ds41_ref/regen_fixtures.sh` |
| Rust tests | `crates/ds4-server/tests/ref_encoding.rs` (encode side), `crates/ds4-server/tests/ref_decode.rs` (decode side) |

The oracle is the carriage's single authority. The CTR still quotes clause
text and line numbers taken from the C implementation that once served as a
second reference; those citations are historical quotations of the texts the
requirements were written from, not a live reference, and where the two
disagreed the disagreement is a delta row with both citations (O-01, O-03,
O-04). A row is decided by the oracle.

Traps measured on this host (python 3.13.5) while building the leg, recorded
so the next reader does not re-learn them:

- a case-level `tools` key is dropped unless the case goes through
  `load_cases`, which moves it into `messages[0]` (encoding.py:847-869);
- `thinking_mode` accepts only `chat` or `thinking` (encoding.py:596);
- the thinking-mode completion must not carry the opening `<think>` token:
  the oracle asserts that no special token appears in content or reasoning
  (encoding.py:982), so it takes the generation, not the prompt tail.

## 2. Part A — the oracle leg's cases

Encode side: our route renders the case, the fixture is the oracle's bytes
(`crates/ds4-server/tests/ref_encoding.rs`).

| Case | Pins | Verdict |
| --- | --- | --- |
| `enc-tools-chat` | the tools preamble and the per-schema line in chat mode (TOOLS_TEMPLATE, encoding.py:481-506) | pass |
| `enc-plain-chat` | the no-tools head: BOS, system, user, assistant content, EOS, and the assistant header before the last turn | pass |
| `enc-argtypes-chat` | the type attribute — `string="true"` for a string child, `string="false"` with the value in json.dumps shape otherwise (encode_arguments_to_dsml, encoding.py:524-556) — plus a merged tool result | pass |
| `enc-think-tools` | thinking mode with tools: the effort header at 75 and the open `<think>` the assistant turn must carry | pass |
| `enc-think-max` | the numeric effort header at 100 (encoding.py:443-472) | pass |
| `enc-think-low` | the same header at the low effort | OPEN O-01 (two halves, see part D) |
| `enc-midsystem-chat` | a mid-conversation system message | pass (closed 2026-10-18, render.rs `pending_merge` + the in-place system branch) |
| `enc-toolresults-order` | tool results ordered by the assistant's call order (encoding.py:233-266) | pass (closed 2026-10-18, render.rs `tool_run_in_call_order`) |

Decode side: the completion bytes are the oracle's own, and our parser must
agree with the oracle's parse (`crates/ds4-server/tests/ref_decode.rs`).

| Case | Pins | Verdict |
| --- | --- | --- |
| `dec-one-call` | one invoke, one string and one integer child, prose before the block | pass |
| `dec-two-calls` | two invokes in one wrapper | pass |
| `dec-complex-args` | every JSON child kind, including a nested array and object | pass (values); the argument-string shape is O-04 |
| `dec-unicode` | a non-ASCII string child round-trips raw (the oracle's json.dumps is `ensure_ascii=False`) | pass |
| `dec-marker-in-value` | a block marker inside a parameter value is data, never structure (inventory G9, see part C) | pass |
| `dec-thinking` | the reasoning/prose split in thinking mode | pass |

## 3. Part B — CTR requirements R-01..R-18

"Case" names the artifact that exercises the requirement; "measured" means it
was run in the session that wrote the row, with the output quoted in the CTR's
evidence index or in the test itself.

| Row | Requirement (clause 6.3) | Case | Verdict at this revision |
| --- | --- | --- | --- |
| R-01 | OpenAI chat serves one DSML call as `tool_calls` | TP-01 / A-01 (`tests/spark/dsml_conformance.sh`) | pass at the served baseline (`iso-baseline-20261018-113636`); the claim is per-artifact, so the gate re-runs it |
| R-02 | typed parameters survive as JSON types | TP-02 / A-02 | pass at the baseline |
| R-03 | two invokes in one block | TP-03 / A-03 | pass at the baseline |
| R-04 | streaming carries the call | TP-04 / A-04 | pass at the baseline |
| R-05 | Anthropic serves `tool_use` | TP-05 / A-05 | pass at the baseline |
| R-06 | Responses serves `function_call` | TP-06 / A-06 | pass at the baseline |
| R-07 | chat turn 2 reuses the sampled block, G > 0 | B-01 | reserved: no case (CTR clause 3 proves the ceiling without the excluded KV work) |
| R-08 | replay survives a differently spelled client echo | B-02 | reserved as R-07 |
| R-09 | Anthropic turn 2 reuses the sampled block | B-03 | reserved as R-07 |
| R-10 | Responses continuation reuses the sampled call | B-04 | reserved as R-07 |
| R-11 | a cut block is not served as assistant content | C-01 (suite) + `tests/dsml.rs::live_tool_root_block_recovers_its_calls` (parser) + `tests/dsml.rs::tracker_enters_tool_mode_on_the_quoted_tool_root` and `tests/generate.rs::v41_live_tool_root_turn_streams_calls_without_markup` (tracker) | host pass: the parser half landed at 23eb6b2, the tracker half at d5ad35a, where the recorded turn streams its calls with no markup in the body (A/B: with the new scan off the wrapper close reaches the client as content); the served verdict is the C-01 suite re-run at the gate |
| R-12 | replay and trace diagnostics are available | `--help` check (Annex B, `flags-baseline-20261018.txt`) | OPEN: 2 of 3 at HEAD (`--disable-exact-dsml-tool-replay`, `--tool-memory-max-ids`, `crates/ds4-server/src/bin/ds4-server-rs.rs:225-231`, advertised at :1327); `--trace` is unit 3.2 |
| R-13 | plain-text KV reuse intact (protection) | TP-13 / D-01 | measured pass at the baseline; re-run every phase |
| R-14 | determinism witness reproducible (protection) | TP-14 / D-02, witness sha16 `adf7157c8a5bbb4b` | measured pass at the baseline; re-run every phase |
| R-15 | exact replay reaches every chat-shaped surface | host: `generate.rs::dsml_replay::sampled_bytes_reach_the_prompt_on_every_chat_surface` (:7170); served: B-01..B-04 with `--trace` | host leg pass (run this session); the served leg is OPEN until `--trace` lands (unit 3.2) and the gate serves it (unit 3.3) |
| R-16 | the replay path has a kill switch | host: `generate.rs::dsml_replay::the_disable_switch_leaves_the_canonical_render` (:7210); served: B-01 with the flag | host leg pass (run this session); the served leg needs the same trace A/B |
| R-17 | a cut turn keeps its truncation signal and a legal finish | C-01, C-02, C-03 (`tests/spark/dsml_conformance.sh:266-312`) | probes exist (added in 903bcf0); verdict OPEN until the gate run (unit 3.3) decides whether the loss is reachable |
| R-18 | the client of record completes a real tool loop | `tests/spark/open_grok_leg.sh` (G1/G2/G3) | OPEN: the leg needs the IUT on 8008 (unit 3.3) |

## 4. Part C — inventory gaps G1..G9

| Row | Gap (inventory section 4) | Case | Verdict at this revision |
| --- | --- | --- | --- |
| G1 | ds41 has no session, so no replay, tool memory, continuation or canonicalization | R-15/R-16 rows carry the replay half; the rest is reserved | partially covered: the host-side replay carriage landed (8891d3e) and `tool_replay_scope` is now `kind == Chat` (`generate.rs:1367`), so a session is no longer the carrier; canonicalization and the in-turn continuation stay reserved (CTR clause 9) |
| G2 | replay was OpenAI-chat-only, off under thinking and live ids | `generate.rs:1367-1380` + the per-surface host test (R-15) | closed for the carriage: the scope is the reference's rule and the attach is per surface; the served verdict rides R-15 |
| G3 | no tool-checkpoint canonicalization | none | out of scope for this line (CTR clause 1.3 excludes the KV store; clause 9 reserves it) |
| G4 | no invalid-DSML tool-error continuation | none yet | reserved in full; the repair half is P3 (unit 3.4) |
| G5 | no `complete_tool_call_inside_thinking` equivalent | none | OPEN, unverified: an Anthropic live-path decision (NC-06), which the oracle leg does not reach |
| G6 | no `build_responses_visible_assistant_suffix` | none yet | OPEN, P4 (unit 3.5) |
| G7 | the two replay flags were absent | R-12's flag check | partially closed: both flags exist at HEAD; `--trace` is unit 3.2 |
| G8 | `try_repair_dsml` deviates: (a) closes unbalanced invokes/params, (b) omits the spaced spelling, (c) picks the style by presence not by earliest occurrence | `tests/parity/retry_c_oracle.c` + `crates/ds4-server/tests/retry.rs` today; the port of `test_dsml_repair_produces_parseable_calls` is P3 | OPEN, P3 (unit 3.4): NC-05 stands as written, and the reference's own refusal comment (`ds4_server.c:6697-6700`) is what the fix must satisfy |
| G9 | no `find_tool_structural_text` counterpart | parse side: `dec-marker-in-value` (pass); streaming side: `stream.rs::text_stream_stray`; repair side: none | partially covered: the rule that a marker inside a value is not structure is measured green on the decode side; the think-scope scan inside the repair path is OPEN with G8 |

## 5. Part D — the leg's own findings

| Row | Finding | Evidence | Class |
| --- | --- | --- | --- |
| O-01 | the oracle renders the requested budget verbatim: a name maps to 50/75/100, an integer in [1,100] renders as itself, and either is written only in thinking mode at index 0 (encoding.py:443-473). Two halves on our side. (a) The low name renders nothing: on the ds41 route the requested effort never reaches the renderer, because `engine_defaults` (set on the V4.1 branch, `bin/ds4-server-rs.rs:611`) collapses low into high through `engine_think_mode` (`parse.rs:2817`), and `head_effort_prefix` knows only high and max (`render.rs:69-74`). (b) A numeric `reasoning_effort` is refused: the value parser reads a string name and nothing else (`format.rs:96-102`) | `enc-think-low` (fixture 1372 bytes vs our 1311) | OPEN, both halves are one designed change: carry the requested budget to the head prefix as a number (with the low name as the 50 case), accept a number in the parse, and leave the engine's own mode collapsed so the reasoning cap and sampling switches do not move |
| O-02 | CLOSED 2026-10-18: a mid-conversation system message writes the system token again at its position, introduces its own turn, and counts as user-like for the assistant header that follows it (encoding.py:621-627, :733-739, `find_last_user_index` :562-582) | `enc-midsystem-chat` | pass; measured before the fix, "Answer in one line." landed before the tools preamble instead of after the user turn. The fix added the in-place system branch and split `pending_merge` from `pending_assistant` in `render.rs`, so a user message following a mid system opens its own `<｜User｜>` exactly as the oracle's merge rule requires (encoding.py:184-231) |
| O-03 | CLOSED 2026-10-18: tool results are ordered by the preceding assistant's call order, not by arrival (encoding.py:233-266) | `enc-toolresults-order` | pass; measured before the fix, the oracle's block carried body a then body b and ours body b then body a, with the other 1767 bytes matching. `tool_run_in_call_order` (`render.rs`) sorts a run of tool-result messages by the ids the preceding assistant declared; a result whose id was never declared keeps the oracle's default key 0 and the sort is stable. This also makes the prompt canonical for a client that sends results in completion order, which the OpenAI wire permits (CTR clause 2.3.8) |
| O-04 | our call arguments minify a composite child (`tool_call_json_args_add`, `tools.rs:150-166`, mirroring `ds4_server.c:5971-5982` over `json_minify_raw_value` :411) while the oracle keeps the parameter body verbatim; the values are equal and both parse | `composite_arguments_are_minified_as_the_reference_does` (pass) | recorded delta, pinned from our side; no action |

## 6. Part E — the client of record's wire contract

The operator's goal for this line is stated as OpenAI compatibility with
`/data/open-grok`, the configured consumer, so the client's own wire types are
the second acceptance oracle beside the CTR.  Every row below was read out of
the client's source at `27b48a6` in this session; "satisfied by construction"
means our writer's branch was read, and the case is what measures it end to
end.

| Row | The client's requirement (source) | Our counterpart | Case | Verdict |
| --- | --- | --- | --- | --- |
| E-01 | `ChatCompletionResponse` requires `id`, `object`, `created`, `model`, `choices` (no serde default) — `xai-grok-sampling-types/src/types.rs:353-359` | final writer emits all five (`stream.rs:2347`), chunks emit id/object/chat.completion.chunk/created/model (`stream.rs:561, 637, 657, 678, 962, 992, 1002`) | tests/stream.rs, tests/generate.rs, client leg | satisfied by construction |
| E-02 | `ChatChoice.index` and `ChatResponseMessage.role` are required; `Role` is `system|user|assistant|tool` (`types.rs:366-370`, `:273-278`, `:384-385`) | writers emit `index:0` and `role:"assistant"` | tests/stream.rs, client leg | satisfied by construction |
| E-03 | `finish_reason` is a CLOSED enum `stop|length|tool_calls|content_filter|function_call`; any other value fails that chunk's deserialization (`types.rs:373-381`) | the ds41 route remaps `error` to `stop` before every writer (`generate.rs:3324`) | R-17's C-02/C-03 probes at the gate | satisfied on the ds41 route; the truncation loss is R-17's open half |
| E-04 | every tool delta needs `index`: `ToolCallDelta.index` has `#[serde(default)]`, so an omitted index collapses distinct calls onto 0 (`types.rs:495-499`; assembly `stream/chat_completions.rs:74-77`, `:199-231`) | every delta carries `"index":N` (`stream.rs:206`) | tests/stream.rs, tests/tool_stream.rs | satisfied by construction |
| E-05 | a call id is required: `ToolCallResponse.id` is a plain `String` (`types.rs:399-401`) | the writer falls back to a generated id when the parse left none (`stream.rs:171-176`, `:200-205`) | tests/stream.rs | satisfied by construction |
| E-06 | `function.name` and `function.arguments` are required strings, arguments a single string (`types.rs:406-409`) | both always emitted, arguments as one JSON object string | tests/stream.rs | satisfied by construction |
| E-07 | streaming is read as SSE `data:` chunks with a tool-call accumulator and a `[DONE]` terminator (`stream/chat_completions.rs:120-260`) | our SSE framing (`stream.rs:561` onward) | the client leg (G1/G2) | OPEN until the client leg runs (unit 3.3) |
| E-08 | reasoning is read from `reasoning_content` only, and the client synthesizes a reasoning sibling only when it carried text (`types.rs:384-396`; `stream/chat_completions.rs:262-266`) | we emit `reasoning_content` on the reasoning channel (`stream.rs:816`, `:859`, `:2357`) | R-15/R-16 host legs; the client leg | partially measured: the host legs pass, the served leg is open |
| E-09 | in thinking mode the client echoes `reasoning_content` on EVERY assistant message, empty included, and replays `function.arguments` as the raw string it holds (invalid JSON replaced by `{}`) (`conversation.rs:3727-3741`, `:3516-3527`) | our parse reads both fields and does not require reasoning; unknown keys are skipped | R-15's host leg (four surfaces parse the client's own echo) | host pass |
| E-10 | tool results are appended in completion order with `parallel_tool_calls = true`, so the server receives them out of call order | `tool_run_in_call_order` re-sorts a run of results by the assistant's declared call order | enc-toolresults-order | closed 2026-10-18: the oracle's order is applied, so a client that sends results in completion order still renders the canonical prompt (O-03) |

## 7. What blocks an "all covered" claim

The OPEN rows, in the order the plan would close them: E-07 and E-08's served
half and R-18, which are the same measurement (the client leg, unit 3.3), then
R-12 and R-15's served leg (`--trace`, unit 3.2), R-17 (the gate's C-01..C-03
probes), G8 and G9's repair-side scan (P3, unit 3.4), G6 (P4), G5 (needs the
Anthropic live path to be probed), O-01 (the effort budget, both halves),
and R-07..R-10 together with G1's remainder and G3/G4, which are reserved to a
unit that may alter KV state.

Covered today, measured in this session or quoted from the CTR's evidence
index: R-01..R-06, R-11 at host level, R-13, R-14, R-15's host leg, R-16's
host leg, E-01..E-06 and E-09 by construction and host test, and every part A
case marked pass.

For the operator's stated goal -- a plain OpenAI server any compatible caller
can drive, the way a hosted DeepSeek endpoint is driven -- the shortest path is
E-07 and E-08: everything a client's deserializer requires is emitted today,
but no leg has yet driven a real client against a served binary (unit 3.3).
