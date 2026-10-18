# DS41-DSML function map

One row per function and per named constant of the DeepSeek-V4.1 encoding
oracle, then one row per DSML function of ours that the oracle does not
define.  This is the review list: what the oracle decides, where this repo
implements it, which case proves it, and what is still open.

The oracle is the official V4.1 text/vision encoding implementation,
operator-supplied 2026-10-18.  It is the authority for the carriage this
project serves; the DSML line serves no other markup for the V4.1 family.
It is vendored byte-identical at `tests/ds41_ref/oracle/encoding.py`
(sha256 `13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2`,
provenance in `tests/ds41_ref/oracle/README.md`) and pinned by that hash in
`tests/ds41_ref/ref_drive.py:39`, so a later edit of it fails the run loudly.
Line numbers below are into that vendored file.

## How to read a row

- Oracle: the function or constant, with its line in the oracle file.
- Decides: the one thing it fixes about the bytes.
- Ours: the counterpart in this repo, `file:line`.  "not located in this
  pass" means this session did not find a counterpart; it is a gap, not a
  claim that none exists.
- Case: the reference case that exercises it, under `tests/ds41_ref/`, with
  its fixture under `crates/ds4-server/tests/fixtures/ds41-ref/`.
- Verdict: `pass` (measured byte-identical to the oracle's fixture today),
  `OPEN` (a case exists or is owed and the bytes differ), `no case` (no case
  yet; nothing measured), `harness` (test-side only, not a served rule),
  `out of surface` (defined by the oracle but outside the OpenAI chat wire).

Verdicts are measured, not assumed: `pass` for the encode side means our
prompt bytes equal the oracle's prompt bytes for that case
(`crates/ds4-server/tests/ref_encoding.rs`), and for the decode side that our
parse yields the oracle's call names and argument values
(`crates/ds4-server/tests/ref_decode.rs`).

## 1. Encode side — request to prompt

| Oracle | Decides | Ours | Case | Verdict |
| --- | --- | --- | --- | --- |
| `to_json` (76) | the JSON literal shape for non-string values: `ensure_ascii=False`, python separators | `crates/ds4-server/src/render.rs:295` `pyjson_write`, `:288` `pyjson_raw_value`, `crates/ds4-server/src/json.rs:379` `json_minify_raw_value` | `enc-argtypes-chat`, `dec-unicode` | pass |
| `tools_from_openai_format` (84) | the schema string handed to the model, in the client's declared order | `crates/ds4-server/src/parse.rs:2145` `parse_tools_value`, `:1811` `openai_function_schema_from_tool`, `:1961` `tool_schema_orders_add_json_wire` | `enc-tools-chat` | pass |
| `_split_tool_name` (101) | the `namespace::name` split, and the two asserts around it | `crates/ds4-server/src/parse.rs:2022`, `:2063` (Responses path); no chat-path rule located | none | OPEN, no case: an OpenAI `namespace` field on a chat tool, and a name containing `::`, are both unmeasured |
| `_tool_name_for_encoding` (116) | the name printed in the invoke tag, namespace-qualified | `crates/ds4-server/src/render.rs:535` `append_dsml_tool_calls_text` | `enc-argtypes-chat` (bare name) | pass for the bare name; the qualified form has no case |
| `tool_calls_from_openai_format` (124) | a replayed assistant's `tool_calls` array into the internal call shape | `crates/ds4-server/src/parse.rs:924` `parse_tool_calls_value`, `:868` `add_tool_call_id` | `enc-argtypes-chat` | pass |
| `decode_dsml_to_arguments` (156) | the arguments JSON string: `string="true"` wraps the body as a JSON string, `string="false"` takes the body as the value | `crates/ds4-server/src/tools.rs:375` `parse_dsml_generated`, `:1270`, `crates/ds4-server/src/json.rs:379` | `dec-one-call`, `dec-complex-args` | pass on names and values; our bytes are minified where the oracle keeps the body verbatim (matrix O-04) |
| `merge_tool_messages` (184) | a `tool` role becomes a `tool_result` block inside a user message, merging into an adjacent merged user | `crates/ds4-server/src/parse.rs:1161` `parse_messages`, `:1235` `append_tool_result_text`; `crates/ds4-server/src/render.rs:1453` | `enc-argtypes-chat` | pass |
| `sort_tool_results_by_call_order` (233) | tool_result blocks inside one user message are ordered by the preceding assistant's call order | not located (no call-order sort anywhere in `crates/ds4-server/src`) | `enc-toolresults-order` | OPEN O-03: arrival order is kept |
| `parse_tagged_text` (273) | the `<image>path</image>` text form expands to image blocks, and a malformed tag raises | not located (`<image>` appears nowhere in `crates/ds4-server/src`) | none | OPEN, no case |
| `_is_image_block` (300) | which content blocks count as images (`image`, `image_url`) | `crates/ds4-server/src/parse.rs:979` `parse_chat_image_url`, `:426` `add_image_data_uri` | none | no case: the reference harness asserts text-only |
| `_extract_image` (305) | the url/data/source normalization, and the refusal when none is present | `crates/ds4-server/src/parse.rs:385` `add_image_base64`, `:426` | none | no case |
| `_process_image_blocks` (323) | images are replaced by `<｜deepseek_image｜>` in place, nested tool-result images included, and the records keep prompt order | `crates/ds4-server/src/render.rs:1327` `append_dsml_user_content`, `crates/ds4-server/src/parse.rs:575` `write_tool_image_placeholders` | none | no case on the reference leg; the vision gate exercises the served path |
| `_validate_no_image_sp_tokens` (355) | a user-supplied image placeholder in `content` or `reasoning_content` is refused | not located | none | OPEN, no case |
| `process_image_messages` (370) | the per-message image pass, and `content` rebuilt from text blocks with `\n\n` | `crates/ds4-server/src/parse.rs:1012` `parse_openai_message_content`, `:529` `append_text_part` | none | no case |
| `_read_until_stop` (399) | the earliest-of-stops scan the whole parser is built on | `crates/ds4-server/src/tools.rs:168` `split_reasoning_content` | every `dec-*` case | pass |
| `find_last_user_index` (562) | the last user-like index; a mid-conversation system message counts as one; a tool-result-only user message does not start a turn | `crates/ds4-server/src/render.rs:1388`-`:1420`, `:222` `role_is_user_like` | `enc-plain-chat` | pass on the cases measured; the mid-system half is OPEN O-02 |
| `render_reasoning_effort` (456) | the effort header: thinking mode, index 0 only, `low/high/max` mapped to 50/75/100, `None` to high | `crates/ds4-server/src/render.rs:67` `head_effort_prefix`, `:36`/`:38` the V4.1 prefixes | `enc-think-max` pass; `enc-think-low` | OPEN O-01: the oracle writes 50, we write nothing |
| `render_tools` (509) | the tools preamble verbatim, with the schema list appended | `crates/ds4-server/src/render.rs:412` `append_dsml_tools_prompt_text` | `enc-tools-chat` | pass |
| `encode_arguments_to_dsml` (524) | one parameter element per argument key, in the call's key order, `string="true"` for a string child and its raw form otherwise; a non-dict `arguments` falls back to a single `arguments` key, a JSON string is decoded twice | `crates/ds4-server/src/render.rs:349` `append_dsml_arg`, `:375` `append_dsml_arguments_from_json` | `enc-argtypes-chat` | pass for a dict; the JSON-string and double-encoded fallbacks have no case |
| `render_message` (585) | one message's bytes, the role branches, the transition tokens, and the assistant header rule | `crates/ds4-server/src/render.rs:1374` `render_dsml_chat_choice_ex` | 5 encode cases | pass on 5 of 8; O-01, O-02, O-03 are three of its rules |
| `_drop_thinking_messages` (749) | before the last user, an assistant's `reasoning_content` is dropped, the message kept | `crates/ds4-server/src/render.rs:1470` (the render-time equivalent) | none | no case: no case carries a historical reasoning behind the last user |
| `_encode_messages_text` (770) | BOS, the loop over messages, and `effective_drop_thinking` forced false when any message carries tools | `crates/ds4-server/src/render.rs:1374`-`:1426` | all encode cases | pass |
| `encode_messages` (810) | the public entry: merge, sort, image pass, then the text render | `crates/ds4-server/src/render.rs:1374` behind the chat route | all encode cases | pass on the cases measured |
| `load_cases` (847) | the harness entry: a case-level `tools` key moves into `messages[0]`, a bare message list is wrapped | `tests/ds41_ref/ref_drive.py:61` | every encode case | harness |
| `encode_case` (872) | the harness entry: encode one case and return its images, case `thinking_mode` winning over the command line | `tests/ds41_ref/ref_drive.py:63` | every encode case | harness |

## 2. Decode side — completion to assistant message

| Oracle | Decides | Ours | Case | Verdict |
| --- | --- | --- | --- | --- |
| `parse_tool_calls` (890) | the invoke/parameter grammar: `>\n` after every open tag, one `name` per invoke, the type attribute, duplicate parameters refused | `crates/ds4-server/src/tools.rs:375` `parse_dsml_generated` | `dec-one-call`, `dec-two-calls`, `dec-complex-args`, `dec-unicode`, `dec-marker-in-value` | pass |
| `parse_message_from_completion_text` (946) | the thinking split at `</think>`, content up to EOS or the block start, no content after the block, no special token left in content or reasoning | `crates/ds4-server/src/tools.rs:375`, `:168` `split_reasoning_content` | every `dec-*` case, incl. `dec-thinking` | pass |
| `tool_calls_to_openai_format` (139) | the wire shape: `type: "function"`, `function.name`, `function.arguments`, `namespace` when present | `crates/ds4-server/src/stream.rs:165` `append_tool_calls_json` | every `dec-*` case (the fixture is the oracle's parse) | pass on names and values |

## 3. Constants and templates

Every literal the two sides must agree on.  Where a row says "other
families", the constant sits beside ours in the same table but belongs to a
model family outside this line.

| Oracle | Ours | Verdict |
| --- | --- | --- |
| `bos_token` (29) | `crates/ds4-server/src/render.rs:78` `DSML_BOS` | pass |
| `eos_token` (30) | `:81` `DSML_EOS` | pass |
| `thinking_start_token` (31) | `:86` `DSML_THINK_START` | pass |
| `thinking_end_token` (32) | `:87` `DSML_THINK_END` | pass |
| `dsml_token` (33) `｜DSML｜` | folded into the tags below | pass |
| `USER_SP_TOKEN` (35) | `:79` `DSML_USER` | pass |
| `ASSISTANT_SP_TOKEN` (36) | `:80` `DSML_ASSISTANT` | pass |
| `SYSTEM_SP_TOKEN` (426) | `:53` `DSML_SYSTEM_TOKEN` | pass |
| `LATEST_REMINDER_SP_TOKEN` (37) | not located | out of surface: the role is not on the OpenAI wire |
| `IMAGE_PLACEHOLDER` (39) | `:92` `DSML_IMAGE` | no case |
| `IMAGE_TAG_PATTERN` (40) | not located | OPEN, no case |
| `DS_TASK_SP_TOKENS` / `VALID_TASKS` (43, 51) | not located | out of surface: internal classification tasks |
| `assistant_msg_template` / `wo_eos` (60, 61) | `render.rs:1374` (EOS always written) | no case for `wo_eos` |
| `response_format_template` (64) | `crates/ds4-server/src/parse.rs:2998` `parse_output_format_value`; no render-side counterpart located | OPEN, no case |
| `tool_output_template` (68) | `crates/ds4-server/src/render.rs:1461` (`<tool_result>`) | pass via `enc-argtypes-chat` |
| `tool_calls_block_name` (428) `" calls"` | `crates/ds4-server/src/tools.rs:43` `DSML_TOOL_CALLS_START_SPACED` | pass |
| `tool_call_tag_name` (429) `" invoke"` | `:45` `DSML_INVOKE_START_SPACED`, `:46` end | pass |
| `tool_parameter_tag_name` (430) `" parameter"` | `:47` `DSML_PARAM_START_SPACED`, `:48` end | pass |
| `tool_call_template` / `tool_calls_template` (432, 435) | `crates/ds4-server/src/render.rs:535`-`:560` | pass |
| `REASONING_EFFORT_TEMPLATE` (443) | `:36` `V41_THINK_HIGH_PREFIX`, `:38` `V41_THINK_MAX_PREFIX` | pass for high and max; the low value has no constant (O-01) |
| `REASONING_EFFORT_MAPPINGS` / `DEFAULT_REASONING_EFFORT` (448, 453) | `crates/ds4-server/src/parse.rs:2257` `apply_think_aliases`, `:2817` `engine_think_mode` | OPEN for low (O-01); the `None` default is measured green |

## 4. Ours with no oracle counterpart

Functions of this repo's DSML line that the oracle does not define.  They are
kept because a served turn needs them, not because the oracle asks for them.

| Ours | Why it exists |
| --- | --- |
| `crates/ds4-server/src/tools.rs:2099` `find_tool_start` (with `crates/ds4-server/src/dsml.rs:303`) | the streaming tracker: the artifact's root spelling (quoted, unquoted, short) is not the oracle's canonical one, so the scan is tolerant where the oracle's grammar refuses |
| `crates/ds4-server/src/dsml.rs:446` `dsml_decode_state_for_text` | the incremental decode state machine that decides when markup is withheld from the stream |
| `crates/ds4-server/src/tools.rs` `DSML_*_SHORT`, `DSML_*` (24-35) | the two older root spellings the artifact still emits; the oracle knows only the spaced one |
| `crates/ds4-server/src/retry.rs:137` `try_repair_dsml` | truncated-call repair; the oracle has no repair path (matrix G8, P3) |
| `crates/ds4-server/src/render.rs:93`-`:138`, other-family markers | Motif, EXAONE, Solar, Qwen, GLM, K2 tag sets: other families, outside this line |
| `crates/ds4-server/src/render.rs:161` `syntax_for_model_id` | family dispatch; the oracle covers V4.1 only |

## 5. The open rows

Ordered by what unblocks them, with the evidence each needs.

1. O-01 low effort (row `render_reasoning_effort`, `enc-think-low`): the
   oracle writes `Reasoning Effort: 50 (range 1-100, ...)`; we write nothing.
   Needs a decision first (the oracle binds, the older C value is retired),
   then the constant, the case, and a re-pinned fixture.
2. O-02 mid-conversation system message (`enc-midsystem-chat`): the oracle
   emits `<｜System｜>content` in place and counts that index as user-like for
   the assistant header; we hoist every system message into the head
   (`crates/ds4-server/src/render.rs:1396`-`:1422`).
3. O-03 tool-result order (`enc-toolresults-order`): `sort_tool_results_by_call_order`
   has no counterpart.
4. The image surface (rows `parse_tagged_text`, `_is_image_block`,
   `_extract_image`, `_process_image_blocks`, `_validate_no_image_sp_tokens`,
   `process_image_messages`): six functions, no case.  The reference harness
   asserts text-only, so nothing here is measured against the oracle.
5. `response_format` rendering (row `response_format_template`): parsed on our
   side, not rendered, no case.
6. The namespace surface (rows `_split_tool_name`, `_tool_name_for_encoding`):
   the qualified name has no case on either side.
7. O-04 arguments bytes: ours minify, the oracle keeps the parameter body
   verbatim.  Values agree; the strings differ.  Recorded, not a defect
   against the wire, since the client echoes our string.

## 6. Reproducing

```bash
# the oracle's own bytes for a case, and its parse of a completion
python3 tests/ds41_ref/ref_drive.py encode tests/ds41_ref/cases/enc-plain-chat.json chat
python3 tests/ds41_ref/ref_drive.py parse  tests/ds41_ref/decode/dec-one-call.txt chat

# the two reference legs against the pinned fixtures
cargo test -p ds4-server --test ref_encoding
cargo test -p ds4-server --test ref_decode
```

Regenerate the fixtures only when the oracle or a case changes:
`bash tests/ds41_ref/regen_fixtures.sh`.
