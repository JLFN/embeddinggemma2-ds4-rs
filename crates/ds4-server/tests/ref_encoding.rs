//! The reference leg, encode side: our DS41 prompt bytes against the official
//! DeepSeek-V4.1 encoding implementation (the oracle).
//!
//! The oracle is the operator-supplied file `/data/ds4/test/encoding.py`
//! (sha256 13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2),
//! untracked in the read-only reference tree `/data/ds4` at fc80bd6.  Each
//! fixture beside this test is the oracle's own output for the case of the
//! same name under `tests/ds41_ref/cases/`, regenerated with
//! `bash tests/ds41_ref/regen_fixtures.sh`; driving the oracle is what
//! `tests/ds41_ref/ref_drive.py` does, and it refuses any other revision.
//! The fixtures therefore pin bytes we did not author.
//!
//! Coverage is traced in `docs/ds41-dsml-reference-matrix.md`: every CTR
//! requirement and every inventory gap names the case that exercises it, and
//! a case that does not yet hold is marked OPEN there rather than dropped.

use ds4_server::{
    render_dsml_chat_choice_ex, ChatMsg, DsmlHead, ThinkMode, ToolCall, ToolChoice,
};

/// The head our ds41 route passes: the engine's V4.1 encoding.py head with the
/// tokenizer's system token (render.rs `dsml_head`).
const V41: DsmlHead = DsmlHead::V41 { system_token: true };

fn msg(role: &str, content: &str) -> ChatMsg {
    ChatMsg {
        role: role.into(),
        content: content.into(),
        ..Default::default()
    }
}

fn assistant(content: &str, calls: Vec<ToolCall>) -> ChatMsg {
    ChatMsg {
        role: "assistant".into(),
        content: content.into(),
        calls,
        ..Default::default()
    }
}

fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: arguments.into(),
    }
}

fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ds41-ref")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Report the first differing byte, with both sides escaped, so a mismatch is
/// readable without dumping two long prompts.
fn diff_report(got: &[u8], want: &[u8]) -> String {
    let at = got
        .iter()
        .zip(want)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| got.len().min(want.len()));
    let lo = at.saturating_sub(60);
    format!(
        "first difference at byte {at} (got {} bytes, want {} bytes)\n  got  ...{}\n  want ...{}",
        got.len(),
        want.len(),
        String::from_utf8_lossy(&got[lo..(at + 60).min(got.len())]),
        String::from_utf8_lossy(&want[lo..(at + 60).min(want.len())]),
    )
}

/// Render one case and compare with the oracle's fixture of the same name.
fn check(name: &str, msgs: &[ChatMsg], schemas: &str, mode: ThinkMode) {
    let got = render_dsml_chat_choice_ex(msgs, schemas, mode, ToolChoice::Auto, V41)
        .unwrap_or_else(|e| panic!("{name}: render refused: {}", e.0));
    let want = fixture(&format!("{name}.prompt.txt"));

    assert_eq!(
        String::from_utf8_lossy(&got),
        String::from_utf8_lossy(&want),
        "{name}: {}",
        diff_report(&got, &want)
    );
}

/// A case's tool, as the route receives it: one client JSON value per line
/// (render.rs:475-520 reformats each to json.dumps shape, which is what the
/// oracle emits).  Each constant mirrors the `tools` array of the case that
/// uses it -- a schema that declares a property its case does not is a test
/// bug, not a carriage difference.
const READ_FILE_SCHEMA: &str = concat!(
    r#"{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"target_file":{"type":"string"},"limit":{"type":"integer"}},"required":["target_file"]}}"#
);
const READ_FILE_SCHEMA_MIN: &str = concat!(
    r#"{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"target_file":{"type":"string"}},"required":["target_file"]}}"#
);

#[test]
fn tools_prompt_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-tools-chat", &msgs, READ_FILE_SCHEMA, ThinkMode::None);
}

#[test]
fn plain_chat_prompt_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Say hello."),
        assistant("Hello.", Vec::new()),
        msg("user", "Say it again."),
    ];
    check("enc-plain-chat", &msgs, "", ThinkMode::None);
}

/// The type attribute: `string="true"` for a string child, `string="false"`
/// with the value in json.dumps shape for every other kind
/// (encoding.py encode_arguments_to_dsml:524-556).  The echoed arguments are
/// the client's raw JSON string, as requirement 2.3.5 replays them.
#[test]
fn typed_parameters_match_official_encoding() {
    let args = r#"{"target_file": "/tmp/x.txt", "limit": 40, "force": true, "ratio": 1.5, "note": null, "deps": ["a", "b"], "opts": {"k": 1}}"#;
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file and label it."),
        assistant("Reading it now.", vec![call("call_1", "read_file", args)]),
        msg("tool", "file body"),
        msg("user", "Now summarise it."),
    ];
    let schemas = concat!(
        r#"{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"target_file":{"type":"string"},"limit":{"type":"integer"},"force":{"type":"boolean"},"ratio":{"type":"number"},"note":{"type":"string"},"deps":{"type":"array","items":{"type":"string"}},"opts":{"type":"object"}},"required":["target_file"]}}"#
    );
    check("enc-argtypes-chat", &msgs, schemas, ThinkMode::None);
}

#[test]
fn thinking_tools_prompt_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-think-tools", &msgs, READ_FILE_SCHEMA_MIN, ThinkMode::High);
}

/// The numeric reasoning-effort header: 75 for the default, 100 for max
/// (encoding.py:444-472).  Only thinking mode renders it, and only at index 0.
#[test]
fn thinking_max_effort_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-think-max", &msgs, READ_FILE_SCHEMA_MIN, ThinkMode::Max);
}

/// OPEN (matrix row O-01): the oracle renders "Reasoning Effort: 50" for the
/// low effort (encoding.py:449-451), while the reference's DeepSeek-41 branch
/// renders no effort line for DS4_THINK_LOW at all -- ds4_think_mode_level
/// returns -1 outside the numeric range and only HIGH/MAX are pinned to 75/100
/// (ds4.c:44201-44209).  Low effort is spelled as the numeric level 50 on both
/// references.  The case stays here so the row is re-measured when the effort
/// surface is decided.
#[test]
#[ignore = "OPEN O-01: the oracle's low=50 prefix has no counterpart in the reference's DS4_THINK_LOW arm"]
fn thinking_low_effort_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
    ];
    check("enc-think-low", &msgs, READ_FILE_SCHEMA_MIN, ThinkMode::Low);
}

/// OPEN (matrix row O-02): a mid-conversation system message.  Both references
/// write the system token again and let the message introduce its own turn
/// (encoding.py:621-627, :733-739; ds4.c:44431-44443 ds41_chat_system_marker
/// pushes the system token once a user or assistant token has been emitted).
/// Our renderer hoists every system message into the head instead, so the
/// prompt differs from both: measured, the case's "Answer in one line." lands
/// before the tools preamble rather than after the user turn.
#[test]
#[ignore = "OPEN O-02: a mid-conversation system message is hoisted into the head (render.rs:1395-1416)"]
fn mid_conversation_system_matches_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Read the file /tmp/x.txt"),
        msg("system", "Answer in one line."),
    ];
    check("enc-midsystem-chat", &msgs, READ_FILE_SCHEMA_MIN, ThinkMode::None);
}

/// OPEN (matrix row O-03): tool results are ordered by the preceding
/// assistant's call order, not by arrival (encoding.py:234-266
/// sort_tool_results_by_call_order).  Our renderer keeps arrival order:
/// measured on this case, the oracle's block carries body a then body b, ours
/// body b then body a; every other byte of the turn matches.
#[test]
#[ignore = "OPEN O-03: tool results keep arrival order, the oracle orders them by call order"]
fn tool_results_in_call_order_match_official_encoding() {
    let msgs = [
        msg("system", "You are a helpful assistant."),
        msg("user", "Check both files."),
        assistant(
            "Checking.",
            vec![
                call("call_a", "read_file", r#"{"target_file": "/tmp/a.txt"}"#),
                call("call_b", "read_file", r#"{"target_file": "/tmp/b.txt"}"#),
            ],
        ),
        msg("tool", "body b"),
        msg("tool", "body a"),
        msg("user", "Summarise both."),
    ];
    check(
        "enc-toolresults-order",
        &msgs,
        READ_FILE_SCHEMA_MIN,
        ThinkMode::None,
    );
}
