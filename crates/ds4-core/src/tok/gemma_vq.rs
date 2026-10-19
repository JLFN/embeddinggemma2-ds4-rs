//! The `gemma4` pre-tokenizer and its BPE, for the gemma_vq family.
//!
//! Authority, every line below `/data/llama.cpp` at 31288c9c unless stated:
//!
//! - `src/llama-vocab.cpp:2360-2362`: `tokenizer.ggml.model == "gemma4"` selects
//!   `LLAMA_VOCAB_PRE_TYPE_GEMMA4` and `escape_whitespaces = true`.
//! - `:528-536`: that pre-type's only regex is `[^\n]+|[\n]+`, with
//!   `byte_encode = false` — raw UTF-8, no GPT-2 byte encoding. The comment
//!   there states the reason: spaces are already U+2581 by the normalizer, and
//!   BPE merge lookup asserts no newlines in a token.
//! - `:1232-1233` + `unicode_regex_split_custom_newlines` (`src/unicode.cpp`):
//!   the regex's fast path is maximal runs of `\n` and of non-`\n`.
//! - `:3537-3539` (`llama_escape_whitespace`) applied at `:3652-3654`: every
//!   space becomes U+2581 in the raw-text fragment before tokenizing. This is
//!   the checkpoint's own `Replace` normalizer
//!   (`.models/EmbeddingGemma-2-914f7f8/tokenizer.json`, `normalizer`), and its
//!   `Split`/`MergedWithPrevious` pre-tokenizer is the same rule.
//! - `:3026-3054` then `:3168-3191`: the special-token cache is every token
//!   whose attribute is CONTROL, USER_DEFINED or UNKNOWN, plus the tokens whose
//!   text matches the EOG literal list (which is where `</s>`, token_type
//!   NORMAL in the artifact, is promoted to CONTROL), sorted by descending text
//!   length. `:3401-3500` (`tokenizer_st_partition`) splits the text on exactly
//!   that cache before any BPE; the LSTRIP/RSTRIP branches there never fire for
//!   a GGUF that carries no strip bits, and this vocabulary carries none.
//! - `:589-745` (`llm_tokenizer_bpe_session::tokenize`): whole-word symbol for
//!   an all-newline word that is a vocabulary piece (`:660-668`), otherwise one
//!   symbol per UTF-8 codepoint, then the merge queue. `:264-278` is the
//!   queue's comparator: rank first, then left position. `:753-775` is the
//!   fallback for a symbol with no id: one `<0xXX>` token per byte, uppercase
//!   hex, because this pre-type is not byte-encoded.
//! - `:2660-2676`: the token_type values (`UNKNOWN` 2, `CONTROL` 3,
//!   `USER_DEFINED` 4, `BYTE` 6, `NORMAL` 1).
//! - `:2785-2791`: `add_bos` is forced true for this pre-type. BOS/EOS are the
//!   caller's step here, as they are for every other family in this file; the
//!   family's ids come from the GGUF's own `tokenizer.ggml.*_token_id` keys.
//!
//! Two named limits, neither reachable from this project's inputs:
//!
//! - Invalid UTF-8. llama.cpp decodes to codepoints and substitutes U+FFFD for
//!   an invalid sequence (`src/unicode.cpp:unicode_cpts_from_utf8`), so a
//!   malformed byte reaches BPE as three bytes. This port works on bytes and
//!   will instead emit the `<0xXX>` pieces for them. Valid UTF-8 input — every
//!   anchor item and every text this project has — is unaffected.
//! - `tokenizer.ggml.ignore_merges = true` is refused by name: absent means
//!   false in llama.cpp (`:2700` region, `ml.get_key(..., false)`), the
//!   artifact has no such key, and a merge-table-free mode would be untested
//!   guesswork rather than a port.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{GgufFile, TokError, Vocab, GGUF_VALUE_INT32, GGUF_VALUE_UINT32};

/// `LLAMA_TOKEN_TYPE_CONTROL`.
const TKN_CONTROL: u32 = 3;
/// `LLAMA_TOKEN_TYPE_USER_DEFINED`.
const TKN_USER_DEFINED: u32 = 4;
/// `LLAMA_TOKEN_TYPE_UNKNOWN`.
const TKN_UNKNOWN: u32 = 2;

/// The vocabulary this family's artifacts carry (the checkpoint's own count).
const VOCAB: usize = 262_144;

/// The literals `llama-vocab.cpp:3026-3054` force to CONTROL, verbatim. Only a
/// literal present in the vocabulary has any effect; measured on this artifact,
/// the base entry `</s>` (id 212) is the one that lands, because the three
/// gemma4 entries that list names (`<eos>` 1, `<turn|>` 106,
/// `<|tool_response>` 50) already carry `token_type` CONTROL.
const EOG_CONTROL_TEXT: &[&[u8]] = &[
    b"<|eot_id|>",
    b"<|eot|>",
    b"<|im_end|>",
    b"<|end|>",
    b"<|return|>",
    b"<|call|>",
    b"<|flush|>",
    b"<|calls|>",
    b"<end_of_turn>",
    b"<|endoftext|>",
    b"</s>",
    b"<|eom_id|>",
    b"<EOT>",
    b"_<EOT>",
    b"[EOT]",
    b"[EOS]",
    b"<|end_of_text|>",
    b"<end_of_utterance>",
    b"<eos>",
    b"<turn|>",
    b"<|tool_response>",
    b"<\xef\xbd\x9cend\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>",
    b"[e~[",
    b"<|ifm|im_end|>",
];

/// U+2581, the escaped space (`llama_escape_whitespace`).
const ESCAPED_SPACE: &[u8] = b"\xe2\x96\x81";

fn invalid(reason: &'static str) -> TokError {
    TokError::InvalidTokenizer(reason)
}

/// Family table build: the special-token cache and the BOS/EOS ids.
pub(super) fn specials(v: &mut Vocab, g: &GgufFile) -> Result<(), TokError> {
    if matches!(g.get_bool("tokenizer.ggml.ignore_merges"), Some(true)) {
        return Err(invalid("gemma_vq ignore_merges"));
    }

    let types = g
        .get_array("tokenizer.ggml.token_type")
        .filter(|a| {
            (a.typ == GGUF_VALUE_UINT32 || a.typ == GGUF_VALUE_INT32)
                && a.len == v.tokens.len() as u64
        })
        .ok_or(invalid("gemma_vq token_type table"))?;
    let types = g
        .array_le_u32s(&types)
        .map_err(|_| invalid("gemma_vq token_type table"))?;

    // CONTROL, USER_DEFINED and UNKNOWN, in id order (llama-vocab.cpp:3168-3174).
    let mut specials: Vec<(Vec<u8>, i32)> = Vec::new();
    for (i, &typ) in types.iter().enumerate() {
        if typ != TKN_CONTROL && typ != TKN_USER_DEFINED && typ != TKN_UNKNOWN {
            continue;
        }
        let token = &v.tokens[i];
        if token.is_empty() {
            continue;
        }
        specials.push((token.clone(), i as i32));
    }
    // The EOG literals, promoted to CONTROL where llama.cpp promotes them.
    for lit in EOG_CONTROL_TEXT {
        if let Some(&id) = v.token_to_id.get(*lit) {
            if !specials.iter().any(|(_, sid)| *sid == id) {
                specials.push((lit.to_vec(), id));
            }
        }
    }
    // Descending text length, as `cache_special_tokens` is ordered; ties by id
    // so the order is defined here (upstream's std::sort leaves equal lengths
    // unspecified, and no two entries of this vocabulary share a length and a
    // prefix relationship that would make the choice observable).
    specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
    v.gemma_specials = specials;

    v.bos_id = g
        .get_token_id("tokenizer.ggml.bos_token_id")
        .ok_or(invalid("gemma_vq bos_token_id"))?;
    v.eos_id = g
        .get_token_id("tokenizer.ggml.eos_token_id")
        .ok_or(invalid("gemma_vq eos_token_id"))?;
    if v.tokens.len() != VOCAB {
        return Err(invalid("gemma_vq vocabulary size"));
    }
    Ok(())
}

/// One piece of the partitioned input text: raw bytes to tokenize, or a
/// special token already resolved to its id.
enum Piece {
    Raw(usize, usize),
    Tok(i32),
}

/// Encode one input text. Mirrors `llama_vocab::tokenize`'s BPE arm for this
/// pre-type: partition, escape, split on newline runs, BPE per word.
pub(super) fn encode(v: &Vocab, text: &[u8], out: &mut Vec<i32>) {
    let mut pieces = vec![Piece::Raw(0, text.len())];
    for (lit, id) in &v.gemma_specials {
        if lit.is_empty() || text.len() < lit.len() {
            continue;
        }
        let mut next: Vec<Piece> = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match piece {
                Piece::Tok(tok) => next.push(Piece::Tok(tok)),
                Piece::Raw(start, len) => {
                    let end = start + len;
                    let mut cur = start;
                    while let Some(rel) = find(&text[cur..end], lit) {
                        let at = cur + rel;
                        if at > cur {
                            next.push(Piece::Raw(cur, at - cur));
                        }
                        next.push(Piece::Tok(*id));
                        cur = at + lit.len();
                    }
                    if cur < end {
                        next.push(Piece::Raw(cur, end - cur));
                    }
                }
            }
        }
        pieces = next;
    }

    for piece in pieces {
        match piece {
            Piece::Tok(id) => out.push(id),
            Piece::Raw(start, len) => encode_raw(v, &text[start..start + len], out),
        }
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    (0..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Escape spaces, then split into maximal runs of `\n` and non-`\n`.
fn encode_raw(v: &Vocab, raw: &[u8], out: &mut Vec<i32>) {
    let mut escaped = Vec::with_capacity(raw.len() + 8);
    for &b in raw {
        if b == b' ' {
            escaped.extend_from_slice(ESCAPED_SPACE);
        } else {
            escaped.push(b);
        }
    }
    let mut pos = 0;
    while pos < escaped.len() {
        let newline = escaped[pos] == b'\n';
        let start = pos;
        while pos < escaped.len() && (escaped[pos] == b'\n') == newline {
            pos += 1;
        }
        encode_word(v, &escaped[start..pos], out);
    }
}

struct Sym {
    start: usize,
    len: usize,
    prev: isize,
    next: isize,
}

/// One word through the merge queue, then id lookup with the byte fallback.
fn encode_word(v: &Vocab, word: &[u8], out: &mut Vec<i32>) {
    if word.is_empty() {
        return;
    }
    // An all-newline word that is a vocabulary piece is emitted whole
    // (llama-vocab.cpp:660-668, the gemma4 fix).
    if word.iter().all(|&b| b == b'\n') {
        if let Some(&id) = v.token_to_id.get(word) {
            out.push(id);
            return;
        }
    }

    let mut syms: Vec<Sym> = Vec::with_capacity(word.len());
    let mut pos = 0;
    while pos < word.len() {
        let n = utf8_len_utf8(word[pos]).min(word.len() - pos);
        syms.push(Sym {
            start: pos,
            len: n,
            prev: syms.len() as isize - 1,
            next: if pos + n == word.len() {
                -1
            } else {
                syms.len() as isize + 1
            },
        });
        pos += n;
    }

    // rank, then left position (llama-vocab.cpp:264-278), with the joined text
    // carried for the staleness check at :688-694.
    let mut queue: BinaryHeap<Reverse<(i32, usize, Vec<u8>)>> = BinaryHeap::new();
    for i in 1..syms.len() {
        push_bigram(v, word, &syms, &mut queue, i - 1, i);
    }
    while let Some(Reverse((_, left, joined))) = queue.pop() {
        let right = syms[left].next;
        if right < 0 || syms[left].len == 0 || syms[right as usize].len == 0 {
            continue;
        }
        let rightu = right as usize;
        let mut text = Vec::with_capacity(syms[left].len + syms[rightu].len);
        text.extend_from_slice(&word[syms[left].start..syms[left].start + syms[left].len]);
        text.extend_from_slice(&word[syms[rightu].start..syms[rightu].start + syms[rightu].len]);
        if text != joined {
            continue;
        }
        syms[left].len += syms[rightu].len;
        syms[rightu].len = 0;
        syms[left].next = syms[rightu].next;
        let after_right = syms[rightu].next;
        if after_right >= 0 {
            syms[after_right as usize].prev = left as isize;
        }
        let (prev, next) = (syms[left].prev, syms[left].next);
        if prev >= 0 {
            push_bigram(v, word, &syms, &mut queue, prev as usize, left);
        }
        if next >= 0 {
            push_bigram(v, word, &syms, &mut queue, left, next as usize);
        }
    }

    // Emission in symbol order: merges fold a right symbol into its left one,
    // so the live symbols keep their creation order, which is the order
    // llama.cpp's next-chain walk produces (llama-vocab.cpp:757-775).
    for sym in &syms {
        if sym.len == 0 {
            continue;
        }
        let piece = &word[sym.start..sym.start + sym.len];
        if let Some(&id) = v.token_to_id.get(piece) {
            out.push(id);
            continue;
        }
        for &byte in piece {
            let mut buf = [b'<', b'0', b'x', 0, 0, b'>'];
            buf[3] = HEX[(byte >> 4) as usize];
            buf[4] = HEX[(byte & 15) as usize];
            if let Some(&id) = v.token_to_id.get(&buf[..]) {
                out.push(id);
            }
        }
    }
}

const HEX: &[u8; 16] = b"0123456789ABCDEF";

fn push_bigram(
    v: &Vocab,
    word: &[u8],
    syms: &[Sym],
    queue: &mut BinaryHeap<Reverse<(i32, usize, Vec<u8>)>>,
    left: usize,
    right: usize,
) {
    if left >= syms.len() || right >= syms.len() {
        return;
    }
    let (a, b) = (&syms[left], &syms[right]);
    if a.len == 0 || b.len == 0 {
        return;
    }
    let a_text = &word[a.start..a.start + a.len];
    let b_text = &word[b.start..b.start + b.len];
    // `find_bpe_rank` keys the table on the pair joined by one space
    // (llama-vocab.cpp:4347-4357); an escaped word has no literal space, so the
    // key is unambiguous.
    let mut key = Vec::with_capacity(a_text.len() + 1 + b_text.len());
    key.extend_from_slice(a_text);
    key.push(b' ');
    key.extend_from_slice(b_text);
    let Some(&rank) = v.merge_rank.get(&key) else {
        return;
    };
    let mut joined = Vec::with_capacity(a_text.len() + b_text.len());
    joined.extend_from_slice(a_text);
    joined.extend_from_slice(b_text);
    queue.push(Reverse((rank, left, joined)));
}

/// The high-nibble UTF-8 length table (`src/unicode.cpp:16-20`), which differs
/// from a strict decoder on invalid lead bytes and is what the reference uses.
fn utf8_len_utf8(byte: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(byte >> 4) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::ModelFamily;
    use std::collections::HashMap;

    /// A `Vocab` for the pre-tokenizer alone: no GGUF, no artifact. The
    /// special-token cache is the list a case needs, in the order `specials()`
    /// would sort it (descending text length).
    fn vocab(tokens: &[&[u8]], merges: &[&[u8]], specials: &[(&[u8], i32)]) -> Vocab {
        let tokens: Vec<Vec<u8>> = tokens.iter().map(|t| t.to_vec()).collect();
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as i32))
            .collect();
        let merges: Vec<Vec<u8>> = merges.iter().map(|m| m.to_vec()).collect();
        let merge_rank = merges
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), i as i32))
            .collect();
        Vocab {
            family: ModelFamily::GemmaVq,
            is_k2_horizon: false,
            qwen_input: false,
            tokens,
            token_to_id,
            merges,
            merge_rank,
            user_defined: HashMap::new(),
            user_defined_max_len: 0,
            user_defined_first: [false; 256],
            motif3_added_first: [false; 256],
            gemma_specials: specials.iter().map(|(t, i)| (t.to_vec(), *i)).collect(),
            bos_id: 2,
            eos_id: 1,
            system_id: 0,
            eot_id: -1,
            im_start_id: -1,
            im_content_id: -1,
            im_end_id: -1,
            user_id: 0,
            assistant_id: 0,
            start_of_turn_id: 0,
            end_of_turn_id: 0,
            tool_id: 0,
            reference_id: 0,
            plan_start_id: 0,
            plan_end_id: 0,
            observation_id: 0,
            sop_id: 0,
            think_start_id: 0,
            think_end_id: 0,
            tool_call_start_id: -1,
            tool_call_end_id: -1,
            tool_response_start_id: -1,
            tool_response_end_id: -1,
            arg_key_start_id: -1,
            arg_key_end_id: -1,
            arg_value_start_id: -1,
            arg_value_end_id: -1,
            latent_start_id: 0,
            latent_pad_id: 0,
            latent_end_id: 0,
            tool_schema_start_id: -1,
            tool_schema_end_id: -1,
            dsml_id: 0,
            image_id: -1,
            dots3_endofsystem_id: -1,
            dots3_endofuser_id: -1,
            dots3_endoftext_id: -1,
        }
    }

    fn run(v: &Vocab, text: &str) -> Vec<i32> {
        let mut out = Vec::new();
        encode(v, text.as_bytes(), &mut out);
        out
    }

    #[test]
    fn space_becomes_u2581_and_merges_run_over_it() {
        // The checkpoint's Replace normalizer, which llama.cpp mirrors by
        // escaping spaces before the newline split (llama-vocab.cpp:3652-3654).
        let v = vocab(
            &[
                b"\xe2\x96\x81",
                b"t",
                b"h",
                b"e",
                b"\xe2\x96\x81t",
                b"\xe2\x96\x81th",
                b"\xe2\x96\x81the",
            ],
            &[
                b"\xe2\x96\x81 t",
                b"\xe2\x96\x81t h",
                b"\xe2\x96\x81th e",
            ],
            &[],
        );
        assert_eq!(run(&v, " the"), vec![6]);
        // No merge across the boundary: the escaped space is its own piece
        // when the pair is not in the table.
        let v2 = vocab(&[b"\xe2\x96\x81", b"a", b"b", b"ab"], &[b"a b"], &[]);
        assert_eq!(run(&v2, "a b"), vec![1, 0, 2]);
        assert_eq!(run(&v2, "ab"), vec![3]);
    }

    #[test]
    fn newline_runs_split_and_a_whole_newline_piece_wins() {
        let v = vocab(
            &[b"a", b"b", b"\n", b"\n\n", b"\n\n\n"],
            &[b"\n \n"],
            &[],
        );
        // "[^\n]+|[\n]+" (llama-vocab.cpp:534): a non-newline run, then a run
        // of newlines.
        assert_eq!(run(&v, "a\n\nb"), vec![0, 3, 1]);
        // A run that is a vocabulary piece is emitted whole
        // (llama-vocab.cpp:660-668); a longer run that is not goes through the
        // merge queue instead, which folds its pairs into "\n\n" pieces.
        assert_eq!(run(&v, "a\n\n\n\nb"), vec![0, 3, 3, 1]);
    }

    #[test]
    fn a_symbol_with_no_id_falls_back_to_its_bytes() {
        // gemma4 is not byte-encoded, so the fallback is <0xXX> with uppercase
        // hex (llama-vocab.cpp:753-775). "é" is U+00E9 = C3 A9.
        let v = vocab(&[b"a", b"<0xC3>", b"<0xA9>"], &[], &[]);
        assert_eq!(run(&v, "aé"), vec![0, 1, 2]);
    }

    #[test]
    fn the_special_partition_emits_ids_and_leaves_the_rest() {
        // llama-vocab.cpp:3401-3500: the partition runs before any escaping or
        // BPE, on the cache from :3168-3191. A partial literal is ordinary text.
        let v = vocab(
            &[b"a", b"b", b"<", b"e", b"o", b"\xe2\x96\x81"],
            &[],
            &[(b"<eos>", 7)],
        );
        assert_eq!(run(&v, "a<eos>b"), vec![0, 7, 1]);
        assert_eq!(run(&v, "<eos>"), vec![7]);
        assert_eq!(run(&v, "<eos"), vec![2, 3, 4]);
        assert_eq!(run(&v, "a<eo b"), vec![0, 2, 3, 4, 5, 1]);
    }
}
