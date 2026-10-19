//! EmbeddingGemma-2 (gemma_vq) tokenizer driver for the Rust host.
//!
//! Phase 4 of embeddinggemma2-handoff.md: the port's tokenizer, measured
//! against the anchor's own token ids before any arithmetic is written. The
//! input string per item is `prompts[prompt_name] + text`, the checkpoint's
//! prompt map taken verbatim from `config_sentence_transformers.json`
//! ("Raw" is the no-prefix case, as the anchor script defines it), and the
//! BOS/EOS flags come from the artifact's own `tokenizer.ggml.*` keys.
//!
//!   cargo run -p ds4-core --release --example gemma_vq_tok -- \
//!       <artifact.gguf> <items.tsv> <config_sentence_transformers.json>
//!
//! Prints one line per item:
//!   <id> <TAB> <prompt_name> <TAB> <ntok> <TAB> <id,id,...>
//! and a summary line beginning with `TOKENS` (items, total tokens, flags).
//! tools/eg2_tok_check.sh diffs this against llama.cpp's own per-item dump.
use ds4_core::{identify_file, GgufFile, Vocab};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: gemma_vq_tok <artifact.gguf> <items.tsv> <config_sentence_transformers.json>");
        std::process::exit(2);
    }
    let g = GgufFile::open(Path::new(&args[0]))?;
    let id = identify_file(&g)?;
    let vocab = Vocab::load(&g, id.shape.family)?;

    let prompts: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)
        .map_err(|e| format!("prompt map: {e}"))?;
    let prompts = prompts
        .get("prompts")
        .and_then(|p| p.as_object())
        .ok_or("prompt map: no prompts object")?;

    let add_bos = g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
    let add_eos = g.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);

    let items = std::fs::read_to_string(&args[1])?;
    let mut n_items = 0usize;
    let mut n_tokens = 0usize;
    for line in items.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.splitn(3, '\t');
        let (item, prompt_name, text) = match (fields.next(), fields.next(), fields.next()) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => return Err(format!("items: bad line {line:?}").into()),
        };
        let prefix = if prompt_name == "Raw" {
            ""
        } else {
            prompts
                .get(prompt_name)
                .and_then(|p| p.as_str())
                .ok_or_else(|| format!("prompt_name {prompt_name:?} is not in the checkpoint's map"))?
        };
        let input = format!("{prefix}{text}");
        let mut ids: Vec<i32> = Vec::new();
        if add_bos {
            ids.push(vocab.bos_id);
        }
        ids.extend(vocab.encode_bytes(input.as_bytes()));
        if add_eos {
            ids.push(vocab.eos_id);
        }
        n_items += 1;
        n_tokens += ids.len();
        let joined = ids
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("{item}\t{prompt_name}\t{}\t{joined}", ids.len());
    }
    println!(
        "TOKENS items={n_items} tokens={n_tokens} bos={} eos={} bos_id={} eos_id={}",
        add_bos as u32, add_eos as u32, vocab.bos_id, vocab.eos_id
    );
    Ok(())
}
