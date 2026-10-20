//! Phase 8, unit 26: the embedding wire's image sequence, printed for a check.
//!
//! The route builds a media item's token ids in Rust (`ds4_core::embed_media_ids`,
//! the same function `Model::embed_item_media_ids` calls) and this driver prints
//! them, so gate 26 can compare the wire's own sequence against the anchor's
//! frozen one WITHOUT running a tower: with `task` Raw and the text `<|image|>`,
//! the result must be exactly the reference's 92 positions --
//!
//!   [2 BOS][255999 <|image>] + 88 x 258880 + [258882 <image|>][1 EOS], span [2, 90)
//!
//! -- which is the sequence the frozen splices in tests/vision/reference/ were
//! produced with (tests/parity/eg2vq_vision_anchor.cpp:106-113 and
//! ds4-rs/tests/parity/eg2vq_splice_cuda_oracle.c's own comment).
//!
//!   cargo run -p ds4-core --release --example gemma_vq_media_ids -- \
//!       <artifact.gguf> <text> <n_soft> <task|Raw>
//!
//! Prints one line: `<n_ids> <TAB> <id,id,...>` and then `SPAN <start> <n_soft>`.
//! Only the vocabulary is loaded, never the weights, so this is a cheap check.
use ds4_core::{identify_file, st_prompt, GgufFile, Vocab};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        eprintln!("usage: gemma_vq_media_ids <artifact.gguf> <text> <n_soft> <task|Raw>");
        std::process::exit(2);
    }
    let g = GgufFile::open(Path::new(&args[0]))?;
    let id = identify_file(&g)?;
    let vocab = Vocab::load(&g, id.shape.family)?;
    let n_soft: u32 = args[2].parse()?;
    let prefix = st_prompt(&args[3])?;
    let add_bos = g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(false);
    let add_eos = g.get_bool("tokenizer.ggml.add_eos_token").unwrap_or(false);
    let ids = ds4_core::embed_media_ids(&vocab, &prefix, &args[1], n_soft, add_bos, add_eos)?;

    let joined: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
    println!("{}\t{}", ids.len(), joined.join(","));
    let start = ids
        .iter()
        .position(|id| *id == ds4_core::GEMMA_VQ_IMAGE_TOKEN_ID)
        .unwrap_or(usize::MAX);
    println!("SPAN {start} {n_soft}");
    Ok(())
}
