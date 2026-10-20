//! Print the token strings an EmbeddingGemma-2 (gemma_vq) artifact carries for a
//! list of ids.
//!
//! Why this exists: the image path needs three ids that are NOT in the artifact's
//! metadata — `image_token_id`, its begin marker and its end marker (the
//! checkpoint's `config.json`: 258880, 255999, 258882).  The reference takes them
//! from the model's own vision metadata (llama.cpp: `glm5-next-vision.image_token_id`
//! and friends); our container has no such key, so the port reads them out of the
//! VOCABULARY it already carries, by string, and this tool is how the strings and
//! the ids are checked against each other.
//!
//!   cargo run -p ds4-core --release --example gemma_vq_vocab -- <artifact.gguf> <id,id,...>
//!
//! Prints one line per id: `<id> <TAB> <string as the vocabulary stores it>`.
use ds4_core::{identify_file, GgufFile, Vocab};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: gemma_vq_vocab <artifact.gguf> <id,id,...>");
        std::process::exit(2);
    }
    let g = GgufFile::open(Path::new(&args[0]))?;
    let id = identify_file(&g)?;
    let vocab = Vocab::load(&g, id.shape.family)?;
    for field in args[1].split(',') {
        let t: i32 = field.trim().parse()?;
        let text = vocab.token_text(t);
        println!("{t}\t{}", String::from_utf8_lossy(&text));
    }
    Ok(())
}
