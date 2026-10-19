//! Generic GGUF inventory through the Rust host's own reader — no family gate,
//! no GPU, no model. This is the "our own reader" leg of the port's gates: the
//! C/CUDA side is the oracle for the arithmetic, and this is the honest read of
//! the container itself.
//!
//!   cargo run -p ds4-core --release --example gguf_inventory -- <file.gguf> [--no-kv|--kv-values]
//!
//! It prints the header's identifying keys and, per tensor, the line the writer's
//! own manifest can be joined against:
//!   T <name> ndim=.. dims=.. type=<n>(<name>) elems=.. bytes=.. rel=.. abs=.. shard=..
//! A file whose tensor table does not parse, whose byte counts disagree with its
//! types, or whose data runs past the file is rejected by the reader, not by this
//! example (crates/ds4-core/src/tensors.rs:241-316).
//!
//! `--kv-values` prints each metadata key's VALUE instead of only its type, so a
//! written file's hparams can be read back without trusting its writer (gate 1b,
//! docs/artifact-gguf-gate1b.txt).
use ds4_core::{GgufFile, TensorInventory};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = args
        .first()
        .ok_or("usage: gguf_inventory FILE.gguf [--no-kv|--kv-values]")?;
    let kv = !args.iter().any(|a| a == "--no-kv");
    let kv_values = args.iter().any(|a| a == "--kv-values");

    let g = GgufFile::open(Path::new(file))?;
    if kv_values {
        print!("{}", g.dump_kv_values());
    } else if kv {
        print!("{}", g.dump_header_kv());
    } else {
        println!(
            "HEADER version={} n_tensors={} n_kv={} alignment={}",
            g.version, g.n_tensors, g.n_kv, g.alignment
        );
    }

    let inv = TensorInventory::from_file(Path::new(file), &g)?;
    println!(
        "DATA_POS {} ALIGN {} PAGE {} SHARDS {} N {}",
        inv.data_pos,
        inv.alignment,
        inv.page,
        inv.shards.len(),
        inv.tensors.len()
    );
    for t in &inv.tensors {
        println!("{}", t.dump_line());
    }
    Ok(())
}
