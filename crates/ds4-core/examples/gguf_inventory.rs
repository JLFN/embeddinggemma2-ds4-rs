//! Generic GGUF inventory through the Rust host's own reader — no family gate,
//! no GPU, no model. This is the "our own reader" leg of the port's gates: the
//! C/CUDA side is the oracle for the arithmetic, and this is the honest read of
//! the container itself.
//!
//!   cargo run -p ds4-core --release --example gguf_inventory -- <file.gguf> [--no-kv|--kv-values] [--family]
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
//!
//! `--family` adds the two checks that need the identified family rather than the
//! container alone: `identify` (the arch route and the family gate) and the
//! family layout, resolved through `BindPlan` — the same `validate_layouts` call
//! `Model::open` makes. It stops there: vocab, chat template and the graph are
//! the model path's business, so a family whose tokenizer is not ported still
//! reports its inventory and layout here.
use ds4_core::{BindPlan, GgufFile, TensorInventory};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let file = args
        .first()
        .ok_or("usage: gguf_inventory FILE.gguf [--no-kv|--kv-values] [--family]")?;
    let kv = !args.iter().any(|a| a == "--no-kv");
    let kv_values = args.iter().any(|a| a == "--kv-values");
    let family = args.iter().any(|a| a == "--family");

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

    if family {
        // Every verdict is printed, and any failure also fails the run: a gate
        // reads the exit status, an operator reads the lines.
        let mut failures: Vec<String> = Vec::new();
        match ds4_core::identify_file(&g) {
            Ok(id) => {
                println!("{}", id.identify_line());
                match ds4_core::validate_file(&g, &id.shape) {
                    Ok(()) => println!("META ok"),
                    Err(e) => {
                        println!("META failed {}", e.token());
                        failures.push(format!("metadata: {}", e.token()));
                    }
                }
                let plan = BindPlan::resolve(id.shape, &inv);
                match plan.missing_required().first() {
                    Some(name) => {
                        println!("BIND missing {name}");
                        failures.push(format!("bind: missing {name}"));
                    }
                    None => println!("BIND ok slots={}", plan.slots.len()),
                }
                match ds4_core::validate_layouts(&plan) {
                    Ok(()) => println!(
                        "LAYOUT ok specs={}",
                        ds4_core::expected_layouts(&id.shape).len()
                    ),
                    Err(e) => {
                        println!("LAYOUT failed {}", e.token());
                        failures.push(format!("layout: {}", e.token()));
                    }
                }
            }
            Err(e) => {
                println!("IDENTIFY failed {}", e.token());
                failures.push(format!("identify: {}", e.token()));
            }
        }
        if !failures.is_empty() {
            return Err(failures.join("; ").into());
        }
    }
    Ok(())
}
