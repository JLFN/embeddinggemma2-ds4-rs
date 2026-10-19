//! leg 2 of gate 4: the Rust host's own route over the frozen item set.
//!
//! The Rust host owns the tokenizer and the artifact's identity/layout; the C
//! reference (`tests/parity/eg2vq_embd_c_oracle`) owns the arithmetic. This
//! test drives the whole path from the host side and checks it against the
//! frozen anchor (`tests/expected/anchor-embeddinggemma2.jsonl`, llama.cpp at
//! the pin through the baseline GGUF):
//!
//!   * the host's token ids must give the anchor's own per-item token count
//!     (the anchor file carries `ntok` per item), so the tokenizer is measured
//!     against the reference without reading its verbose log;
//!   * the pooled, normalized 768-dim vector must agree with the anchor's at
//!     mean cosine >= 0.9999, with the three MRL widths re-normalized after
//!     the slice and reported beside it.
//!
//! Skips when the environment does not name the pieces (the tree's other
//! oracle tests do the same):
//!   DS4_EG2VQ_C_ORACLE  tests/parity/eg2vq_embd_c_oracle
//!   DS4_EG2VQ_IDENTITY  the GGUF the host identifies and takes its vocabulary
//!                       from: always the `gemma_vq` artifact, whose arch name
//!                       this tree's identity route owns (the baseline carries
//!                       llama.cpp's "gemma-embedding2" instead)
//!   DS4_EG2VQ_WEIGHTS   the GGUF handed to the oracle: the baseline BF16 for
//!                       the declared gate (the anchor's own model), the VQ
//!                       artifact for the container report
//!   DS4_EG2VQ_ITEMS     tests/anchor/anchor-items.tsv
//!   DS4_EG2VQ_PROMPTS   .models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
//!   DS4_EG2VQ_ANCHOR    tests/expected/anchor-embeddinggemma2.jsonl
//!   DS4_EG2VQ_MIN_COS   optional: the mean-cosine floor to assert. Unset means
//!                       report only, which is what the VQ artifact leg wants —
//!                       its distance from the anchor is the quantization, not
//!                       the port (RECIPE.md section 5), and phase 5 owns it.
use std::path::PathBuf;
use std::process::Command;

use ds4_core::{identify_file, GgufFile, Vocab};

struct Item {
    id: String,
    ntok: usize,
    ids: Vec<i32>,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var(name).ok().map(PathBuf::from)
}

#[test]
fn host_route_matches_the_anchor() {
    let (Some(oracle), Some(identity), Some(weights), Some(items), Some(prompts), Some(anchor)) = (
        env_path("DS4_EG2VQ_C_ORACLE"),
        env_path("DS4_EG2VQ_IDENTITY"),
        env_path("DS4_EG2VQ_WEIGHTS"),
        env_path("DS4_EG2VQ_ITEMS"),
        env_path("DS4_EG2VQ_PROMPTS"),
        env_path("DS4_EG2VQ_ANCHOR"),
    ) else {
        return;
    };

    // The host's own identity + vocabulary route, off the artifact itself.
    let g = GgufFile::open(&identity).expect("artifact opens");
    let id = identify_file(&g).expect("artifact identifies");
    let vocab = Vocab::load(&g, id.shape.family).expect("family vocabulary loads");

    let prompt_map: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&prompts).expect("prompt map reads"))
            .expect("prompt map parses");
    let prompt_map = prompt_map["prompts"].as_object().expect("prompts object");

    // Item inputs, exactly as the anchor's own producer builds them.
    let items_text = std::fs::read_to_string(&items).expect("items read");
    let mut rows: Vec<Item> = Vec::new();
    for line in items_text.lines() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        assert_eq!(f.len(), 3, "bad item line {line:?}");
        let prefix = if f[1] == "Raw" {
            ""
        } else {
            prompt_map[f[1]].as_str().expect("prompt name is in the map")
        };
        let mut ids = vec![vocab.bos_id];
        ids.extend(vocab.encode_bytes(format!("{prefix}{}", f[2]).as_bytes()));
        ids.push(vocab.eos_id);
        rows.push(Item {
            id: f[0].to_string(),
            ntok: ids.len(),
            ids,
        });
    }
    assert!(!rows.is_empty(), "no items read");

    // The ids file the C oracle consumes.
    let ids_path = std::env::temp_dir().join("eg2vq-ref-ids.tsv");
    let mut ids_file = String::new();
    for r in &rows {
        let joined = r
            .ids
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",");
        ids_file.push_str(&format!("{}\t{}\t{}\t{}\n", r.id, "-", r.ntok, joined));
    }
    std::fs::write(&ids_path, ids_file).expect("ids file writes");

    let out = Command::new(&oracle)
        .arg(&weights)
        .arg(&ids_path)
        .output()
        .expect("oracle runs");
    assert!(
        out.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for line in stdout.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 3, "bad oracle line");
        let v: Vec<f32> = cols[2]
            .split(' ')
            .map(|x| x.parse::<f32>().expect("float"))
            .collect();
        assert_eq!(v.len(), 768, "oracle vector width");
        vectors.push(v);
    }
    assert_eq!(vectors.len(), rows.len(), "one vector per item");

    // The anchor's own per-item counts and vectors.
    let anchor_text = std::fs::read_to_string(&anchor).expect("anchor reads");
    let mut want: std::collections::HashMap<String, (usize, Vec<f32>)> = Default::default();
    for line in anchor_text.lines() {
        let v: serde_json::Value = serde_json::from_str(line).expect("anchor line");
        if v["kind"] != "item" {
            continue;
        }
        let dims = v["dims"]["768"].as_array().expect("768 dims");
        want.insert(
            v["id"].as_str().expect("id").to_string(),
            (
                v["ntok"].as_u64().expect("ntok") as usize,
                dims.iter().map(|x| x.as_f64().expect("f64") as f32).collect(),
            ),
        );
    }

    let mut count_bad = 0usize;
    let (mut min_cos768, mut sum_cos768, mut n) = (f32::MAX, 0.0f32, 0usize);
    let mut widths = vec![(768usize, f32::MAX, 0.0f32, 0usize); 4];
    let mut worst = String::new();
    for (i, r) in rows.iter().enumerate() {
        let (ntok, refv) = want.get(&r.id).expect("item is in the anchor");
        if *ntok != r.ntok {
            count_bad += 1;
            eprintln!("count mismatch {}: anchor {} host {}", r.id, ntok, r.ntok);
        }
        let got = &vectors[i];
        for (k, w) in [768usize, 512, 256, 128].iter().enumerate() {
            let a: Vec<f32> = refv[..*w].to_vec();
            let b: Vec<f32> = got[..*w].to_vec();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let c = dot / (na * nb);
            widths[k].1 = widths[k].1.min(c);
            widths[k].2 += c;
            widths[k].3 += 1;
            if *w == 768 {
                if c < min_cos768 {
                    min_cos768 = c;
                    worst = r.id.clone();
                }
                sum_cos768 += c;
                n += 1;
            }
        }
    }
    assert_eq!(count_bad, 0, "token counts differ from the anchor");
    let mean768 = sum_cos768 / n as f32;
    for (w, minc, sumc, cnt) in &widths {
        eprintln!(
            "width {w}: min {:.9} mean {:.9} over {cnt}",
            minc,
            sumc / *cnt as f32
        );
    }
    eprintln!("768: mean {mean768:.9} min {min_cos768:.9}, worst item {worst}");
    match std::env::var("DS4_EG2VQ_MIN_COS") {
        Ok(v) if !v.trim().is_empty() => {
            let floor: f32 = v.parse().expect("DS4_EG2VQ_MIN_COS is a number");
            assert!(
                mean768 >= floor,
                "768 mean cosine {mean768} is below the declared {floor}"
            );
        }
        _ => eprintln!("(no DS4_EG2VQ_MIN_COS: derived vectors, nothing asserted)"),
    }
}
