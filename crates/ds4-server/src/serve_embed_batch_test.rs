//! Unit 27: the embedding route's batching, at the HTTP level.
//!
//! What these tests are for: `/v1/embeddings` prepares every item and then
//! embeds the TEXT items of a request in ONE tower call (`embed_gemma_vq_batch`,
//! the packed row block) while each item that carries an image keeps its own
//! call (`embed_gemma_vq_image`, one device vision workspace per engine).  That
//! grouping is a ROUTE decision -- the wire's answer cannot show it, because a
//! batched item's vector is its single-item vector -- so it is tested here with
//! a recording engine that implements the DecodeIo trait's embedding half and
//! keeps every call it was handed, over a real socket exactly as
//! `serve_embed_media_test.rs` does.
//!
//! The end-to-end path with the real artifact and the real tower is gate 27's
//! job (`tools/eg2_embed_batch_gate27.sh`), and the batch's equality claim is
//! the device oracle's.
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use crate::generate::{DecodeIo, GenerateError, VisionProbe};
use crate::serve::{handle_client_inner, ServerConfig, ServerInner};

/// Every tower call the route made, in order.
#[derive(Debug, Default)]
struct Calls {
    /// one entry per `embed_gemma_vq_batch`: the packed ids and the spans
    batches: Vec<(Vec<i32>, Vec<u32>)>,
    /// one entry per `embed_gemma_vq_image`: the item's ids
    images: Vec<Vec<i32>>,
}

/// An embedding engine that answers the trait's embedding half and records it.
/// The vectors it returns are distinguishable per item (item `i` is 1.0 + i), so
/// the response's order can be read back from the JSON itself.
struct Recorder {
    calls: Arc<Mutex<Calls>>,
}

impl DecodeIo for Recorder {
    fn model_id(&self) -> i32 {
        0
    }
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        Ok(text.bytes().map(i32::from).collect())
    }
    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        Ok(text.iter().map(|b| i32::from(*b)).collect())
    }
    fn token_text(&self, _token: i32) -> Result<Vec<u8>, GenerateError> {
        Ok(Vec::new())
    }
    fn token_is_stop(&self, _token: i32) -> bool {
        false
    }
    fn sync(&mut self, _tokens: &[i32]) -> Result<(), GenerateError> {
        Ok(())
    }
    fn eval(&mut self, _token: i32) -> Result<(), GenerateError> {
        Ok(())
    }
    fn pos(&self) -> i32 {
        0
    }
    fn ctx(&self) -> i32 {
        2048
    }
    fn generation(&self) -> u64 {
        0
    }
    fn invalidate(&mut self) {}
    fn sample(
        &mut self,
        _temperature: f32,
        _top_k: i32,
        _top_p: f32,
        _min_p: f32,
        _rng: &mut u64,
    ) -> i32 {
        -1
    }
    fn shutdown(&mut self) -> Result<(), GenerateError> {
        Ok(())
    }
    fn embeds_text(&self) -> bool {
        true
    }
    /// The artifact's own flags are BOS and EOS on, and this fake marks them the
    /// way the artifact's ids do (2 and 1) so the spans are checkable by eye.
    fn embed_item_ids(
        &self,
        prefix: &str,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<i32>, GenerateError> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(2);
        }
        ids.extend(format!("{prefix}{text}").bytes().map(i32::from));
        if add_eos {
            ids.push(1);
        }
        Ok(ids)
    }
    fn embed_gemma_vq_batch(
        &self,
        ids: &[i32],
        item_off: &[u32],
    ) -> Result<Vec<f32>, GenerateError> {
        self.calls
            .lock()
            .unwrap()
            .batches
            .push((ids.to_vec(), item_off.to_vec()));
        let n_items = item_off.len() - 1;
        let mut out = vec![0.0f32; n_items * 768];
        for i in 0..n_items {
            for j in 0..768 {
                out[i * 768 + j] = 1.0 + i as f32;
            }
        }
        Ok(out)
    }
    /// The sequence's own construction, in the same shape the real one has: the
    /// model's begin marker, one placeholder per soft row, its end marker.  The
    /// route only carries the result.
    fn embed_item_media_ids(
        &self,
        _prefix: &str,
        _text: &str,
        n_soft: u32,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<i32>, GenerateError> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(2);
        }
        ids.push(255999);
        for _ in 0..n_soft {
            ids.push(ds4_core::GEMMA_VQ_IMAGE_TOKEN_ID);
        }
        ids.push(258882);
        if add_eos {
            ids.push(1);
        }
        Ok(ids)
    }
    fn vision_probe(&self, _data: &[u8]) -> Result<VisionProbe, GenerateError> {
        Ok(VisionProbe { token_count: 3 })
    }
    fn vision_encode_rows(
        &self,
        _data: &[u8],
        expect_rows: u32,
    ) -> Result<Vec<f32>, GenerateError> {
        Ok(vec![0.0f32; expect_rows as usize * 512])
    }
    fn embed_gemma_vq_image(
        &self,
        ids: &[i32],
        _soft: &[f32],
    ) -> Result<Vec<f32>, GenerateError> {
        self.calls.lock().unwrap().images.push(ids.to_vec());
        Ok(vec![7.0f32; 768])
    }
}

/// One request through the real handler, one response, as bytes.
fn drive(cfg: &ServerConfig, request: &str, engine: &mut dyn DecodeIo) -> String {
    let inner = Mutex::new(ServerInner::from_cfg(cfg));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    write!(client, "{request}").unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    handle_client_inner(cfg, &inner, &mut server, Some(engine), None);
    drop(server);
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    String::from_utf8(response).unwrap()
}

fn post_embeddings(cfg: &ServerConfig, body: &str, engine: &mut dyn DecodeIo) -> String {
    drive(
        cfg,
        &format!(
            "POST /v1/embeddings HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
        engine,
    )
}

/// The gemma_vq embedding family as the bin reports it at boot.
fn embedding_cfg() -> ServerConfig {
    let mut cfg = ServerConfig::test_cfg();
    cfg.model_id = "EmbeddingGemma-2-914f7f8".into();
    cfg.backend = "cuda".into();
    cfg.family = "gemma_vq".into();
    cfg.embedding = true;
    cfg.embed_caps = ds4_core::embed_caps(ds4_core::ModelFamily::GemmaVq);
    // the artifact's tokenizer flags, as the tests' ids assume (BOS 2, EOS 1)
    cfg.add_bos = true;
    cfg.add_eos = true;
    cfg
}

/// The embedding arrays of a response body, as the printed floats.
fn embeddings(response: &str) -> Vec<Vec<String>> {
    let body = response.split("\r\n\r\n").nth(1).unwrap_or("");
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "the route did not answer 200: {response}"
    );
    body.split("\"embedding\":[")
        .skip(1)
        .map(|rest| {
            let list = rest.split(']').next().unwrap();
            list.split(',').map(|v| v.to_string()).collect::<Vec<_>>()
        })
        .collect()
}

/// The ids the route's own construction produces for `text`: BOS, the task
/// prefix, the text, EOS -- read through the same prompt map the route uses
/// (`ds4_core::st_prompt`), so the span arithmetic below is the route's, not a
/// copy of it.
fn expected_ids(text: &str, add_bos: bool, add_eos: bool) -> Vec<i32> {
    let prefix = ds4_core::st_prompt("Retrieval-query").expect("the checkpoint's default task");
    let mut ids = Vec::new();
    if add_bos {
        ids.push(2);
    }
    ids.extend(format!("{prefix}{text}").bytes().map(i32::from));
    if add_eos {
        ids.push(1);
    }
    ids
}

#[test]
fn three_text_items_are_one_tower_call() {
    let cfg = embedding_cfg();
    let calls = Arc::new(Mutex::new(Calls::default()));
    let mut engine = Recorder {
        calls: Arc::clone(&calls),
    };
    let response = post_embeddings(&cfg, r#"{"input":["one","three","seven"]}"#, &mut engine);
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.batches.len(),
        1,
        "the three text items must be ONE tower call, got {}",
        calls.batches.len()
    );
    assert!(calls.images.is_empty(), "no item carried an image");
    // The one call carries the three items' ids laid end to end, in request
    // order, with the spans between them -- nothing else and nothing missing.
    let mut expected: Vec<i32> = Vec::new();
    let mut off: Vec<u32> = vec![0];
    for text in ["one", "three", "seven"] {
        let ids = expected_ids(text, cfg.add_bos, cfg.add_eos);
        expected.extend_from_slice(&ids);
        off.push(expected.len() as u32);
    }
    assert_eq!(calls.batches[0].1, off);
    assert_eq!(calls.batches[0].0, expected);
    // and the response is one embedding per item, in the request's order: the
    // fake returns 1.0 for item 0, 2.0 for item 1, 3.0 for item 2.
    let got = embeddings(&response);
    assert_eq!(got.len(), 3);
    assert_eq!(got[0][0], "1.0000000");
    assert_eq!(got[1][0], "2.0000000");
    assert_eq!(got[2][0], "3.0000000");
    assert!(response.contains(&format!("\"prompt_tokens\":{}", expected.len())));
}

#[test]
fn a_media_item_keeps_its_own_call_and_the_text_items_share_one() {
    let mut cfg = embedding_cfg();
    cfg.vision = true;
    let calls = Arc::new(Mutex::new(Calls::default()));
    let mut engine = Recorder {
        calls: Arc::clone(&calls),
    };
    // index 0 is text, index 1 carries the image, index 2 is text: the batch must
    // carry items 0 and 2 (in that order) and the image its own call -- and the
    // response must still be index-ordered.
    let response = post_embeddings(
        &cfg,
        r#"{"input":["ab",{"content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}]},"cd"]}"#,
        &mut engine,
    );
    let calls = calls.lock().unwrap();
    assert_eq!(calls.batches.len(), 1, "the two text items share one call");
    // the batch carries items 0 and 2 only, in that order
    let a = expected_ids("ab", cfg.add_bos, cfg.add_eos);
    let b = expected_ids("cd", cfg.add_bos, cfg.add_eos);
    assert_eq!(calls.batches[0].1, vec![0, a.len() as u32, (a.len() + b.len()) as u32]);
    let mut expected = a.clone();
    expected.extend_from_slice(&b);
    assert_eq!(calls.batches[0].0, expected);
    // the image item: bos, boi, one placeholder per soft row (the fake's probe
    // says 3), eoi, eos -- and NOT in the batch
    assert_eq!(calls.images.len(), 1);
    assert_eq!(calls.images[0], vec![2, 255999, 258880, 258880, 258880, 258882, 1]);
    assert!(!calls.batches[0].0.contains(&258880), "the image is not in the batch");
    let got = embeddings(&response);
    assert_eq!(got.len(), 3);
    // item 0 is the batch's first (1.0), item 1 the image (7.0), item 2 the
    // batch's second (2.0)
    assert_eq!(got[0][0], "1.0000000");
    assert_eq!(got[1][0], "7.0000000");
    assert_eq!(got[2][0], "2.0000000");
}

#[test]
fn a_request_of_only_media_items_makes_no_batch_call() {
    let mut cfg = embedding_cfg();
    cfg.vision = true;
    let calls = Arc::new(Mutex::new(Calls::default()));
    let mut engine = Recorder {
        calls: Arc::clone(&calls),
    };
    let image = r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="}}"#;
    let response = post_embeddings(
        &cfg,
        &format!(r#"{{"input":[{{"content":[{image}]}},{{"content":[{image}]}}]}}"#),
        &mut engine,
    );
    let calls = calls.lock().unwrap();
    assert!(
        calls.batches.is_empty(),
        "a request with no text item must not make a batch call"
    );
    assert_eq!(calls.images.len(), 2, "one call per media item");
    let got = embeddings(&response);
    assert_eq!(got.len(), 2);
    assert_eq!(got[0][0], "7.0000000");
    assert_eq!(got[1][0], "7.0000000");
}

#[test]
fn a_batch_refusal_is_a_five_hundred_that_quotes_the_engine() {
    let cfg = embedding_cfg();
    let mut engine = RefusingBatch;
    let response = post_embeddings(&cfg, r#"{"input":["a","bb"]}"#, &mut engine);
    assert!(
        response.starts_with("HTTP/1.1 500"),
        "an engine fault is a 500, got: {response}"
    );
    assert!(
        response.contains("the engine refused the packed block"),
        "the reason must be quoted: {response}"
    );
}

/// An engine whose batch entry fails, to pin the route's error class: a fault of
/// the engine's own (not a malformed request) is a 500, never a 400.
struct RefusingBatch;

impl DecodeIo for RefusingBatch {
    fn model_id(&self) -> i32 {
        0
    }
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>, GenerateError> {
        Ok(text.bytes().map(i32::from).collect())
    }
    fn tokenize_rendered_chat(&self, text: &[u8]) -> Result<Vec<i32>, GenerateError> {
        Ok(text.iter().map(|b| i32::from(*b)).collect())
    }
    fn token_text(&self, _token: i32) -> Result<Vec<u8>, GenerateError> {
        Ok(Vec::new())
    }
    fn token_is_stop(&self, _token: i32) -> bool {
        false
    }
    fn sync(&mut self, _tokens: &[i32]) -> Result<(), GenerateError> {
        Ok(())
    }
    fn eval(&mut self, _token: i32) -> Result<(), GenerateError> {
        Ok(())
    }
    fn pos(&self) -> i32 {
        0
    }
    fn ctx(&self) -> i32 {
        2048
    }
    fn generation(&self) -> u64 {
        0
    }
    fn invalidate(&mut self) {}
    fn sample(
        &mut self,
        _temperature: f32,
        _top_k: i32,
        _top_p: f32,
        _min_p: f32,
        _rng: &mut u64,
    ) -> i32 {
        -1
    }
    fn shutdown(&mut self) -> Result<(), GenerateError> {
        Ok(())
    }
    fn embeds_text(&self) -> bool {
        true
    }
    fn embed_item_ids(
        &self,
        prefix: &str,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<i32>, GenerateError> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(2);
        }
        ids.extend(format!("{prefix}{text}").bytes().map(i32::from));
        if add_eos {
            ids.push(1);
        }
        Ok(ids)
    }
    fn embed_gemma_vq_batch(
        &self,
        _ids: &[i32],
        _item_off: &[u32],
    ) -> Result<Vec<f32>, GenerateError> {
        Err(GenerateError::Engine(
            "the engine refused the packed block".into(),
        ))
    }
}
