//! Phase 8, unit 26: the image half of the embedding route, at the HTTP level.
//!
//! What these tests are for: `/health`'s `embed` object is the contract a caller
//! reads before sending an image (`ds4_core::EmbedCaps::to_json`), and the
//! `/v1/embeddings` parser is where an image in the WRONG shape is refused -- on
//! a text-only family, as a non-data-URI, as a second image in one item.  Both
//! are pure parse/render facts, so they are testable without a model: the tests
//! below drive `handle_client_inner` over a real socket exactly as
//! `serve_503_body_test.rs` does, and the end-to-end path with a real artifact
//! and a real tower is gate 26's job (`tools/eg2_embed_media_gate26.sh`).
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

use crate::serve::{handle_client_inner, ServerConfig, ServerInner};

/// One request through the real handler, one response, as bytes.
fn drive(cfg: &ServerConfig, request: &str) -> String {
    let inner = Mutex::new(ServerInner::from_cfg(cfg));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    write!(client, "{request}").unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap();
    let (mut server, _) = listener.accept().unwrap();
    handle_client_inner(cfg, &inner, &mut server, None, None);
    drop(server);
    let mut response = Vec::new();
    client.read_to_end(&mut response).unwrap();
    String::from_utf8(response).unwrap()
}

fn get_health(cfg: &ServerConfig) -> String {
    drive(cfg, "GET /health HTTP/1.1\r\n\r\n")
}

/// A `POST /v1/embeddings` with `body` as its JSON payload.
fn post_embeddings(cfg: &ServerConfig, body: &str) -> String {
    drive(
        cfg,
        &format!(
            "POST /v1/embeddings HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        ),
    )
}

/// The `gemma_vq` embedding family as the bin reports it at boot.
fn embedding_cfg() -> ServerConfig {
    let mut cfg = ServerConfig::test_cfg();
    cfg.model_id = "EmbeddingGemma-2-914f7f8".into();
    cfg.backend = "cuda".into();
    cfg.family = "gemma_vq".into();
    cfg.embedding = true;
    cfg.embed_caps = ds4_core::embed_caps(ds4_core::ModelFamily::GemmaVq);
    cfg
}

fn one_pixel_png() -> String {
    // A real 1x1 PNG, base64: the shortest payload the decoder's magic-byte
    // check accepts.  Its own pixels are irrelevant here -- these tests stop at
    // the parse, and the tower's placement of it is gate 26's business.
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        .to_string()
}

#[test]
fn health_reports_the_image_half_when_the_family_has_one() {
    let cfg = embedding_cfg();
    let response = get_health(&cfg);
    let expected_image = ds4_core::embed_caps(ds4_core::ModelFamily::GemmaVq)
        .unwrap()
        .image
        .unwrap()
        .to_json()
        .to_string();
    assert!(
        response.contains(&format!("\"image\":{expected_image}")),
        "the health row does not carry the image contract the caps render: {response}"
    );
    // The two text numbers the caps have always carried stay where they were:
    // this unit adds the image half, it does not move the old one.
    assert!(response.contains("\"item_tokens_max\":2048"), "{response}");
    assert!(response.contains("\"batch_rows_max\":8192"), "{response}");
    assert!(response.contains("\"embedding\":true"), "{response}");
    assert!(response.ends_with("\n"), "{response}");
}

#[test]
fn health_reports_no_embed_for_a_text_only_family() {
    // A decode family carries no embedding caps at all (`embed_caps` is None),
    // and that must render as a null rather than as zeros a caller could read as
    // limits.
    let cfg = ServerConfig::test_cfg();
    let response = get_health(&cfg);
    assert!(response.contains("\"embed\":null"), "{response}");
    assert!(response.contains("\"embedding\":false"), "{response}");
}

#[test]
fn an_image_on_a_text_only_route_is_refused_by_name() {
    // Same request as the accepted one below, on a family whose embedding path
    // is text only: the refusal must name the contract, not the encoder.
    let mut cfg = embedding_cfg();
    cfg.embed_caps = None;
    let body = format!(
        "{{\"input\":[{{\"content\":[{{\"type\":\"text\",\"text\":\"a photo: <|image|>\"}},\
         {{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{}\"}}}}]}}]}}",
        one_pixel_png()
    );
    let response = post_embeddings(&cfg, &body);
    assert!(
        response.starts_with("HTTP/1.1 400"),
        "a text-only route must refuse an image with a 400: {response}"
    );
    assert!(
        response.contains("this model's embedding path is text only"),
        "{response}"
    );
}

#[test]
fn an_image_uri_that_is_not_a_data_uri_is_refused_by_name() {
    let cfg = embedding_cfg();
    let body = "{\"input\":[{\"content\":[{\"type\":\"text\",\"text\":\"<|image|>\"},\
                {\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.test/a.png\"}}]}]}";
    let response = post_embeddings(&cfg, body);
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    // The message is the chat route's own decoder's, not a second wording: one
    // image path means one refusal vocabulary.
    assert!(response.contains("image_url must be a base64 data URI"), "{response}");
}

#[test]
fn a_second_image_in_one_item_is_refused_by_name() {
    let cfg = embedding_cfg();
    let png = one_pixel_png();
    let body = format!(
        "{{\"input\":[{{\"content\":[{{\"type\":\"text\",\"text\":\"two: <|image|>\"}},\
         {{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{png}\"}}}},\
         {{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{png}\"}}}}]}}]}}"
    );
    let response = post_embeddings(&cfg, &body);
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert!(response.contains("more than one image"), "{response}");
    assert!(response.contains("embed.image.per_item_max"), "{response}");
}

#[test]
fn an_unknown_content_part_type_is_refused_by_name() {
    let cfg = embedding_cfg();
    let body = "{\"input\":[{\"content\":[{\"type\":\"text\",\"text\":\"hi\"},\
                {\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,AAAA\"}}]}]}";
    let response = post_embeddings(&cfg, body);
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    assert!(response.contains("audio_url"), "{response}");
}

#[test]
fn a_well_formed_media_item_reaches_the_engine() {
    // With no `DecodeIo` supplied the owner answers 503, which is exactly the
    // proof wanted here: the request PARSED (an image item, its URI decoded) and
    // moved past `/v1/embeddings`' own gates to the engine call.  A parse
    // refusal would have been a 400 instead.
    let cfg = embedding_cfg();
    let body = format!(
        "{{\"input\":[{{\"content\":[{{\"type\":\"text\",\"text\":\"a photo: <|image|>\"}},\
         {{\"type\":\"image_url\",\"image_url\":{{\"url\":\"{}\"}}}}]}}]}}",
        one_pixel_png()
    );
    let response = post_embeddings(&cfg, &body);
    assert!(
        response.starts_with("HTTP/1.1 503"),
        "a parsed media item must reach the engine (503 without one): {response}"
    );
}

#[test]
fn the_text_only_shape_is_unchanged() {
    // Regression guard for the old surface: a plain string input still parses
    // and still reaches the engine, and an empty array is still a 400.
    let cfg = embedding_cfg();
    let ok = post_embeddings(&cfg, "{\"input\":\"a query\"}");
    assert!(ok.starts_with("HTTP/1.1 503"), "{ok}");
    let empty = post_embeddings(&cfg, "{\"input\":[]}");
    assert!(empty.starts_with("HTTP/1.1 400"), "{empty}");
    assert!(empty.contains("must not be an empty array"), "{empty}");
}
