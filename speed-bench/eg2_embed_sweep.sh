#!/bin/sh
# eg2_embed_sweep.sh -- the EmbeddingGemma-2 embedding path in the speed-bench
# CSV standard.
#
# WHY A SEPARATE SWEEP (checked first, rule 0.2): ds41_spark_sweep.sh measures
# text frontiers against one generation server, ds41_vision_sweep.sh the ds41
# vision tower.  Neither touches the embedding path, and nothing else does
# either -- verified, not assumed: `ds4-bench --help` has no embed verb,
# `crates/ds4-perf` has no embedding workload (its only "embed" hits are
# unrelated strings in serving/review.rs:73 and csv.rs:4), and GET /v1/stats
# counts only the openai_chat/completion/responses surfaces -- `routes` has no
# embed row, so the server neither counts nor times an embedding request.  That
# is why this sweep exists and why its server rows are wall clock.
#
# ONE ds4 PROCESS PER MACHINE (measured, not assumed): the binary refuses a
# second instance -- `ds4 -m ... --embed ...` while a server runs prints
# "another ds4 process is already running (pid N); refusing to start" and exits
# 2.  The CLI rows therefore run FIRST, before this script boots anything, and
# in attach mode they are recorded as skipped rather than faked.  This is also
# why the CLI's cost is a real cost centre: one process, one model mapping and
# one item per call, with no server to amortize them.
#
# WHAT IS MEASURED, and by whom.  Text enters through POST /v1/embeddings
# (crates/ds4-server/src/serve.rs:1408) with task Retrieval-document
# ("title: none | text: ", crates/ds4-core/src/lib.rs:2921); the query side uses
# Retrieval-query.  On the CLI rows the per-item tower time is the port's OWN
# number: the `secs` field in the JSON the CLI prints (family/task/prompt/
# n_tokens/add_bos/add_eos/dims/secs/embedding).  The server route carries no
# elapsed field in its response and no counter in /v1/stats, so every server row
# is wall clock around curl, labelled `wall` in the csv's `clock` column -- never
# presented as the tower's own time.
#
# Protocol (speed-bench README): one fresh server process per boot run; the exact
# artifact, backend and workload recorded; the model digest and the server's
# /health caps stored beside the csv; one warmup request before any timed row;
# every sample retained.
#
# usage:
#   boot mode (starts its own server; never touches another session's):
#     eg2_embed_sweep.sh <gguf> <pages_dir> [port]
#   attach mode (measures a server already listening; starts nothing, skips the
#   cli rows by name):
#     EG2_EMBED_URL=http://127.0.0.1:8098 eg2_embed_sweep.sh <gguf> <pages_dir>
#
# env: EG2_EMBED_CSV (default speed-bench/eg2_embed_<host>.csv),
#      EG2_EMBED_REPEATS (single-item repeats, default 20),
#      EG2_EMBED_BATCHES (default "1 2 4 8 14"; the axis is ITEMS PER REQUEST,
#        bounded by the server's own 8192-token request cap -- a page of this
#        corpus is ~565 tokens, so ~14 pages is the ceiling, and a larger value
#        is refused by name rather than silently split.  A row that could not
#        complete the corpus is marked `partial` and is NOT comparable to a full
#        row),
#      EG2_EMBED_CLI (cli binary, default ./ds4),
#      EG2_EMBED_BACKEND (cuda|cpu, default cuda)
set -u

HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
ROOT=$(cd "$HERE/.." && pwd) || exit 1
GGUF=${1:?usage: eg2_embed_sweep.sh <gguf> <pages_dir> [port]}
PAGES=${2:?usage: eg2_embed_sweep.sh <gguf> <pages_dir> [port]}
PORT=${3:-8098}
CLI="${EG2_EMBED_CLI:-$ROOT/ds4}"
SERVER="${EG2_EMBED_SERVER:-$ROOT/ds4-server}"
BACKEND="${EG2_EMBED_BACKEND:-cuda}"
REPEATS="${EG2_EMBED_REPEATS:-20}"
BATCHES="${EG2_EMBED_BATCHES:-1 2 4 8 14}"
HOSTNAME_SHORT=$(hostname -s)
CSV="${EG2_EMBED_CSV:-$HERE/eg2_embed_${HOSTNAME_SHORT}.csv}"
ATTACH="${EG2_EMBED_URL:-}"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/eg2-embed-sweep.XXXXXX") || exit 1
SRV_PID=""
SINGLE_TOKENS=0

cleanup() {
    [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

if [ -n "$ATTACH" ]; then URL="$ATTACH"; else URL="http://127.0.0.1:$PORT"; fi

# ---------------------------------------------------------------- artifacts
echo "== artifacts =="
echo "gguf   $GGUF"
echo "sha256 $(sha256sum "$GGUF" | cut -d' ' -f1)"
echo "pages  $PAGES ($(ls "$PAGES"/p*.txt | wc -l) files)"

printf 'label,backend,clock,items,tokens,wall_s,tokens_per_s,note\n' > "$CSV"
echo "csv $CSV"

ITEM="$WORK/item.txt"
head -c 400 "$PAGES/p109.txt" > "$ITEM"

# ---------------------------------------------------- 1. CLI cold start FIRST
# Before anything boots: one process, the model mapping, one item.  The port's
# own `secs` is the tower time; the wall is what a caller pays.
if [ -n "$ATTACH" ]; then
    printf 'cli_cold,%s,wall,0,0,0,0,SKIPPED: attach mode and one ds4 process per machine\n' \
        "$BACKEND" >> "$CSV"
    echo "== cli cold start: SKIPPED (a server is already up; ds4 allows one process) =="
else
    echo "== cli cold start (process + load + one item) =="
    t0=$(date +%s.%N)
    if "$CLI" -m "$GGUF" --backend "$BACKEND" --embed-file "$ITEM" \
            --task Retrieval-query > "$WORK/cli.json" 2>"$WORK/cli.err"; then
        t1=$(date +%s.%N)
        cli_secs=$(jq -r '.secs' "$WORK/cli.json")
        cli_toks=$(jq -r '.n_tokens' "$WORK/cli.json")
        cli_wall=$(echo "$t1 - $t0" | bc)
        printf 'cli_cold,%s,wall,1,%s,%s,%s,port_secs=%s\n' \
            "$BACKEND" "$cli_toks" "$cli_wall" \
            "$(echo "$cli_toks / $cli_wall" | bc -l | cut -c1-8)" "$cli_secs" >> "$CSV"
        echo "  wall ${cli_wall}s, tower secs ${cli_secs}s, ${cli_toks} tokens"
    else
        rc=$?
        printf 'cli_cold,%s,wall,0,0,0,0,FAILED rc=%s: %s\n' \
            "$BACKEND" "$rc" "$(head -1 "$WORK/cli.err" | tr -d ',' | cut -c1-60)" >> "$CSV"
        echo "  cli failed: $(head -1 "$WORK/cli.err")"
    fi
fi

# ------------------------------------------------------------------- helper
# one request: input json array file -> prints "<wall_s> <tokens>" on stdout,
# refuses on stderr with the server's own message.
post() {
    body=$1
    t0=$(date +%s.%N)
    curl -s --max-time 600 -X POST "$URL/v1/embeddings" \
        -H 'Content-Type: application/json' --data-binary "@$body" > "$WORK/resp.json"
    t1=$(date +%s.%N)
    if jq -e '.error' "$WORK/resp.json" >/dev/null 2>&1; then
        echo "REFUSED: $(jq -r '.error.message' "$WORK/resp.json")" >&2
        return 1
    fi
    printf '%s %s\n' "$(echo "$t1 - $t0" | bc)" "$(jq '.usage.total_tokens' "$WORK/resp.json")"
}

# ------------------------------------------------------------- 2. boot/attach
if [ -z "$ATTACH" ]; then
    echo "== booting one fresh server (backend $BACKEND, port $PORT) =="
    "$SERVER" --host 127.0.0.1 --port "$PORT" -m "$GGUF" \
        --backend "$BACKEND" --model-id "eg2-embed-sweep" > "$WORK/server.log" 2>&1 &
    SRV_PID=$!
    i=0
    while [ $i -lt 120 ]; do
        if curl -s --max-time 2 "$URL/health" > "$WORK/health.json" 2>/dev/null &&
           [ -s "$WORK/health.json" ]; then break; fi
        i=$((i + 1)); sleep 1
    done
    [ -s "$WORK/health.json" ] || { echo "server never became healthy" >&2; exit 1; }
else
    curl -s --max-time 5 "$URL/health" > "$WORK/health.json"
fi
cp "$WORK/health.json" "$HERE/$(basename "$CSV" .csv)_health.json" 2>/dev/null || true
echo "health: $(cat "$WORK/health.json")"

# --------------------------------------------- 3. warmup + single-item latency
echo "== warmup + single-item latency ($REPEATS repeats) =="
jq -Rs '{model:"eg2-embed-sweep", task:"Retrieval-query", input:[.]}' "$ITEM" > "$WORK/q.json"
warm=$(post "$WORK/q.json") || exit 1
SINGLE_TOKENS=$(echo "$warm" | cut -d' ' -f2)
echo "  warmup: $(echo "$warm" | cut -d' ' -f1)s, $SINGLE_TOKENS tokens"
: > "$WORK/singles"
i=0
while [ $i -lt "$REPEATS" ]; do
    post "$WORK/q.json" | awk '{print $1}' >> "$WORK/singles"
    i=$((i + 1))
done
sort -g "$WORK/singles" > "$WORK/singles.sorted"
n=$(wc -l < "$WORK/singles.sorted")
p50=$(sed -n "$(( (n + 1) / 2 ))p" "$WORK/singles.sorted")
p95=$(sed -n "$(( (n * 95 + 99) / 100 ))p" "$WORK/singles.sorted")
smin=$(head -1 "$WORK/singles.sorted"); smax=$(tail -1 "$WORK/singles.sorted")
mean=$(awk '{s+=$1} END{printf "%.6f", s/NR}' "$WORK/singles")
printf 'srv_single_p50,%s,wall,1,%s,%s,%s,%s repeats; min=%s max=%s mean=%s\n' \
    "$BACKEND" "$SINGLE_TOKENS" "$p50" \
    "$(echo "$SINGLE_TOKENS / $p50" | bc -l | cut -c1-8)" \
    "$REPEATS" "$smin" "$smax" "$mean" >> "$CSV"
printf 'srv_single_p95,%s,wall,1,%s,%s,%s,p95\n' \
    "$BACKEND" "$SINGLE_TOKENS" "$p95" \
    "$(echo "$SINGLE_TOKENS / $p95" | bc -l | cut -c1-8)" >> "$CSV"
echo "  p50 ${p50}s  p95 ${p95}s  (mean ${mean}s, min ${smin}, max ${smax})"

# --------------------------------------------------- 4. batch throughput curve
echo "== batch throughput (corpus = all pages in $PAGES) =="
echo "[" > "$WORK/corpus.json"
first=1
for f in "$PAGES"/p*.txt; do
    if [ "$first" -eq 0 ]; then printf ',\n' >> "$WORK/corpus.json"; fi
    first=0
    jq -Rs . "$f" >> "$WORK/corpus.json"
done
echo "]" >> "$WORK/corpus.json"
total_items=$(jq 'length' "$WORK/corpus.json")
echo "  corpus: $total_items items"
for b in $BATCHES; do
    acc_items=0; acc_tokens=0; acc_wall=0; partial=0
    start=0
    while [ "$start" -lt "$total_items" ]; do
        end=$((start + b)); [ "$end" -gt "$total_items" ] && end=$total_items
        jq -c --argjson s "$start" --argjson e "$end" \
            '{model:"eg2-embed-sweep", task:"Retrieval-document", input: .[$s:$e]}' \
            "$WORK/corpus.json" > "$WORK/body.json"
        if ! out=$(post "$WORK/body.json"); then partial=1; break; fi
        w=$(echo "$out" | cut -d' ' -f1); t=$(echo "$out" | cut -d' ' -f2)
        acc_wall=$(echo "$acc_wall + $w" | bc); acc_tokens=$((acc_tokens + t))
        acc_items=$((acc_items + (end - start)))
        start=$end
    done
    [ "$acc_tokens" -eq 0 ] && { echo "  batch $b: nothing measured"; continue; }
    rate=$(echo "$acc_tokens / $acc_wall" | bc -l | cut -c1-8)
    note="corpus=$total_items pages"
    [ "$partial" -eq 1 ] && note="PARTIAL corpus=$total_items pages; cap refused the next request"
    printf 'srv_batch_%s,%s,wall,%s,%s,%s,%s,%s\n' \
        "$b" "$BACKEND" "$acc_items" "$acc_tokens" "$acc_wall" "$rate" "$note" >> "$CSV"
    echo "  batch $b: $acc_items items, $acc_tokens tokens, ${acc_wall}s, ${rate} tokens/s"
done

echo
echo "== csv =="
column -s, -t "$CSV"
