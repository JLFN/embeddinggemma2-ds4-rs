#!/bin/sh
# qa-identity/build.sh — build the old-vs-new bit-identity driver.
#
# WHAT IT DOES: extracts the PRE-change header (parent commit 31285531 of this
# unit) into ./old/cuda/eg2vq_vq.cuh, copies ds41_vq_fmt.h beside it, and links
# two translation units — runner_old.cu (the old kernel, macro-renamed so both
# definitions can coexist) and runner_new.cu (the shipped kernel) — against
# main.cu, which feeds the SAME payload and activation bytes to both and
# compares the raw float bytes of the output.
#
# The driver exists because "the new kernel is bit-identical" is a claim about
# bytes: it must be checked by giving both kernels identical input in one
# process, not by comparing two runs of the same binary.
#
# usage: bash build.sh && ./qa-identity
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)          # -> ds4-rs
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}

mkdir -p "$HERE/old/cuda"
git -C "$ROOT" show 31285531:cuda/eg2vq_vq.cuh > "$HERE/old/cuda/eg2vq_vq.cuh"
git -C "$ROOT" show 31285531:ds41_vq_fmt.h > "$HERE/old/ds41_vq_fmt.h"

"$NVCC" -O3 --use_fast_math -std=c++17 -arch=sm_89 -Xcompiler -march=native \
    -I"$HERE" -I"$ROOT" -rdc=false \
    -o "$HERE/qa-identity" \
    "$HERE/main.cu" "$HERE/runner_old.cu" "$HERE/runner_new.cu"

echo "built $HERE/qa-identity"
