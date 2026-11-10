#!/usr/bin/env python3
"""Render the EmbeddingGemma-2 embedding path's before/after performance record.

Three panels, every number read from a committed raw file rather than copied
from a record document:

  1. the corpus ladder   -- tokens/s against batch size, one line per shipped
     build, from the speed-bench corpus CSVs (143 pages / 80,747 tokens)
  2. the attention kernel -- median ms per tower shape, original vs vectorized
     vs grouped, from the committed bench_eg2vq_attn outputs
  3. the forward's split  -- where the kernel time of one 8-item / 2,451-token
     forward goes, from the committed nsys cuda_gpu_kern_sum reports

Requires matplotlib 3.11 (the same convention as plot-glm53-vq-64k.py and
plot-ling31-vq-64k.py beside this file).
usage: plot-eg2-embed-perf.py [DATA_DIR]   (default: eg2-embedding-2026-10-20)

Writes perf-before-after.png, perf-summary.json and perf-tables.md into
DATA_DIR.  The corpus CSVs are read from <repo-root>/speed-bench (override with
EG2_SPEED_DIR).
"""
import csv
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCHMARKS = Path(__file__).resolve().parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else BENCHMARKS / "eg2-embedding-2026-10-20"
SPEED = Path(os.environ.get("EG2_SPEED_DIR", DATA.parents[2] / "speed-bench"))
DATE = DATA.name[-10:]

# --- the corpus ladder (speed-bench CSVs, in the order the units landed) -------
# Each entry: label, colour, list of CSV basenames (one per run of that build).
LADDER = [
    ("baseline", "#687789", ["eg2_embed_vizzio.csv"]),
    ("attention vectorized", "#087f8c", ["eg2_embed_vizzio_attn-vec4.csv",
                                         "eg2_embed_vizzio_attn-vec4_r2.csv",
                                         "eg2_embed_vizzio_attn-vec4_r3.csv"]),
    ("payload GEMM rows 8", "#7c3aed", ["eg2_embed_vizzio_vq-ab-new-r1.csv",
                                        "eg2_embed_vizzio_vq-ab-new-r2.csv"]),
    ("attention grouped", "#c2410c", ["eg2_embed_vizzio_attn-ab-new-r1.csv",
                                      "eg2_embed_vizzio_attn-ab-new-r2.csv"]),
]
BATCHES = (1, 2, 4, 8)

# --- the attention kernel (committed bench_eg2vq_attn outputs) -----------------
KERNEL = [
    ("original", "#687789", ["bench-attn-before.txt"]),
    ("vectorized", "#087f8c", ["bench-attn-after.txt"]),
    ("grouped", "#c2410c", ["attn-ab-new-r1.txt", "attn-ab-new-r2.txt"]),
]

# --- the forward's split (committed nsys reports) ------------------------------
# 434 eg2vq_vq_gemm_kernel + 48 eg2vq_attn_kernel launches plus 778 small ones
# per forward; each file is one profile of the same request.  Only the pair the
# third unit's record uses is charted: the older committed report
# (nsys-kern-sum-b8-before-vq-r8.txt, 401.2 ms) profiles a build whose score
# plane the same record calls superseded, so it is named in the footnote and not
# drawn as a "before".
NSYS = [
    ("before: payload GEMM rows 8", "nsys-kern-sum-b8-vq-r8.txt"),
    ("after: + attention grouped", "nsys-kern-sum-b8-attn-grouped.txt"),
]
NSYS_COLS = {"gemm": "#7c3aed", "attention": "#c2410c", "other": "#94a3b8"}

NSYS_ROW = re.compile(
    r"^\s+(\d+\.\d+)\s+([\d,]+)\s+(\d+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+([\d,.]+)\s+(.*)$")
KERNEL_ROW = re.compile(
    r"^(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def corpus():
    """tokens/s per batch size for every build, mean and min-max across its runs."""
    builds = []
    for label, colour, files in LADDER:
        rows = []
        for name in files:
            with (SPEED / name).open() as source:
                table = {r["label"]: r for r in csv.DictReader(source)}
            rows.append([float(table[f"srv_batch_{b}"]["tokens_per_s"]) for b in BATCHES])
        columns = list(zip(*rows))
        builds.append({
            "label": label, "colour": colour, "runs": len(files), "files": files,
            "mean": [mean(c) for c in columns],
            "min": [min(c) for c in columns],
            "max": [max(c) for c in columns],
        })
    return builds


def kernels():
    """Median ms per tower shape for every build of the attention kernel."""
    shapes = None
    builds = []
    for label, colour, files in KERNEL:
        runs = []
        for name in files:
            table = {}
            for line in (DATA / name).read_text().splitlines():
                m = KERNEL_ROW.match(line)
                if m:
                    table[m.group(1)] = (int(m.group(2)), int(m.group(3)),
                                         float(m.group(5)), float(m.group(7)))
            runs.append(table)
        order = list(runs[0])
        if shapes is None:
            shapes = order
        assert order == shapes, f"{files} covers different shapes than the first build"
        builds.append({
            "label": label, "colour": colour, "runs": len(files), "files": files,
            "rows": [runs[0][s][0] for s in shapes],
            "keys": [runs[0][s][1] for s in shapes],
            "ms_med": [mean(r[s][2] for r in runs) for s in shapes],
            "gflop_s": [mean(r[s][3] for r in runs) for s in shapes],
        })
    return shapes, builds


def splits():
    """Kernel time per category from each committed nsys report."""
    profiles = []
    for label, name in NSYS:
        total = {"gemm": 0.0, "attention": 0.0, "other": 0.0}
        launches = {"gemm": 0, "attention": 0, "other": 0}
        for line in (DATA / name).read_text().splitlines():
            m = NSYS_ROW.match(line)
            if not m:
                continue
            ns = float(m.group(2).replace(",", ""))
            which = m.group(9)
            key = ("gemm" if "eg2vq_vq_gemm" in which else
                   "attention" if "eg2vq_attn_kernel" in which else "other")
            total[key] += ns / 1e6
            launches[key] += int(m.group(3))
        profiles.append({"label": label, "file": name, "ms": total,
                         "launches": launches,
                         "total_ms": sum(total.values())})
    return profiles


def plot(builds, shapes, kbuilds, profiles, report):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13.5, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.34, wspace=0.34)
    fig.suptitle("EmbeddingGemma-2 · VQ embedding path · before and after", fontsize=18,
                 fontweight="bold", y=0.975)
    fig.text(0.5, 0.935, "One RTX 4070 SUPER (sm_89) · 143 pages / 80,747 tokens of Swedish "
             f"legal text · {DATE}", ha="center", color="#4b5563")

    # 1. the corpus ladder
    ax = fig.add_subplot(grid[0, :])
    for build in builds:
        last = build["mean"][-1]
        if build["runs"] > 1:
            ax.fill_between(BATCHES, build["min"], build["max"], color=build["colour"],
                            alpha=0.17, linewidth=0)
        ax.plot(BATCHES, build["mean"], color=build["colour"], linewidth=2,
                marker="o", markersize=5,
                label=f"{build['label']} · {last:,.0f} tok/s at batch 8")
    ax.set_title("The corpus, end to end: throughput against batch size", loc="left",
                 fontsize=13, pad=10)
    ax.set_ylabel("Tokens / second")
    ax.set_xlabel("Items per request")
    ax.set_xticks(BATCHES)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", framealpha=0.92)
    ax.margins(y=0.22)
    first, last = builds[0]["mean"][-1], builds[-1]["mean"][-1]
    ax.annotate(f"{last / first:.2f}x", xy=(8, (first + last) / 2),
                xytext=(6.55, (first + last) / 2), fontsize=13, fontweight="bold",
                color=builds[-1]["colour"], va="center")

    # 2. the attention kernel, per tower shape
    ax = fig.add_subplot(grid[1, 0])
    positions = range(len(shapes))
    width = 0.26
    for i, build in enumerate(kbuilds):
        ax.bar([p + (i - 1) * width for p in positions], build["ms_med"], width,
               color=build["colour"], label=build["label"])
    ax.set_yscale("log")
    ax.set_title("The attention kernel alone, per tower shape", loc="left", fontsize=13, pad=10)
    ax.set_ylabel("Milliseconds (median, log scale)")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(shapes, rotation=38, ha="right", fontsize=9)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=10)
    for p in positions:
        first_ms = kbuilds[0]["ms_med"][p]
        last_ms = kbuilds[-1]["ms_med"][p]
        ax.annotate(f"{first_ms / last_ms:.1f}x", xy=(p + width, last_ms), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9,
                    color=builds[-1]["colour"])

    # 3. where one forward's kernel time goes
    ax = fig.add_subplot(grid[1, 1])
    left = [0.0] * len(profiles)
    for key, colour in NSYS_COLS.items():
        values = [p["ms"][key] for p in profiles]
        ax.barh(range(len(profiles)), values, left=left, color=colour, height=0.55,
                label=f"{key} ({sum(p['launches'][key] for p in profiles[:1])} launches)")
        for i, (value, start) in enumerate(zip(values, left)):
            if value < 12:
                continue
            ax.text(start + value / 2, i, f"{value:.1f}", ha="center", va="center",
                    fontsize=10, color="white")
        left = [a + b for a, b in zip(left, values)]
    for i, profile in enumerate(profiles):
        ax.text(profile["total_ms"] + 3, i, f"{profile['total_ms']:.1f} ms",
                va="center", fontsize=10, color="#111827")
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels(["before", "after"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, max(p["total_ms"] for p in profiles) * 1.18)
    ax.set_title("One 8-item / 2,451-token forward: the kernel split\n"
                 "before = payload GEMM rows 8, after = + attention grouped",
                 loc="left", fontsize=12, pad=10)
    ax.set_xlabel("Milliseconds of kernel time in the profile")
    ax.grid(alpha=0.2, axis="x")
    ax.legend(loc="lower right", framealpha=0.92, fontsize=9)

    fig.text(0.5, 0.055,
             "Corpus: speed-bench/eg2_embed_vizzio*.csv (shape: one fresh ds4-server per sweep).  "
             "Attention: docs/benchmarks/eg2-embedding-2026-10-20/bench-attn-{before,after}.txt and "
             "attn-ab-new-r{1,2}.txt.\nSplit: the committed nsys cuda_gpu_kern_sum reports of one "
             "8-item / 2,451-token request; the older report (401.2 ms) profiles a build whose score "
             "plane the record\ncalls superseded, so it is not drawn as a before.  Curves and bars are the mean of "
             "each build's runs (bands: their observed min-max, not a confidence interval); the\n"
             "corpus runs span windows, so only a build's own interleaved comparison is evidence on its "
             "own.  No performance counter was read (ncu is unavailable here).",
             ha="center", va="center", fontsize=8.5, color="#4b5563", linespacing=1.6)
    fig.subplots_adjust(top=0.90, bottom=0.175, left=0.07, right=0.975)
    fig.savefig(DATA / "perf-before-after.png", dpi=180)
    plt.close(fig)


def tables(builds, shapes, kbuilds, profiles):
    lines = ["# EmbeddingGemma-2 embedding path: before and after, generated", "",
             f"Generated from the committed raw files by `plot-eg2-embed-perf.py` ({DATE}).", "",
             "## The corpus, end to end (143 pages / 80,747 tokens, tokens/s)", "",
             "| build | runs | batch 1 | batch 2 | batch 4 | batch 8 |",
             "|---|---:|---:|---:|---:|---:|"]
    for build in builds:
        cells = " | ".join(f"{v:,.0f}" for v in build["mean"])
        lines.append(f"| {build['label']} | {build['runs']} | {cells} |")
    lines += ["", "Speedup against the baseline, batch 8: "
              f"{builds[-1]['mean'][-1] / builds[0]['mean'][-1]:.2f}x "
              f"({builds[0]['mean'][-1]:,.0f} to {builds[-1]['mean'][-1]:,.0f} tokens/s).", "",
              "## The attention kernel alone (median ms per shape)", "",
              "| shape | rows | keys | original | vectorized | grouped | original/grouped |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for i, shape in enumerate(shapes):
        lines.append(f"| `{shape}` | {kbuilds[0]['rows'][i]:,} | {kbuilds[0]['keys'][i]:,} | "
                     + " | ".join(f"{k['ms_med'][i]:.4f}" for k in kbuilds)
                     + f" | {kbuilds[0]['ms_med'][i] / kbuilds[-1]['ms_med'][i]:.2f}x |")
    lines += ["", "## One forward's kernel split (nsys, 8 items / 2,451 tokens)", "",
              "| build | payload GEMM | attention | small ops | total |", "|---|---:|---:|---:|---:|"]
    for profile in profiles:
        lines.append(f"| {profile['label']} | {profile['ms']['gemm']:.1f} ms | "
                     f"{profile['ms']['attention']:.1f} ms | {profile['ms']['other']:.1f} ms | "
                     f"{profile['total_ms']:.1f} ms |")
    lines += ["", "The two profiles are not window-comparable: the payload GEMM row moves "
              "(139.2 to 126.9 ms) with its code unchanged between them, and the third unit's "
              "record names an older committed report (401.2 ms, attention 159.7 ms) whose score "
              "plane that record calls superseded. The split, not the delta, is what these "
              "reports are used for.", ""]
    (DATA / "perf-tables.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    builds = corpus()
    shapes, kbuilds = kernels()
    profiles = splits()
    report = {"date": DATE, "corpus": builds, "attention_kernel": {"shapes": shapes, "builds": kbuilds},
              "forward_split": profiles}
    (DATA / "perf-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(builds, shapes, kbuilds, profiles, report)
    tables(builds, shapes, kbuilds, profiles)
    for build in builds:
        print(f"corpus  {build['label']:24s} batch 8 {build['mean'][-1]:8,.0f} tok/s "
              f"({build['runs']} run(s))")
    for build in kbuilds:
        print(f"kernel  {build['label']:24s} {build['ms_med'][2]:.4f} ms page565_s")
    for profile in profiles:
        print(f"split   {profile['label']:24s} {profile['total_ms']:.1f} ms total")
