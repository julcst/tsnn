#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""Print a side-by-side comparison of benchmark.py (TSNN) and benchmark_tcnn.py
(tiny-cuda-nn) results as per-iteration GPU device time: train/optimizer from
the main training loop, plus a separate dedicated inference-only pass run
after training completes -- see the timing comments in each benchmark's
run() / run_inference_benchmark() for exactly what each one brackets.

Pass --tcnn-jit with a benchmark_tcnn.py --jit result to add tcnn's
jit_fusion variant as an extra column, placed right after TSNN (JIT is the
fairer tcnn comparison; plain tcnn comes last)."""

import argparse
import json


def load(path):
    return json.loads(open(path).read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsnn", default="bench_tsnn.json")
    parser.add_argument("--tcnn", default="bench_tcnn.json")
    parser.add_argument("--tcnn-jit", default=None, help="Optional benchmark_tcnn.py --jit result, added as an extra column")
    parser.add_argument("--markdown", action="store_true", help="Emit a GitHub-flavored markdown table instead")
    args = parser.parse_args()

    a = load(args.tsnn)
    b = load(args.tcnn)
    ai, bi = a["inference"], b["inference"]

    # Column order: TSNN, tcnn (JIT) if present, tcnn (plain).
    columns = [("TSNN (Slang)", a, ai)]
    if args.tcnn_jit:
        c = load(args.tcnn_jit)
        columns.append(("tiny-cuda-nn (JIT)", c, c["inference"]))
    columns.append(("tiny-cuda-nn", b, bi))

    steps = a["config"]["steps"]
    kernels = [
        ("Training (fwd+bwd)", "train_gpu_time_s", steps),
        ("Optimizer step", "optimize_gpu_time_s", steps),
    ]

    rows = []
    for name, key, count in kernels:
        rows.append((name, [r[key] / count for _, r, _ in columns]))
    rows.append(("Inference", [i["gpu_time_s"] / i["iters"] for _, _, i in columns]))

    def speedups(times):
        # Relative change of every other column vs. TSNN (index 0).
        return [times[i] / times[0] for i in range(1, len(times))]

    if args.markdown:
        speedup_headers = " | ".join(f"vs TSNN ({name})" for name, _, _ in columns[1:])
        print("| Kernel | " + " | ".join(name for name, _, _ in columns) + f" | {speedup_headers} |")
        print("|" + "---|" * (len(columns) * 2))
        for name, times in rows:
            cells = " | ".join(f"{t * 1e6:.1f} us" for t in times)
            speedup_cells = " | ".join(f"{s:.2f}x" for s in speedups(times))
            print(f"| {name} | {cells} | {speedup_cells} |")
        print(f"\nPer-iteration GPU time, averaged over {steps:,} training steps / {ai['iters']} inference passes, "
              f"batch {a['config']['batch_size']:,}, {a['gpu']}.")
        psnrs = ", ".join(f"{name} {r['final_psnr']:.2f} dB" for name, r, _ in columns)
        print(f"Final PSNR: {psnrs}.")
        return

    label_width = 20
    col_width = max(18, max(len(n) for n, _, _ in columns) + 2)
    widths = [label_width] + [col_width] * len(columns)
    print("".join(f"{h:>{w}}" for h, w in zip([""] + [n for n, _, _ in columns], widths)))
    print(f"GPU: {a['gpu']}")
    print(f"{'Steps':{label_width}}" + "".join(f"{r['config']['steps']:>{col_width},}" for _, r, _ in columns))
    print(f"{'Batch size':{label_width}}" + "".join(f"{r['config']['batch_size']:>{col_width},}" for _, r, _ in columns))
    print(f"{'Activation':{label_width}}" + "".join(f"{r['config']['activation']:>{col_width}}" for _, r, _ in columns))
    print("-" * sum(widths))
    print(f"Per-iteration GPU time ({steps:,} training steps / {ai['iters']} inference passes):\n")
    for name, times in rows:
        cells = "".join(f"{t * 1e6:>{col_width - 2}.1f}us" for t in times)
        print(f"{name:{label_width}}{cells}")
    print()
    print("Relative change vs. TSNN (>1x = TSNN faster):\n")
    for name, times in rows:
        cells = "".join(f"{s:>{col_width - 1}.2f}x" for s in speedups(times))
        print(f"{name:{label_width}}{'':{col_width}}{cells}")
    print()
    print(f"{'Final PSNR (dB)':{label_width}}" + "".join(f"{r['final_psnr']:>{col_width}.2f}" for _, r, _ in columns))


if __name__ == "__main__":
    main()
