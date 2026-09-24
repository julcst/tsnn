#!/usr/bin/env -S uv run --script
"""Data/viz boundary for the ndebench comparison; never opens a GPU, like plot.py.

Reads results.json + the *_final_logpdf.npy arrays written by benchmark.py and projects
them into lab-neural-importance-sampling/data/ndebench/:

  convergence.csv   method, step, samples_consumed, gpu_seconds, wall_seconds, nll, kl
                    (kl = nll - reference_entropy, i.e. KL(target || model); see plot.py's
                    metric-choice note -- raw, unfloored here, the log-scale floor is a
                    plotting concern applied by the figure script, not a data one)
  finals.csv        method, kl, variance, importance_ratio_mean, degenerate_fraction,
                    train_ms, eval_ms, sample_ms  (from results.json's final_metrics/timing)
  logpdf/reference.npy, logpdf/<method>.npy
                    RAW log-density (float32), box-averaged 4x (3259x3250 -> ~814x812) --
                    no colormap, no vmin/vmax, no exp() baked in. Deliberately kept in the
                    same log domain as benchmark.py's own *_final_logpdf.npy: the figure
                    script owns every visualization choice (colormap, shared vs. per-panel
                    scale, density vs. log-density axis), this script owns only reading the
                    benchmark's numbers. Downsampled because full native-resolution float32
                    is ~42 MB/array (~380 MB for the full set); at this resolution each
                    array is ~2.6 MB (~24 MB total), still exact enough for a print figure.
  meta.json          generated date, falcor commit, methods (both export and caption
                    display order), reference entropy, run configuration.

Usage:
  uv run --script export.py [output_directory] [--report-repo ../../../../lab-neural-importance-sampling]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Export order (results.json dict order) vs. caption display order/label are different
# things: export order doesn't matter for CSV rows, but the paper wants GT, NSF-L, NSF-Q,
# NSF-RQS, DF-N, DF-L, TMM, HDF, HGGrid specifically -- ndebench_data.py's DISPLAY_ORDER
# carries that, keyed off these same result-key names.
DISPLAY_LABELS = {
    "NSFLinear": "NSF-L",
    "NSFQuadratic": "NSF-Q",
    "NSFRQS": "NSF-RQS",
    "DFN": "DF-N",
    "DFL": "DF-L",
    "TMM": "TMM",
    "HDF": "HDF",
    "HGGrid": "HGGrid",
    "HDFG4L3": "HDF-G4L3",
    "HDFG3L4": "HDF-G3L4",
    "HDFG4L3B": "HDF-G4L3-blob",
    "HDFG3L4B": "HDF-G3L4-blob",
    "HDFG4L3S": "HDF-G4L3-blob-s",
    "HGGridG2L3": "HGGrid-G2L3",
    "HGGridG2L3B": "HGGrid-G2L3-blob",
    "HGGridG4L2": "HGGrid-G4L2",
    "HGGridG4L2B": "HGGrid-G4L2-blob",
    "HGGridG4L2S": "HGGrid-G4L2-blob-s",
    "HDFFastest": "HDF-G4L3-H32D2",
    "HDFFast": "HDF-G8L2-H32D2",
    "HGGridFastest": "HGGrid-G8L1-H32D2-K4",
    "HGGridFast": "HGGrid-G4L2-H16D3-K8",
}
DISPLAY_ORDER = [
    "NSFLinear",
    "NSFQuadratic",
    "NSFRQS",
    "DFN",
    "DFL",
    "TMM",
    "HDF",
    "HGGrid",
    "HDFG4L3",
    "HDFG3L4",
    "HDFG4L3B",
    "HDFG3L4B",
    "HDFG4L3S",
    "HGGridG2L3",
    "HGGridG2L3B",
    "HGGridG4L2",
    "HGGridG4L2B",
    "HGGridG4L2S",
    "HDFFastest",
    "HDFFast",
    "HGGridFastest",
    "HGGridFast",
]

LOGPDF_DOWNSAMPLE = 4


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return None


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    os.replace(tmp, path)


def box_average(a: np.ndarray, factor: int) -> np.ndarray:
    h, w = a.shape
    h2, w2 = h // factor, w // factor
    return a[: h2 * factor, : w2 * factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_directory", type=Path, nargs="?", default=HERE / "output",
                     help="benchmark.py OUTDIR holding results.json + *_final_logpdf.npy")
    ap.add_argument("--report-repo", type=Path,
                     default=HERE.parents[4] / "lab-neural-importance-sampling",
                     help="lab-neural-importance-sampling checkout to write data/ndebench/ into")
    args = ap.parse_args()

    outdir = args.output_directory.resolve()
    report_repo = args.report_repo.resolve()
    data_dir = report_repo / "data" / "ndebench"
    logpdf_dir = data_dir / "logpdf"

    report = json.loads((outdir / "results.json").read_text())
    architectures = report["architectures"]
    entropy = float(report["reference_entropy"])
    methods = [m for m in DISPLAY_ORDER if m in architectures]
    missing = set(architectures) - set(methods)
    if missing:
        raise SystemExit(f"results.json has methods not in DISPLAY_ORDER: {sorted(missing)}")

    # ── convergence.csv ─────────────────────────────────────────────────────
    conv_rows = []
    for method in methods:
        for row in architectures[method]["checkpoints"]:
            nll = row["nll"]
            conv_rows.append((
                method, row["step"], row["samples_consumed"],
                row["cumulative_training_gpu_seconds"], row["wall_seconds"],
                nll, nll - entropy,
            ))
    write_csv(
        data_dir / "convergence.csv",
        ["method", "step", "samples_consumed", "gpu_seconds", "wall_seconds", "nll", "kl"],
        conv_rows,
    )

    # ── finals.csv ──────────────────────────────────────────────────────────
    final_rows = []
    for method in methods:
        result = architectures[method]
        fm = result["final_metrics"]
        timing = result["timing"]
        meta = result.get("metadata", {})
        final_rows.append((
            method,
            fm["kl_divergence"],
            fm["sample_importance_ratio_variance"],
            fm["sample_importance_ratio_mean"],
            fm["sample_importance_ratio_degenerate_fraction"],
            timing["mean_training_ms_per_update"],
            timing["pdf_evaluation"]["mean_ms"],
            timing["sampling"]["mean_ms"],
            meta.get("trainable_parameters"),
            meta.get("parameter_elements_fp16"),
        ))
    write_csv(
        data_dir / "finals.csv",
        ["method", "kl", "variance", "importance_ratio_mean", "degenerate_fraction",
         "train_ms", "eval_ms", "sample_ms", "params", "buffer_elements"],
        final_rows,
    )

    # ── logpdf/ ─────────────────────────────────────────────────────────────
    logpdf_dir.mkdir(parents=True, exist_ok=True)
    reference_logpdf = np.load(outdir / "reference_logpdf.npy")
    np.save(logpdf_dir / "reference.npy", box_average(reference_logpdf, LOGPDF_DOWNSAMPLE).astype(np.float32))
    for method in methods:
        logpdf = np.load(outdir / f"{method}_final_logpdf.npy")
        np.save(logpdf_dir / f"{method}.npy", box_average(logpdf, LOGPDF_DOWNSAMPLE).astype(np.float32))

    # ── meta.json ────────────────────────────────────────────────────────────
    meta = {
        "generated": datetime.date.today().isoformat(),
        "falcor_commit": git_commit(HERE),
        "methods": methods,
        "display_labels": {m: DISPLAY_LABELS[m] for m in methods},
        "reference_entropy": entropy,
        "logpdf_downsample": LOGPDF_DOWNSAMPLE,
        "configuration": report.get("configuration"),
        "notes": (
            "Single fixed scene (2D density estimation of examples/einstein.png as a "
            "target pdf, see benchmark.py) -- the varying axis is purely the NIS/NDE "
            "architecture (Mueller-style coupling flows NSF-L/NSF-Q/NSF-RQS, "
            "discretized-flow variants DF-N/DF-L, TMM, and the hash-grid architectures "
            "HDF/HGGrid). kl in both CSVs is KL(target || model) = nll - "
            "reference_entropy; convergence.csv's kl is raw (unfloored) since a plotted "
            "value must clip to a small epsilon for a log-y axis but the exported data "
            "should not silently hide a negative/near-zero value -- see plot.py's "
            "kl_floor comment. finals.csv's kl/variance/degenerate_fraction are the "
            "single end-of-training measurement from results.json's final_metrics; "
            "train_ms/eval_ms/sample_ms are mean per-call timings from a fixed warmed-up "
            "workload (results.json's timing block), not cumulative training time. "
            "logpdf/*.npy are RAW log-density, box-averaged in log space (not "
            "exponentiated, not colormapped, not normalized) -- the figure script decides "
            "vmin/vmax/colormap/density-vs-log-density at plot time, e.g. a shared "
            "vmax = exp(reference logpdf).max() across every panel so a local over-peak in "
            "one architecture's map (a training instability, see FINDING.md) just clips to "
            "the colormap's brightest color instead of needing a per-panel or percentile "
            "scale."
        ),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f">>> exported {len(conv_rows)} convergence rows, {len(final_rows)} finals rows, "
        f"{len(methods) + 1} logpdf arrays -> {data_dir}"
    )


if __name__ == "__main__":
    main()
