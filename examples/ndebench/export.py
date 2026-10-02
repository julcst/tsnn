#!/usr/bin/env -S uv run --script
"""Export raw benchmark CSVs and downsampled log-density arrays to a report repo.

Usage: uv run export.py output --report-repo /path/to/report
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from models import ARCHITECTURES, LABELS as DISPLAY_LABELS

HERE = Path(__file__).resolve().parent

DISPLAY_ORDER = ARCHITECTURES

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
    ap.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        default=HERE / "output",
        help="benchmark.py OUTDIR holding results.json + *_final_logpdf.npy",
    )
    ap.add_argument(
        "--report-repo",
        type=Path,
        default=HERE.parents[4] / "lab-neural-importance-sampling",
        help="lab-neural-importance-sampling checkout to write data/ndebench/ into",
    )
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
            conv_rows.append(
                (
                    method,
                    row["step"],
                    row["samples_consumed"],
                    row["cumulative_training_gpu_seconds"],
                    row["wall_seconds"],
                    nll,
                    nll - entropy,
                )
            )
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
        final_rows.append(
            (
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
            )
        )
    write_csv(
        data_dir / "finals.csv",
        [
            "method",
            "kl",
            "variance",
            "importance_ratio_mean",
            "degenerate_fraction",
            "train_ms",
            "eval_ms",
            "sample_ms",
            "params",
            "buffer_elements",
        ],
        final_rows,
    )

    # ── logpdf/ ─────────────────────────────────────────────────────────────
    logpdf_dir.mkdir(parents=True, exist_ok=True)
    reference_logpdf = np.load(outdir / "reference_logpdf.npy")
    np.save(
        logpdf_dir / "reference.npy",
        box_average(reference_logpdf, LOGPDF_DOWNSAMPLE).astype(np.float32),
    )
    for method in methods:
        logpdf = np.load(outdir / f"{method}_final_logpdf.npy")
        np.save(
            logpdf_dir / f"{method}.npy", box_average(logpdf, LOGPDF_DOWNSAMPLE).astype(np.float32)
        )

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
            "KL(target || model) = NLL - reference entropy. Variance is the model-sample "
            "importance-ratio estimate, with non-finite ratios excluded and their fraction "
            "reported separately. Times are warmed GPU dispatch measurements. Arrays are "
            "box-averaged in log space; figures choose their own color scale."
        ),
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(
        f">>> exported {len(conv_rows)} convergence rows, {len(final_rows)} finals rows, "
        f"{len(methods) + 1} logpdf arrays -> {data_dir}"
    )


if __name__ == "__main__":
    main()
