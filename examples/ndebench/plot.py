#!/usr/bin/env -S uv run --script
"""Plot artifacts written by benchmark.py; this script never opens a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def find_output_directory(parser: argparse.ArgumentParser) -> Path:
    """Newest results.json under HERE, else HERE/output (benchmark.py's default)."""
    candidates = [
        p
        for p in HERE.glob("**/results.json")
        if not any(part.startswith(".") or part == "__pycache__" for part in p.parts)
    ]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime).parent
    if (HERE / "output" / "results.json").exists():
        return HERE / "output"
    parser.error(
        f"no results.json found under {HERE} (searched '**/results.json' and 'output/'); "
        "run benchmark.py first or pass an explicit output directory"
    )
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    output = args.output_directory or find_output_directory(parser)
    report = json.loads((output / "results.json").read_text())
    reference = np.load(output / "reference_logpdf.npy")
    entropy = -float(
        np.sum(
            np.load(output / "reference_masses.npy")[np.isfinite(reference)]
            * reference[np.isfinite(reference)],
            dtype=np.float64,
        )
    )
    architectures = report["architectures"]

    # Plotted quantity is KL(target || model) = NLL - reference_entropy, not
    # raw NLL. Two reasons: (1) it's the natural non-negative, converges-to-0
    # quantity for a log-y axis -- raw NLL routinely goes negative here
    # (a continuous density's cross-entropy can be < 0), which a log axis
    # can't display at all; (2) on a log axis, NSF-Quadratic's real
    # order-of-magnitude training-instability spike (see FINDING.md) is
    # naturally compressed alongside everything else instead of needing the
    # percentile-based y-clipping/off-scale-callout hack a linear axis
    # required. A floor epsilon guards against a near-zero or (floating-point
    # noise) slightly negative KL hitting log(0)/log(negative).
    kl_floor = 1e-4
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name, result in architectures.items():
        rows = result["checkpoints"]
        # Skip step 0 (pre-training initialization): step/gpu_seconds are
        # both 0 there, which log-x can't place anyway, and it carries no
        # training-dynamics information.
        steps = [row["step"] for row in rows[1:]]
        gpu_seconds = [row["cumulative_training_gpu_seconds"] for row in rows[1:]]
        kl = [max(row["nll"] - entropy, kl_floor) for row in rows[1:]]
        axes[0].plot(steps, kl, label=name)
        axes[1].plot(gpu_seconds, kl, label=name)

    for ax, xlabel in zip(
        axes,
        (
            "optimizer step",
            # benchmark.py's --seconds-per-method budgets this exact
            # quantity (GPU-timestamp-measured train-loop kernel time, not
            # wall-clock -- see its own comment for why: wall-clock also
            # counts Python/driver dispatch overhead and CPU<->GPU sync
            # waits, disproportionate at this benchmark's small per-step
            # batch sizes and not representative of a production loop's
            # actual GPU cost), so under the default equal-time run every
            # architecture's line in this panel should reach roughly the
            # same right-hand edge.
            "cumulative training GPU seconds",
        ),
    ):
        ax.set(xlabel=xlabel, ylabel="KL(target || model)", xscale="log", yscale="log")
        ax.legend(fontsize=8)
    fig.savefig(output / "nll.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    reference_pdf = np.exp(reference)
    models = {
        name: np.exp(np.load(output / f"{name}_final_logpdf.npy")) for name in architectures
    }
    panels = [("reference", reference_pdf), *models.items()]

    # A shared vmax taken from the raw max lets a single architecture with a
    # tiny fraction of pathologically over-peaked pixels (e.g. a training
    # instability that briefly drove a spline's local density into the
    # hundreds/thousands -- see FINDING.md) blow out the color scale for
    # EVERY panel, reference included, making the whole plot look blank. A
    # high percentile (per panel, then maxed across panels) is the standard
    # outlier-robust choice: it still shows a genuinely sharper-but-healthy
    # model's extra detail, while a handful of outlier pixels just saturate
    # to the top color instead of crushing everything else to near-black.
    vmax_percentile = 99.9
    vmax = max(float(np.nanpercentile(values, vmax_percentile)) for _, values in panels)
    vmax = max(vmax, 1e-6)  # guard against an all-zero panel collapsing the scale to 0

    # Extra vertical room (vs. the NLL figure above) for the per-panel metric
    # caption below each image; constrained_layout is dropped in favor of an
    # explicit bottom margin since captions are placed via ax.transAxes at a
    # negative y, which constrained_layout doesn't reliably budget for.
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 5.6))
    axes = np.atleast_1d(axes)
    image = None
    for axis, (name, values) in zip(axes, panels):
        image = axis.imshow(values, origin="upper", vmin=0.0, vmax=vmax, cmap="magma")
        clipped = float(np.mean(values > vmax)) * 100.0
        title = f"{name} PDF"
        if clipped > 0.01:
            title += f"\n({clipped:.2f}% of pixels saturated)"
        axis.set_title(title)
        axis.set_axis_off()

        if name == "reference":
            continue
        result = architectures[name]
        final = result.get("final_metrics", {})
        timing = result.get("timing", {})
        kl = final.get("kl_divergence")
        variance = final.get("sample_importance_ratio_variance")
        degenerate = final.get("sample_importance_ratio_degenerate_fraction")
        train_ms = timing.get("mean_training_ms_per_update")
        eval_ms = timing.get("pdf_evaluation", {}).get("mean_ms")
        sample_ms = timing.get("sampling", {}).get("mean_ms")
        variance_line = f"sample variance: {variance:.3g}" if variance is not None else "sample variance: n/a"
        if degenerate is not None and degenerate > 0.001:
            variance_line += f" ({degenerate * 100:.2f}% degenerate)"
        caption = "\n".join(
            [
                f"KL divergence: {kl:.4f}" if kl is not None else "KL divergence: n/a",
                variance_line,
                f"train time: {train_ms:.4f} ms" if train_ms is not None else "train time: n/a",
                f"eval time: {eval_ms:.4f} ms" if eval_ms is not None else "eval time: n/a",
                f"sample time: {sample_ms:.4f} ms" if sample_ms is not None else "sample time: n/a",
            ]
        )
        axis.text(
            0.5,
            -0.04,
            caption,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            family="monospace",
        )
    fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.27, wspace=0.15)
    fig.colorbar(image, ax=list(axes), label=f"density (clipped at p{vmax_percentile})")
    fig.savefig(output / "pdf_panels.png", dpi=180)
    plt.close(fig)
    print(f"wrote {output / 'nll.png'} and {output / 'pdf_panels.png'}")


if __name__ == "__main__":
    main()
