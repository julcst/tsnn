#!/usr/bin/env -S uv run --script
"""CPU-only figures from a benchmark run: PNGs for README, PDFs for print."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from PIL import Image

from models import ARCHITECTURES, LABELS

HERE = Path(__file__).resolve().parent
PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#666666",
    "#A6761D",
    "#332288",
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)


def metric_table(report: dict, output: Path) -> dict[str, dict]:
    """Use deterministic texel quadrature for Pearson χ²; keep sample checks separate."""
    masses = np.load(output / "reference_masses.npy")
    metrics = {}
    for name, result in report["architectures"].items():
        logpdf = np.load(output / f"{name}_final_logpdf.npy").astype(np.float64)
        q = np.exp(logpdf) / masses.size
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = np.where(q > 0, (masses - q) ** 2 / q, np.where(masses > 0, np.inf, 0))
        chi2 = float(np.sum(integrand))
        timing = result["timing"]
        metrics[name] = {
            "kl": result["final_metrics"]["kl_divergence"],
            "variance": max(0.0, chi2),
            "sample_mrays": timing["sampling"]["throughput_per_s"] / 1e6,
            "eval_mrays": timing["pdf_evaluation"]["throughput_per_s"] / 1e6,
            "train_ms": timing["mean_training_ms_per_update"],
            "params": result["metadata"]["trainable_parameters"],
            "degenerate": result["final_metrics"]["sample_importance_ratio_degenerate_fraction"],
        }
    return metrics


def pareto_frontier(metrics: dict[str, dict]) -> list[str]:
    """Maximize throughput and minimize variance, excluding non-finite sample-ratio runs."""
    reliable = {name: m for name, m in metrics.items() if m["degenerate"] == 0}
    return [
        name
        for name, m in reliable.items()
        if np.isfinite(m["variance"])
        and not any(
            other != name
            and n["sample_mrays"] >= m["sample_mrays"]
            and n["variance"] <= m["variance"]
            and (n["sample_mrays"] > m["sample_mrays"] or n["variance"] < m["variance"])
            for other, n in reliable.items()
        )
    ]


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_convergence(output: Path, report: dict, names: list[str], colors: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    entropy = report["reference_entropy"]
    for name in names:
        rows = report["architectures"][name]["checkpoints"][1:]
        ax.plot(
            [r["cumulative_training_gpu_seconds"] for r in rows],
            [max(r["nll"] - entropy, 1e-5) for r in rows],
            color=colors[name],
            label=LABELS.get(name, name),
            linewidth=1.7,
        )
    ax.set(xlabel="Training GPU time (s)", ylabel="KL(target ‖ model) ↓", yscale="log")
    ax.grid(alpha=0.18, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
    save_figure(fig, output, "convergence")


def plot_pareto(output: Path, metrics: dict, names: list[str]) -> None:
    """Speed × inverse χ²: both improve toward the top right.

    Color and diagonal guides show χ² / (samples/s × 1 ms), the expected
    variance of the normalized importance-sampling estimator at equal time.
    """
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    speed = np.array([metrics[n]["sample_mrays"] / 1000 for n in names])
    quality = np.array([1 / metrics[n]["variance"] for n in names])
    # Gsamples/s × 1 ms = one million samples per unit of x.
    # Multiplying variance by 1e9 leaves the plotted color value 1000/(x*y).
    variance_nano = 1000 / (speed * quality)
    norm = LogNorm(vmin=variance_nano.min() / 1.1, vmax=variance_nano.max() * 1.1)
    cmap = plt.get_cmap("viridis_r")
    ax.set(xscale="log", yscale="log")
    xlim = (speed.min() / 1.1, speed.max() * 1.1)
    ylim = (quality.min() / 1.2, quality.max() * 1.2)
    ax.set(xlim=xlim, ylim=ylim)

    x = np.geomspace(*xlim, 200)
    for value in (3, 5, 8, 12, 20):
        y = 1000 / (value * x)
        visible = (y > ylim[0]) & (y < ylim[1])
        if np.any(visible):
            ax.plot(x[visible], y[visible], color="#bbbbbb", lw=0.9, ls=":", zorder=0)
            idx = np.flatnonzero(visible)[int(0.7 * (len(np.flatnonzero(visible)) - 1))]
            ax.text(
                x[idx],
                y[idx],
                f"{value:g}",
                fontsize=8,
                color="#777777",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            )

    frontier = sorted(pareto_frontier(metrics), key=lambda n: metrics[n]["sample_mrays"])
    ax.plot(
        [metrics[n]["sample_mrays"] / 1000 for n in frontier],
        [1 / metrics[n]["variance"] for n in frontier],
        color="#555555",
        ls="--",
        lw=1,
        zorder=1,
    )
    for i, name in enumerate(names):
        m = metrics[name]
        point_color = cmap(norm(variance_nano[i]))
        text_color = (
            "black" if np.dot(point_color[:3], [0.2126, 0.7152, 0.0722]) > 0.55 else "white"
        )
        ax.scatter(
            speed[i],
            quality[i],
            s=180,
            color=point_color,
            edgecolor=text_color,
            linewidth=1,
            label=f"{i + 1}  {LABELS.get(name, name)}" + (" *" if m["degenerate"] else ""),
        )
        ax.annotate(
            str(i + 1),
            (speed[i], quality[i]),
            color=text_color,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.set(
        xlabel="Sampling throughput (billion samples/s) →", ylabel="Inverse Pearson χ² (1 / χ²) ↑"
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 4, 5, 6, 8)))
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(alpha=0.12, which="both")
    ax.text(
        0.98,
        0.97,
        "Better speed and fit ↗",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
    scale = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(
        scale,
        ax=ax,
        location="bottom",
        pad=0.08,
        fraction=0.07,
        ticks=[v for v in (3, 5, 8, 12, 20) if norm.vmin <= v <= norm.vmax],
    )
    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    colorbar.ax.xaxis.set_minor_formatter(NullFormatter())
    colorbar.set_label("Expected variance after 1 ms = χ² / (throughput × 1 ms)  [×10⁻⁹] ↓")
    if any(m["degenerate"] for m in metrics.values()):
        fig.text(
            0.02,
            -0.02,
            "* Non-finite sample ratios; excluded from frontier. Colors assume a valid sampler.",
            fontsize=9,
        )
    save_figure(fig, output, "pareto")


def plot_images(output: Path, names: list[str]) -> None:
    """Shared linear density scale; resize only for display, keeping raw arrays intact."""
    reference = np.exp(np.load(output / "reference_logpdf.npy"))
    panels = [("Target", reference)]
    for name in names:
        panels.append(
            (LABELS.get(name, name), np.exp(np.load(output / f"{name}_final_logpdf.npy")))
        )
    cols = 5
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3), layout="constrained")
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (label, density) in zip(axes.flat, panels):
        thumbnail = Image.fromarray(density.astype(np.float32))
        thumbnail.thumbnail((512, 512), Image.Resampling.BOX)
        im = ax.imshow(np.asarray(thumbnail), cmap="magma", vmin=0, vmax=float(reference.max()))
        ax.set_title(label, fontsize=9, pad=6)
    fig.colorbar(im, ax=list(axes.flat), shrink=0.8, label="Probability density (shared scale)")
    save_figure(fig, output, "images")


def format_summary_table(report: dict, output: Path) -> str:
    metrics = metric_table(report, output)
    lines = [
        "| Configuration | Parameters | KL ↓ | χ² ↓ | Train ms/update ↓ | Eval M/s ↑ | Sample M/s ↑ | Non-finite ratios |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, m in metrics.items():
        lines.append(
            f"| {LABELS.get(name, name)} | {m['params']:,} | {m['kl']:.4f} | "
            f"{m['variance']:.4f} | {m['train_ms']:.3f} | "
            f"{m['eval_mrays']:.1f} | {m['sample_mrays']:.1f} | {m['degenerate']:.2%} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path, nargs="?", default=HERE / "output")
    args = parser.parse_args()
    output = args.output_directory
    report = json.loads((output / "results.json").read_text())
    names = [n for n in ARCHITECTURES if n in report["architectures"]]
    names += [n for n in report["architectures"] if n not in names]
    colors = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    metrics = metric_table(report, output)
    plot_convergence(output, report, names, colors)
    plot_pareto(output, metrics, names)
    plot_images(output, names)
    table = format_summary_table(report, output)
    (output / "summary.md").write_text(table + "\n")
    print(table)
    print("Sampling/variance Pareto frontier:", ", ".join(pareto_frontier(metrics)))
    print(f"Wrote PNG/PDF figures and summary.md to {output}")


if __name__ == "__main__":
    main()
