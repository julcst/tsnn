#!/usr/bin/env -S uv run --script
"""Plot artifacts written by benchmark.py; this script never opens a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def clipped(values: np.ndarray, floor: float) -> np.ndarray:
    """Display-only clipping: the saved arrays are intentionally untouched."""
    return np.maximum(values, floor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--logpdf-floor", type=float, default=-12.0)
    args = parser.parse_args()
    output = args.output_directory
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name, result in architectures.items():
        rows = result["checkpoints"]
        steps = [row["step"] for row in rows]
        nll = [row["nll"] for row in rows]
        seconds = [row["cumulative_training_gpu_seconds"] for row in rows]
        axes[0].plot(steps, nll, marker="o", label=name)
        axes[1].plot(seconds, nll, marker="o", label=name)
    for ax, xlabel in zip(axes, ("optimizer step", "cumulative training GPU seconds")):
        ax.axhline(entropy, color="0.35", linestyle="--", label="reference entropy")
        ax.set(xlabel=xlabel, ylabel="target-distribution NLL")
        ax.legend()
    fig.savefig(output / "nll.png", dpi=180)
    plt.close(fig)

    models = {name: np.load(output / f"{name}_final_logpdf.npy") for name in architectures}
    limits = (
        args.logpdf_floor,
        max(float(np.nanmax(reference)), *(float(np.nanmax(x)) for x in models.values())),
    )
    fig, axes = plt.subplots(
        1, len(models) + 1, figsize=(4 * (len(models) + 1), 4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    panels = [("reference", reference), *models.items()]
    image = None
    for axis, (name, values) in zip(axes, panels):
        image = axis.imshow(
            clipped(values, args.logpdf_floor),
            origin="upper",
            vmin=limits[0],
            vmax=limits[1],
            cmap="magma",
        )
        axis.set_title(f"{name} logPDF")
        axis.set_axis_off()
    fig.colorbar(image, ax=axes, label="log density (display clipped)")
    fig.savefig(output / "logpdf_panels.png", dpi=180)
    plt.close(fig)
    print(f"wrote {output / 'nll.png'} and {output / 'logpdf_panels.png'}")


if __name__ == "__main__":
    main()
