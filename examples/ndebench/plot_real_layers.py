#!/usr/bin/env python3
"""
Visualize intermediate sampling steps for NSF-Linear, DFN16, HDF and HGGrid,
using the *real* SlangPy-trained networks on examples/einstein.png (not an
offline numpy/EM approximation) -- see Stages.slang and each architecture's
own sampleTrajectoryFixed()/stage1MarginalX()/stage1MarginalCoarse()/
stage1LogDensityX0() instrumentation methods, added alongside their real
sample()/evalLogPDF without changing either.

For each architecture: train to convergence via the same NLL loop
benchmark.py uses (image-sampled data, trainNLL + Adam), then evaluate,
with NO random sampling anywhere in this script:
  - stage 2 (final) density: the exact evalLogPDF grid, via the existing
    inferEval_<Arch> kernel (same one benchmark.py itself uses).
  - stage 1 (after layer 1) density: also exact. DFN16/HDF/HGGrid's first
    layer doesn't depend on any continuous input (IArchitecture's ctx is a
    trivial constant), so it's a direct softmax read-back of a handful of
    numbers, not a histogram of samples. NSF-Linear's first layer does
    depend on a continuous (still-latent) z1, so it's evaluated at the same
    fixed z1=0 its own hero sample uses -- i.e. just that one coupling
    step's log-det, exponentiated; layer 2 isn't applied at all.
  - one deterministic "hero" sample trajectory (prior / after layer 1 /
    final) per architecture: every random draw sample() would make is fixed
    at the median quantile 0.5 instead, so the traced point is a
    reproducible "typical" sample rather than one lucky/unlucky realization.

Run with: uv run plot_real_layers.py [--seconds-per-arch S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import slangpy as spy
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import benchmark as bm  # noqa: E402  (reuse its Runner/make_runner/image helpers)

ARCHS = ["NSFLinear", "DFN16", "HDF", "HGGrid"]
LABELS = {
    "NSFLinear": "NSF-Linear  (K=16 coupling flow)",
    "DFN16": "DFN16  (factorization, nearest-neighbor)",
    "HDF": "HDF  (8x8 coarse hist -> 8x8 fine hist)",
    "HGGrid": "HGGrid  (8x8 coarse hist -> K=4 GMM)",
}
RES = 220          # heatmap grid resolution
SEED = 42          # training data (image sampling) seed only -- nothing plotted is randomly sampled
K_DFN16 = 16
G_COARSE = 8


def make_query_grid(res: int) -> np.ndarray:
    c = (np.arange(res) + 0.5) / res
    gx, gy = np.meshgrid(c, c, indexing="xy")
    return np.ascontiguousarray(np.stack([gx, gy], axis=-1), dtype=np.float32).reshape(-1, 2)


def train_equal_time(r: bm.Runner, arch: str, seconds: float, group_size: int, seed: int) -> None:
    r.reset(arch)
    total_train = 0.0
    step = 0
    with tqdm(total=seconds, desc=f"{arch} train", unit="gpu-s") as pbar:
        while total_train < seconds:
            result = r.train_group(arch, step, group_size, 1e-3, 128.0, 1.0, seed)
            step = result["step"]
            total_train += result["total"]
            pbar.update(min(seconds, total_train) - pbar.n)


def eval_density(r: bm.Runner, arch: str, query: spy.Buffer, out: spy.Buffer, n: int) -> np.ndarray:
    r.kernels[bm.REGISTRY[arch]["eval"]].dispatch(
        thread_count=[n, 1, 1],
        vars={"gParams": r.params, "gSamples": query, "gLogPDFs": out, "gLayout": r.layout_by_arch[arch]},
    )
    r.device.wait()
    return np.exp(np.frombuffer(out.to_numpy(), dtype=np.float32).copy())


def hero_sample(r: bm.Runner, arch: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One deterministic 'typical' trajectory (u=0.5 everywhere sample() would
    draw randomly) via hero_<Arch> -- see Stages.slang."""
    prior_buf = bm.buffer(r.device, None, 8)
    after1_buf = bm.buffer(r.device, None, 8)
    final_buf = bm.buffer(r.device, None, 8)
    r.kernels[f"hero_{arch}"].dispatch(
        thread_count=[1, 1, 1],
        vars={
            "gParams": r.params,
            "gHeroPrior": prior_buf,
            "gHeroAfterLayer1": after1_buf,
            "gHeroFinal": final_buf,
            "gLayout": r.layout_by_arch[arch],
        },
    )
    r.device.wait()
    to2 = lambda b: np.frombuffer(b.to_numpy(), dtype=np.float32).reshape(2)
    return to2(prior_buf), to2(after1_buf), to2(final_buf)


def stage1_density(r: bm.Runner, arch: str, marginal_buf: spy.Buffer, res: int) -> np.ndarray:
    """Exact after-layer-1 density on an (res, res) grid (row ~ y, col ~ x) --
    no sampling anywhere. See Stages.slang's stage1Marginal_<Arch> kernels."""
    query_x = (np.arange(res) + 0.5) / res

    if arch == "DFN16":
        r.kernels["stage1Marginal_DFN16"].dispatch(
            thread_count=[K_DFN16, 1, 1],
            vars={"gParams": r.params, "gStage1Marginal": marginal_buf, "gLayout": r.layout_by_arch[arch]},
        )
        r.device.wait()
        probs = np.frombuffer(marginal_buf.to_numpy(), dtype=np.float32)[:K_DFN16].copy()
        col_bin = np.clip((query_x * K_DFN16).astype(int), 0, K_DFN16 - 1)
        density_1d = probs[col_bin] * K_DFN16
        return np.tile(density_1d[None, :], (res, 1))

    if arch in ("HDF", "HGGrid"):
        r.kernels[f"stage1Marginal_{arch}"].dispatch(
            thread_count=[G_COARSE * G_COARSE, 1, 1],
            vars={"gParams": r.params, "gStage1Marginal": marginal_buf, "gLayout": r.layout_by_arch[arch]},
        )
        r.device.wait()
        probs = np.frombuffer(marginal_buf.to_numpy(), dtype=np.float32)[: G_COARSE * G_COARSE].copy()
        probs2d = probs.reshape(G_COARSE, G_COARSE)  # [cx, cy], matching architectures' grid2bin(cell)=cx*dim+cy
        idx = np.clip((query_x * G_COARSE).astype(int), 0, G_COARSE - 1)  # same binning on both axes
        # dens[row, col] = probs2d[idx[col], idx[row]] * 64 (row ~ y, col ~ x)
        return probs2d[idx][:, idx].T * (G_COARSE * G_COARSE)

    # NSFLinear: invert layer 1 (conditioned on the hero trajectory's fixed
    # z1=0), then log-prior + logAbsDet per query column -- no sampling or
    # integration. Kernel returns log-density (matches evalLogPDF's own
    # convention), so exp() here like eval_density() does for stage 2.
    r.kernels["stage1Marginal_NSFLinear"].dispatch(
        thread_count=[res, 1, 1],
        vars={
            "gParams": r.params,
            "gStage1Marginal": marginal_buf,
            "gLayout": r.layout_by_arch[arch],
            "Stage1CB": {"gStage1Count": res},
        },
    )
    r.device.wait()
    density_1d = np.exp(np.frombuffer(marginal_buf.to_numpy(), dtype=np.float32)[:res].copy())
    return np.tile(density_1d[None, :], (res, 1))


def save_frame_pdf(dens: np.ndarray, pt: np.ndarray, vmax: float, path: Path) -> None:
    """Bare single-panel PDF: just the density image and the marker, no axes,
    ticks, title, or border -- for dropping straight into a figure/slide."""
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(dens, extent=(0, 1, 0, 1), origin="lower", cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    ax.scatter([pt[0]], [pt[1]], c="cyan", s=140, edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # row 0 = top of image, matches load_image/eval convention
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=bm.ROOT / "examples/einstein.png")
    p.add_argument("--seconds-per-arch", type=float, default=6.0)
    p.add_argument("--batch-size", type=int, default=1 << 16)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "nde_layers_real.png")
    args = p.parse_args()

    masses, width, height = bm.load_image(args.image)
    marginal, conditional = bm.image_distribution(masses)
    grid = bm.texel_centers(width, height)

    print("Building runner (compiling shaders for", ", ".join(ARCHS), ")...")
    r = bm.make_runner(marginal, conditional, width, height, grid, args.batch_size, 4096, ARCHS)

    for arch in ARCHS:
        r.kernels[f"hero_{arch}"] = r.device.create_compute_kernel(
            r.device.load_program(module_name="Stages", entry_point_names=[f"hero_{arch}"])
        )
        r.kernels[f"stage1Marginal_{arch}"] = r.device.create_compute_kernel(
            r.device.load_program(module_name="Stages", entry_point_names=[f"stage1Marginal_{arch}"])
        )

    query = make_query_grid(RES)
    query_buf = bm.buffer(r.device, query, query.nbytes, rw=False)
    eval_out = bm.buffer(r.device, None, RES * RES * 4)
    marginal_buf = bm.buffer(r.device, None, RES * 4)  # RES >= 64, big enough for every arch's use

    results = {}
    for arch in ARCHS:
        train_equal_time(r, arch, args.seconds_per_arch, 64, SEED)

        stage2 = eval_density(r, arch, query_buf, eval_out, RES * RES).reshape(RES, RES)
        stage1 = stage1_density(r, arch, marginal_buf, RES)
        prior_pt, after1_pt, final_pt = hero_sample(r, arch)

        results[arch] = dict(
            stage1=stage1.astype(np.float32),
            stage2=stage2.astype(np.float32),
            prior=prior_pt,
            after1=after1_pt,
            final=final_pt,
        )

    fig, axes = plt.subplots(len(ARCHS), 3, figsize=(11, 15), constrained_layout=True)
    col_titles = ["Prior + 1 sample", "After layer 1", "After layer 2 (final)"]
    stage_keys = ["prior", "layer1", "final"]
    frames_dir = args.output.parent / "frames"

    for row, arch in enumerate(ARCHS):
        d = results[arch]
        vmax = max(d["stage1"].max(), d["stage2"].max(), 1e-6)
        panels = [
            (np.ones((RES, RES)), d["prior"], col_titles[0]),
            (d["stage1"], d["after1"], col_titles[1]),
            (d["stage2"], d["final"], col_titles[2]),
        ]
        for stage_key, (dens, pt, _) in zip(stage_keys, panels):
            save_frame_pdf(dens, pt, vmax, frames_dir / f"{arch}_{stage_key}.pdf")
        for col, (dens, pt, title) in enumerate(panels):
            ax = axes[row, col]
            # Same vmax for all three panels in a row: the flat prior (density
            # == 1 everywhere) should render as dim/uniform, not blown out to
            # the colormap's bright end the way a separate vmax=1.0 did.
            # nearest: these densities are genuinely piecewise-constant (histogram
            # bins) -- the default interpolation blurs bin boundaries into a
            # false-looking gradient when upsampling this array for display.
            ax.imshow(
                dens, extent=(0, 1, 0, 1), origin="lower", cmap="magma", vmin=0, vmax=vmax,
                interpolation="nearest",
            )
            ax.scatter([pt[0]], [pt[1]], c="cyan", s=90, edgecolors="white", linewidths=1.2, zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)  # row 0 = top of image, matches load_image/eval convention
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(LABELS[arch], fontsize=9.5, labelpad=8)

    fig.suptitle(
        f"ndebench architectures on {args.image.name}: real SlangPy-trained prior -> layer 1 -> layer 2\n"
        f"({args.seconds_per_arch:.0f} GPU-s training each; 1 deterministic sample per row, no MC noise)",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")
    print(f"wrote {len(ARCHS) * len(stage_keys)} bare frame PDFs to {frames_dir}/")


if __name__ == "__main__":
    main()
