#!/usr/bin/env -S uv run --script
"""Plot artifacts written by benchmark.py; this script never opens a GPU.

Produces four separate PDFs (styled after lab-neural-importance-sampling/figures'
ndebench_convergence.py / ndebench_maps.py -- colorblind-safe qualitative palette,
log-log grid with sub-decade ticks, frameless legends) rather than one dense PNG:

- convergence.pdf -- KL(target || model) vs. training GPU time, one line per focus arch.
- pareto.pdf       -- ALL architectures, sampling throughput vs. importance-sampling ratio
                       variance (log-log, so a line of constant eq-time achieved variance is
                       a straight diagonal -- see `plot_pareto`), bubble size = params, with
                       the speed/variance Pareto frontier traced and labeled.
- images.pdf       -- learned-density thumbnails, reference + focus archs only.
- table.pdf        -- the focus archs' numeric comparison, as its own figure.

"Focus" (what convergence.pdf/images.pdf/table.pdf show) is the pareto plot's own
Pareto-optimal set on sampling speed vs. IS-ratio variance (see `pareto_frontier`) plus a
short, hand-picked ALWAYS_SHOW list of architectures worth seeing even though they're
dominated (NSF-Q's training instability, DF-N/DF-L as NSF's explicit-density
counterparts) -- so the frontier itself stays fully data-driven while still not
silently dropping the handful of architectures whose story isn't "did it win."
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
KL_FLOOR = 1e-4

# Display names, mirroring lab-neural-importance-sampling/figures/ndebench_data.py's
# LABELS (kept independent, not imported, since this script must run standalone without
# the lab checkout -- see the module docstring's "never opens a GPU"/self-contained goal).
#
# The HDF*/HGGrid* variants spell out their actual cascade geometry (matching benchmark.py's
# hdf_levels_layout(G, L, ...)/hggrid_levels_layout(G, L, ..., K, ...) calls: L levels of a
# GxG grid, "NG" = HGGrid's final N-Gaussian continuous head) instead of the bare "Fast"/
# "Fastest" qualifiers -- those named a speed ranking, not what configuration produced it.
# "(d2)" marks the two variants that share their baseline's exact grid geometry but use a
# 2-layer MLP instead of the baseline's 3 (see FINDING.md's HDF/HGGrid optimal-config entry).
LABELS = {
    "TMM": "TMM",
    "DFN": "DF-N",
    "DFL": "DF-L",
    "NSFLinear": "NSF-L",
    "NSFQuadratic": "NSF-Q",
    "NSFRQS": "NSF-RQS",
    "HDF": "HDF 8×8→8×8",
    "HGGrid": "HGGrid 8×8→4G",
    "HDFFast": "HDF 8×8→8×8 (d2)",
    "HDFFastest": "HDF 4×4→4×4→4×4",
    "HGGridFast": "HGGrid 4×4→4×4→8G",
    "HGGridFastest": "HGGrid 8×8→4G (d2)",
    # Bin-matched to NSFLinear's kNumBins=16 (default DFN/DFL use 32) -- see FINDING.md's
    # DF-vs-NSF training-speed entry and architectures/DFN.slang's DFNImpl<K> comment.
    "DFN16": "DF-N (16 bins)",
    "DFL16": "DF-L (16 bins)",
}

# Colorblind-safe (Okabe-Ito) colors for the figures' focus set (Pareto frontier plus the
# always-shown architectures below), extended with a few standard tab10 hues past Okabe-
# Ito's own 8 -- the frontier can grow (e.g. a new architecture that's simply the fastest
# thing benchmarked joins it at the fast end even if nothing else changes), so this needs
# headroom rather than a fixed 8. Everything not in the focus set still shares one neutral
# gray (GRAY, below) rather than drawing its own color, which is what keeps a 12+-way
# pareto scatter readable instead of a color-legend soup.
FOCUS_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442",
    "#9467BD", "#8C564B", "#17BECF", "#7F7F7F",
)
GRAY = "#B0B0B0"
DARK_GRAY = "#595959"

# Kept in the comparison even when they're not Pareto-optimal on speed/KL: NSF-Q is the
# architecture's own documented training-instability cautionary tale (FINDING.md), and
# DF-N/DF-L are the direct "explicit discretized density" counterparts to the NSF coupling
# flows -- worth seeing side by side against NSF-L/NSF-RQS even though neither wins on
# speed or KL here.
ALWAYS_SHOW = ("NSFQuadratic", "DFN", "DFL")


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


def style_log_axis(axis) -> None:
    """Sub-decade ticks (1-2-5) so a log axis reads as plain numbers instead of sparse
    powers of ten, matching lab-neural-importance-sampling/figures' convergence plots."""
    axis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def metric_table(report: dict) -> dict[str, dict]:
    """Flatten each architecture's nested results.json fields into one dict of the scalars
    every figure below needs (kl, variance, timings, params), keyed by architecture name.
    None for any field that's missing/non-finite, so callers can filter rather than crash."""
    out = {}
    for name, result in report["architectures"].items():
        meta = result.get("metadata", {})
        timing = result.get("timing", {})
        final = result.get("final_metrics", {})
        eval_stats = timing.get("pdf_evaluation", {})
        sample_stats = timing.get("sampling", {})

        def finite(x):
            return float(x) if isinstance(x, (int, float)) and np.isfinite(x) else None

        sample_throughput = sample_stats.get("throughput_per_s")
        eval_throughput = eval_stats.get("throughput_per_s")
        sample_mrays = finite(sample_throughput / 1e6) if sample_throughput else None
        variance = finite(final.get("sample_importance_ratio_variance"))

        # benchmark.py's own "mean_ms" field (time_kernel(), ~line 709) really is
        # milliseconds (mean GPU seconds for one --timing-workload=4096-ray batch
        # dispatch, times 1000) despite reading as a suspiciously tiny number -- e.g. a
        # 4096-ray batch at ~400 Mrays/s legitimately takes ~0.01 ms (10 microseconds),
        # not the 0.01 NANOseconds a bare "[us]" label without this x1000 would imply
        # (physically impossible: faster than a single GPU clock cycle). Converted to
        # actual microseconds here so the table's "[us]" columns show what they claim to.
        def batch_us(stats: dict) -> float | None:
            mean_ms = finite(stats.get("mean_ms"))
            return mean_ms * 1000.0 if mean_ms is not None else None

        out[name] = {
            "params": meta.get("trainable_parameters"),
            "buffer_fp16": meta.get("parameter_elements_fp16"),
            "train_ms": finite(timing.get("mean_training_ms_per_update")),
            "eval_us": batch_us(eval_stats),
            "eval_mrays": finite(eval_throughput / 1e6) if eval_throughput else None,
            "sample_us": batch_us(sample_stats),
            "sample_mrays": sample_mrays,
            "kl": finite(final.get("kl_divergence")),
            "variance": variance,
            "degenerate": finite(final.get("sample_importance_ratio_degenerate_fraction")),
            "eqtime_var": eqtime_variance(variance, sample_mrays),
        }
    return out


def eqtime_variance(variance: float | None, sample_mrays: float | None, budget_ms: float = 1.0) -> float | None:
    """Monte-Carlo estimator variance actually achieved by importance-sampling from this
    model for `budget_ms` milliseconds of pure sampling, i.e. Var(p_ref/p_model) / N where
    N is however many samples that time buys at this architecture's own measured sampling
    throughput -- the single number a NIS user actually cares about (a distribution can win
    on speed, on Var(ratio), or on neither, and still lose or win here). Ranking by this is
    equivalent to ranking by variance/throughput for ANY budget (the budget is just a
    constant scale factor); 1ms is chosen purely to keep the printed numbers in a legible
    range, not because it's individually meaningful. Both final_metrics.
    sample_importance_ratio_variance and timing.sampling.throughput_per_s already exist in
    every results.json this script reads, so this needs no new benchmark.py measurement or
    GPU rerun -- see FINDING.md/this session's own notes for the (larger) alternative of
    tracking this per-checkpoint during training instead of only at the final model."""
    if variance is None or sample_mrays is None or sample_mrays <= 0:
        return None
    samples_in_budget = sample_mrays * 1e6 * (budget_ms / 1000.0)
    return variance / samples_in_budget


def pareto_frontier(metrics: dict[str, dict]) -> list[str]:
    """Architecture names on the (sampling speed, IS-ratio variance) Pareto frontier:
    maximize sample_mrays, minimize variance. Returned fastest-first. A point is on the
    frontier iff no faster point also beats it on variance -- equivalently, walking
    fastest-to-slowest, it's a new running minimum of variance.

    Variance (not KL) is the criterion pareto.pdf actually plots and this frontier is
    drawn on: it's the quantity a real importance-sampling renderer pays for (Var(ratio)
    directly sets estimator noise for a fixed sample count), where KL is a deterministic
    grid-quadrature quantity with no sampling-cost interpretation. On this benchmark's
    data the two rankings happen to pick the same architectures, but variance is the
    principled choice, not a coincidence."""
    candidates = [n for n, m in metrics.items() if m["sample_mrays"] is not None and m["variance"] is not None]
    ordered = sorted(candidates, key=lambda n: metrics[n]["sample_mrays"], reverse=True)
    frontier = []
    best_variance = float("inf")
    for name in ordered:
        if metrics[name]["variance"] < best_variance:
            frontier.append(name)
            best_variance = metrics[name]["variance"]
    return frontier


def plot_convergence(output: Path, report: dict, entropy: float, focus: list[str], colors: dict[str, str]) -> None:
    """KL(target || model) vs. cumulative training GPU time, one line per focus arch.

    GPU seconds (not optimizer step) is the x-axis: benchmark.py's --seconds-per-method
    budgets exactly this quantity (GPU-timestamp training-kernel time, not wall-clock),
    so under the default equal-time run every line should reach roughly the same
    right-hand edge, and step count alone wouldn't compare architectures that cost a
    different amount per step (e.g. NSF-RQS's train_ms is ~4x TMM's, see table.pdf).
    """
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for name in focus:
        rows = report["architectures"][name]["checkpoints"]
        # Skip checkpoint 0 (pre-training init): gpu_seconds is 0 there, unplaceable on
        # log-x, and carries no training-dynamics information.
        rows = [r for r in rows if r["cumulative_training_gpu_seconds"] > 0]
        x = [r["cumulative_training_gpu_seconds"] for r in rows]
        y = [max(r["nll"] - entropy, KL_FLOOR) for r in rows]
        ax.plot(x, y, color=colors[name], lw=1.4, label=LABELS.get(name, name))

    ax.set(xlabel="training time [GPU s]", ylabel="KL(target || model)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", color="0.9", lw=0.5)
    style_log_axis(ax.xaxis)
    style_log_axis(ax.yaxis)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output / "convergence.pdf")
    plt.close(fig)


def label_push_direction(name: str, positions_log: dict[str, tuple[float, float]]) -> tuple[float, float, str, str]:
    """(dx, dy, ha, va) offset-points tuple for ax.annotate, pointing away from `name`'s
    nearest neighbor among `positions_log` (every plotted point's (log10 x, log10 y)) --
    whichever axis that neighbor is closer on, in SCREEN terms: a neighbor with smaller
    data-y is above (variance's y-axis is inverted, lower = better = higher on screen), a
    neighbor with smaller data-x is to the left (x-axis is not inverted)."""
    x, y = positions_log[name]
    other = min(
        (positions_log[n] for n in positions_log if n != name),
        key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2,
        default=None,
    )
    if other is None:
        return (0, 22, "center", "bottom")
    ddx, ddy = x - other[0], y - other[1]
    # 22-24pt, well past the 10-12pt a bare label would need: has to clear not just the
    # label's own marker but the NEIGHBOR's bubble radius too (trainable params ranges
    # ~5x across this benchmark's architectures, and NSF-RQS's own bubble -- the biggest
    # in this dataset -- extends most of the way to its closest neighbors regardless of
    # which way THEIR label is pushed; see git history).
    if abs(ddy) >= abs(ddx):
        return (0, -22, "center", "top") if ddy >= 0 else (0, 22, "center", "bottom")
    return (24, 0, "left", "center") if ddx >= 0 else (-24, 0, "right", "center")


def plot_pareto(
    output: Path, metrics: dict[str, dict], frontier: list[str], focus: list[str], colors: dict[str, str]
) -> None:
    """Sampling speed vs. IS-ratio variance for every architecture; bubble area encodes
    trainable parameter count. Both axes log-scaled specifically so that a line of constant
    eq-time achieved variance (variance / (throughput * budget) -- see eqtime_variance(),
    the same quantity table.pdf's "eq-time IS var" column reports) is a straight diagonal:
    variance = (eqtime_var * budget) * throughput is linear in log-log space. A few of
    those diagonals are drawn as reference -- a point's perpendicular distance below one
    tells you it beats that eq-time variance, which speed and variance alone (as separate
    axes) don't make visually obvious.

    Three tiers, to keep a 12-point scatter legible instead of a 12-label pileup:
    `frontier` (this data's own Pareto-optimal set) is bold, connected, and labeled;
    `focus` minus `frontier` (the always-shown-but-dominated architectures) gets a plain
    colored dot with a small label; everything else is an unlabeled gray dot, named in a
    caption instead of fighting for space on the plot itself."""
    all_names = [n for n, m in metrics.items() if m["sample_mrays"] is not None and m["variance"] is not None]
    params = [metrics[n]["params"] for n in all_names if metrics[n]["params"] is not None]
    lo, hi = min(params), max(params)
    other = [n for n in all_names if n not in focus]

    def size(p: int) -> float:
        t = 0.0 if hi <= lo else (p - lo) / (hi - lo)
        return 50.0 + 260.0 * t

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)

    for name in other:
        m = metrics[name]
        ax.scatter(m["sample_mrays"], m["variance"], s=size(m["params"]), color=GRAY, alpha=0.55, zorder=2)

    # Frontier line: `frontier` is already sorted fastest-first (pareto_frontier's own
    # order), so plotting it as-is traces the efficient-frontier staircase from
    # "fast/coarse" to "slow/best-quality" without a separate sort here.
    frontier_x = [metrics[n]["sample_mrays"] for n in frontier]
    frontier_y = [metrics[n]["variance"] for n in frontier]
    ax.plot(frontier_x, frontier_y, color=DARK_GRAY, lw=1.0, ls="--", zorder=3)

    # Non-frontier focus archs (NSF-Q/DF-N/DF-L): a plain dot plus a small, non-bold label
    # offset consistently below-right -- these sit well clear of the crowded frontier
    # cluster (see the coordinates in results.json), so they don't need the frontier
    # labels' collision-avoidance cycling.
    for name in focus:
        if name in frontier:
            continue
        m = metrics[name]
        ax.scatter(m["sample_mrays"], m["variance"], s=size(m["params"]), color=colors[name], edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(
            LABELS.get(name, name),
            (m["sample_mrays"], m["variance"]),
            xytext=(7, -7),
            textcoords="offset points",
            fontsize=7,
            color=colors[name],
            ha="left",
            va="top",
            zorder=3,
        )

    # Direction of each frontier label is chosen per-point (push away from that point's
    # own nearest neighbor among EVERYTHING plotted, gray dots included) rather than
    # cycled by frontier position: which architectures end up adjacent in this data can
    # shift as the frontier's own membership changes (e.g. a new architecture becoming
    # the fastest point moves every later index by one), and a fixed cycle tuned to one
    # frontier's shape breaks the moment that shape changes -- see git history for the
    # DFN16 case that motivated this.
    positions_log = {
        n: (math.log10(metrics[n]["sample_mrays"]), math.log10(metrics[n]["variance"])) for n in all_names
    }
    for name in frontier:
        m = metrics[name]
        ax.scatter(
            m["sample_mrays"],
            m["variance"],
            s=size(m["params"]),
            color=colors[name],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        dx, dy, ha, va = label_push_direction(name, positions_log)
        ax.annotate(
            LABELS.get(name, name),
            (m["sample_mrays"], m["variance"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=colors[name],
            ha=ha,
            va=va,
            zorder=4,
        )

    # Directionality goes in the axis labels themselves (lower = better on variance is
    # already unusual enough to spell out) rather than floating corner annotations --
    # those had nowhere to sit that stayed clear of the frontier cluster or the legends.
    ax.set(
        xlabel="sampling throughput [Mrays/s]  (higher = faster)",
        ylabel="importance-sampling ratio variance  (lower = better)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    style_log_axis(ax.xaxis)
    style_log_axis(ax.yaxis)
    ax.grid(True, which="both", color="0.92", lw=0.5)

    # Diagonals of constant eq-time achieved variance (see docstring): fixed BEFORE adding
    # the diagonal lines themselves, so they span exactly the data's own range instead of
    # matplotlib re-autoscaling to fit a diagonal that overshoots the actual points.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    def diagonal_y(x, eqtime_var):
        return eqtime_var * x * 1e6 * (1.0 / 1000.0)  # eqtime_variance()'s own formula, budget_ms=1, solved for variance

    xs = np.array(xlim)

    def inset_x(frac):
        return float(np.exp(np.log(xlim[0]) + frac * (np.log(xlim[1]) - np.log(xlim[0]))))

    # Labeled at a fixed inset x per line (in log space) rather than at its literal edge
    # crossing: the steepest line (1e-6) sits in the "worse" (high-variance) region across
    # its ENTIRE visible span, which is exactly where the size legend also sits (lower
    # right) -- so it gets its own smaller inset fraction (left of the legend) while the
    # two shallower lines, which pass well above the legend everywhere on the right, use a
    # shared inset near the right edge.
    label_fracs = {1e-7: 0.92, 3e-7: 0.92, 1e-6: 0.55}
    for eqtime_var in (1e-7, 3e-7, 1e-6):
        ax.plot(xs, diagonal_y(xs, eqtime_var), color="0.75", lw=0.7, ls=":", zorder=1)
        x_label = inset_x(label_fracs[eqtime_var])
        ax.annotate(
            f"{eqtime_var:.0e}",
            (x_label, diagonal_y(x_label, eqtime_var)),
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="0.6",
            ha="center",
            va="bottom",
            zorder=1,
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Bubble-size key as plain text in the caption below, not a scatter-marker legend: a
    # legend box claims a fixed corner of the plot, and which corner is actually empty
    # depends on which architectures end up plotted where -- which changes as the
    # benchmark's own architecture list grows (see label_push_direction's docstring for
    # the same issue hitting the frontier labels). Text in the caption has no position to
    # collide with.
    caption = f"bubble area ∝ trainable params ({lo:,}–{hi:,}). dotted diagonals: constant eq-time IS variance (Var(ratio) / samples in a 1ms sampling budget) -- same value table.pdf reports"
    if other:
        other_names = ", ".join(LABELS.get(n, n) for n in other)
        caption += f". also benchmarked, dominated on speed/variance (unlabeled gray dots): {other_names}"
    # Well below the xlabel (which constrained_layout places around axes y=-0.1) so the
    # two never fight for the same row; bbox_inches="tight" below extends the saved
    # canvas to include this regardless of constrained_layout's own margin reservation.
    ax.text(
        0.0,
        -0.18,
        textwrap.fill(caption, width=110),
        transform=ax.transAxes,
        fontsize=6.5,
        color="0.4",
        ha="left",
        va="top",
    )

    fig.savefig(output / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_images(output: Path, output_dir: Path, report: dict, focus: list[str]) -> None:
    """Reference + focus-arch learned-density thumbnails on one shared colormap scale.

    vmax comes from the reference panel's own 99.9th percentile, not maxed across model
    panels -- otherwise one diverged/unstable model would dominate the shared scale and
    wash out every healthy panel, reference included (see git history for the incident
    this guarded against).
    """
    reference_pdf = np.exp(np.load(output_dir / "reference_logpdf.npy"))
    vmax = max(float(np.nanpercentile(reference_pdf, 99.9)), 1e-6)
    panels = [("reference", "reference", reference_pdf)]
    for name in focus:
        values = np.exp(np.load(output_dir / f"{name}_final_logpdf.npy"))
        panels.append((name, LABELS.get(name, name), values))

    ncols = min(3, len(panels))
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    image = None
    for axis, (name, label, values) in zip(axes, panels):
        image = axis.imshow(values, origin="upper", vmin=0.0, vmax=vmax, cmap="magma")
        axis.set_axis_off()
        params = report["architectures"][name]["metadata"].get("trainable_parameters") if name != "reference" else None
        title = "reference" if name == "reference" else f"{label} ({params:,} params)"
        clipped = float(np.mean(values > vmax)) * 100.0
        if clipped > 0.01:
            title += f"\n{clipped:.2f}% saturated"
        axis.set_title(title, fontsize=9)
    for axis in axes[len(panels):]:
        axis.set_axis_off()

    fig.colorbar(image, ax=list(axes[: len(panels)]), shrink=0.85, label="density (clipped at reference's p99.9)")
    fig.savefig(output / "images.pdf")
    plt.close(fig)


def plot_table(output: Path, metrics: dict[str, dict], focus: list[str], colors: dict[str, str]) -> None:
    """The focus architectures' numeric comparison as its own figure -- split out of the
    density-map grid so images.pdf isn't squeezed to make room for caption text."""
    columns = [
        ("params", "params", "{:,.0f}", "min"),
        ("kl", "KL", "{:.4f}", "min"),
        ("variance", "IS var", "{:.3f}", "min"),
        ("eqtime_var", "eq-time IS var\n(1ms budget)", "{:.2e}", "min"),
        ("train_ms", "train\n[ms]", "{:.3f}", "min"),
        ("eval_us", "eval\n[µs]", "{:.2f}", "min"),
        ("sample_us", "sample\n[µs]", "{:.2f}", "min"),
        ("sample_mrays", "sample\n[Mrays/s]", "{:.1f}", "max"),
    ]
    best = {}
    for key, _, _, direction in columns:
        values = [(metrics[n][key], n) for n in focus if metrics[n][key] is not None]
        if not values:
            continue
        best[key] = (min if direction == "min" else max)(values)[1]

    cell_text = []
    for name in focus:
        m = metrics[name]
        row = [LABELS.get(name, name)]
        for key, _, fmt, _ in columns:
            value = m[key]
            row.append("n/a" if value is None else fmt.format(value))
        cell_text.append(row)

    # Arch names now carry the full cascade geometry (e.g. "HDF 4x4>4x4>4x4"), much wider
    # than the rest of the (mostly numeric) columns -- widen the figure for that one column
    # and let auto_set_column_width fit each column to its actual longest cell instead of
    # giving every column the same share of the figure.
    arch_width = max(len(row[0]) for row in cell_text + [["arch"]])
    fig, ax = plt.subplots(figsize=(0.14 * arch_width + 1.1 * len(columns), 0.9 + 0.42 * len(focus)))
    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        colLabels=["arch"] + [label for _, label, _, _ in columns],
        loc="center",
        cellLoc="center",
        edges="horizontal",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(columns) + 1)))
    table.scale(1, 2.0)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.7 if r in (0, len(cell_text)) else 0.4)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            continue
        name = focus[r - 1]
        if c == 0:
            cell.set_text_props(color=colors[name], fontweight="bold")
        else:
            key = columns[c - 1][0]
            if best.get(key) == name:
                cell.set_text_props(fontweight="bold")

    degenerate_notes = [
        f"{LABELS.get(n, n)}: {metrics[n]['degenerate'] * 100:.1f}%"
        for n in focus
        if metrics[n]["degenerate"] and metrics[n]["degenerate"] > 0.001
    ]
    if degenerate_notes:
        ax.text(
            0.0,
            -0.06,
            "degenerate importance-sample fraction: " + ", ".join(degenerate_notes),
            transform=ax.transAxes,
            fontsize=7.5,
            color="0.35",
        )

    fig.savefig(output / "table.pdf", bbox_inches="tight")
    plt.close(fig)


def format_summary_table(report: dict) -> str:
    """Markdown comparison table (arch, params, timing, quality) for ALL architectures --
    mirrors the hand-written tables in FINDING.md, so a sweep's results.json can be turned
    into a pasteable table without re-deriving the columns by hand. Unfiltered (unlike the
    four focus-only PDFs above): this is the raw record of everything that ran."""
    headers = [
        "arch", "params", "buffer (fp16)", "train ms/step",
        "eval us", "eval Mrays/s", "sample us", "sample Mrays/s", "KL", "IS var",
        "eq-time IS var (1ms)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for name, m in metric_table(report).items():
        def num(x, nd=4):
            return "n/a" if x is None else f"{x:,.{nd}f}" if isinstance(x, float) else f"{x:,}"

        row = [
            name,
            num(m["params"], 0),
            num(m["buffer_fp16"], 0),
            num(m["train_ms"]),
            num(m["eval_us"], 2),
            num(m["eval_mrays"], 2),
            num(m["sample_us"], 2),
            num(m["sample_mrays"], 2),
            num(m["kl"]),
            num(m["variance"], 3),
            "n/a" if m["eqtime_var"] is None else f"{m['eqtime_var']:.3e}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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

    metrics = metric_table(report)
    frontier = pareto_frontier(metrics)
    focus = frontier + [n for n in ALWAYS_SHOW if n in metrics and n not in frontier]
    if len(focus) > len(FOCUS_PALETTE):
        raise ValueError(
            f"{len(focus)} focus architectures ({', '.join(focus)}) but FOCUS_PALETTE only "
            f"has {len(FOCUS_PALETTE)} colors -- add another distinguishable hue there"
        )
    colors = dict(zip(focus, FOCUS_PALETTE))

    plot_convergence(output, report, entropy, focus, colors)
    plot_pareto(output, metrics, frontier, focus, colors)
    plot_images(output, output, report, focus)
    plot_table(output, metrics, focus, colors)

    table = format_summary_table(report)
    (output / "summary.md").write_text(table + "\n")

    print(f"Pareto-optimal on sampling speed vs. IS-ratio variance: {', '.join(frontier)}")
    print(f"Also shown (not Pareto-optimal here): {', '.join(n for n in focus if n not in frontier)}")
    print(f"wrote {output / 'convergence.pdf'}, {output / 'pareto.pdf'}, {output / 'images.pdf'}, {output / 'table.pdf'}")
    print(table)
    print(f"wrote {output / 'summary.md'}")


if __name__ == "__main__":
    main()
