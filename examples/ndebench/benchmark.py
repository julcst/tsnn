#!/usr/bin/env -S uv run --script
"""Single-pass, GPU-timed image density-estimation benchmark.

Run with ``uv run benchmark.py``.  Results are deliberately self-contained so
``plot.py`` can be used later on a machine without SlangPy or a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy
from PIL import Image
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Architectures wired into the NDEBENCH_*_KERNEL macros (one struct per
# architectures/<name>.slang, implementing IArchitecture).
ARCHITECTURES = ("TMM", "HGGrid", "DFN", "DFL", "NSFLinear", "NSFQuadratic", "NSFRQS", "HDF")

# Module and entry-point prefix each REGISTRY kind's kernel lives under,
# mirroring the *_KERNEL macro invocations in the corresponding .slang file
# (e.g. NDEBENCH_TRAIN_NLL_KERNEL(TMM) in TrainNLL.slang -> "trainNLL_TMM").
KERNEL_SPECS = {
    "metadata": ("Metadata.slang", "metadata"),
    "init": ("Optimize.slang", "init"),
    "train": ("TrainNLL.slang", "trainNLL"),
    "eval": ("InferEval.slang", "inferEval"),
    "sample": ("InferSample.slang", "inferSample"),
}
MODULE_BY_KIND = {kind: module for kind, (module, _prefix) in KERNEL_SPECS.items()}

# Cartesian product of architecture x kernel kind, instead of a hand-written
# per-architecture dict.
REGISTRY = {
    arch: {kind: f"{prefix}_{arch}" for kind, (_module, prefix) in KERNEL_SPECS.items()}
    for arch in ARCHITECTURES
}


def aligned4(n: int) -> int:
    return (n + 3) & ~3


# Per-architecture MLP shapes as (input, hidden, depth, output), one tuple per
# TSNN.Utils.MLP<> block, in the same order each architecture's own .slang
# file chains them (e.g. DFN's XNet then YNet). Mirrors the `typealias ... =
# MLP<...>` lines in examples/ndebench/architectures/*.slang -- kept in sync
# by hand since Slang has no host-side reflection for this; a mismatch here
# throws in compute_layout's caller (the cross-check against the shader's own
# getParamCount(), see make_runner) rather than silently mis-sizing buffers.
MLP_LAYOUTS: dict[str, list[tuple[int, int, int, int]]] = {
    "TMM": [(1, 32, 3, 16 * 5)],  # Net: K=16
    "HGGrid": [(1, 32, 3, 64), (1 + 16, 32, 3, 4 * 5)],  # CoarseNet, FineNet: K=4
    "DFN": [(1, 32, 3, 32), (1 + 12, 32, 3, 32)],  # XNet, YNet
    "DFL": [(1, 32, 3, 32), (1 + 12, 32, 3, 32)],  # XNet, YNet
    "NSFLinear": [(1 + 32, 32, 3, 16), (1 + 32, 32, 3, 16)],  # kMLP0, kMLP1
    "NSFQuadratic": [(1 + 32, 32, 3, 33), (1 + 32, 32, 3, 33)],  # kSplineOut = 2*16+1
    "NSFRQS": [(1 + 32, 32, 3, 47), (1 + 32, 32, 3, 47)],  # kSplineOut = 3*16-1
    "HDF": [(1, 32, 3, 64), (1 + 16, 32, 3, 64)],  # CoarseNet, FineNet
}


def compute_layout(device: spy.Device, mlp_specs: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, int]:
    """Per-layer (weightOffset, biasOffset) byte pairs for a chain of MLPs,
    mirroring TSNN.Utils.MLP's __init -- except each layer's WEIGHT matrix is
    sized via Device.get_coop_vec_matrix_size(TrainingOptimal) instead of the
    naive `sizeof(half) * inSize * outSize` that __init itself uses.

    That naive size is only correct for RowMajor. `coopVecOuterProductAccumulate`
    (the backward pass's gradient scatter, see TrainNLL.slang) requires
    TrainingOptimal on current hardware -- the Slang stdlib's own doc comment
    on it says so -- and TrainingOptimal's real per-matrix byte size is
    device-defined (there is no shader-side equivalent of this query), and can
    be well above the naive count (e.g. on this GPU a 32x32 fp16 matrix needs
    3584 bytes vs. 2048 naive; a 32x1 needs 512 vs. 64). Using the naive size
    to lay out a TrainingOptimal buffer silently under-allocates every layer,
    which was the actual root cause of the "DFN/DFL lower half washed out"
    bug -- see FINDING.md's "ndebench: DFN/DFL's lower ~47% of rows..." entry
    and its follow-up. Bias vectors are never TrainingOptimal-reordered/padded
    (only matrices are), so they keep the naive size unchanged.

    Returns (flat uint32 array of interleaved (weightOffset, biasOffset)
    pairs -- one per layer, MLPs concatenated in `mlp_specs` order, suitable
    for a `StructuredBuffer<uint2>` -- and the total size in fp16 half-units,
    matching what IArchitecture.getParamCount() reports for the same layout).
    """
    offsets: list[tuple[int, int]] = []
    byte_off = 0
    for input_dim, hidden, depth, output in mlp_specs:
        for l in range(depth + 1):
            in_size = input_dim if l == 0 else hidden
            out_size = output if l == depth else hidden
            weight_off = byte_off
            wsize = device.get_coop_vec_matrix_size(
                out_size, in_size, spy.CoopVecMatrixLayout.training_optimal, spy.DataType.float16
            )
            byte_off = aligned4(byte_off + wsize)
            bias_off = byte_off
            byte_off = aligned4(byte_off + out_size * 2)  # bias: plain fp16 vector
            offsets.append((weight_off, bias_off))
    flat = np.array(offsets, dtype=np.uint32).reshape(-1)
    return flat, byte_off // 2


def load_image(path: Path) -> tuple[np.ndarray, int, int]:
    """Return normalized native-resolution image masses, width, and height."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    masses = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("image luminance must be finite and non-negative")
    total = float(masses.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("image luminance has zero or invalid total weight")
    height, width = masses.shape
    return masses / total, width, height


def image_distribution(masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal-over-rows + per-row-conditional-over-columns CDF decomposition.

    A single flat CDF over all texels is unusable in float32 at einstein.png's
    resolution (1.06e7 texels): mean texel mass (~9.4e-8) is below the ulp of
    1.0 (6.0e-8), so roughly half the texels would collapse into zero-width
    intervals and never be sampled. Splitting into a marginal over rows
    (`height` entries) and a conditional over columns per row (`width`
    entries), each spanning [0, 1], keeps step sizes around 1/height or
    1/width -- far above the ulp.
    """
    row_mass = masses.sum(axis=1)
    marginal = np.ascontiguousarray(np.cumsum(row_mass, dtype=np.float64))
    marginal[-1] = 1.0
    safe = np.where(row_mass > 0, row_mass, 1.0)  # zero-mass rows: avoid 0/0
    conditional = np.ascontiguousarray(np.cumsum(masses, axis=1, dtype=np.float64) / safe[:, None])
    conditional[:, -1] = 1.0
    if not np.isclose(marginal[-1], 1.0) or np.any(np.diff(marginal) < 0):
        raise ValueError("invalid normalized marginal CDF")
    if np.any(np.diff(conditional, axis=1) < 0):
        raise ValueError("invalid normalized conditional CDF")
    return marginal.astype(np.float32), conditional.astype(np.float32)


def texel_centers(width: int, height: int) -> np.ndarray:
    col, row = np.meshgrid(np.arange(width), np.arange(height))
    return np.ascontiguousarray(
        np.stack(((col + 0.5) / width, (row + 0.5) / height), axis=-1), dtype=np.float32
    ).reshape(-1, 2)


def smoke_reduce(masses: np.ndarray, extent: int = 64) -> np.ndarray:
    """Mass-preserving reduction used only to keep the GPU smoke test small."""
    row_groups = np.array_split(np.arange(masses.shape[0]), min(extent, masses.shape[0]))
    col_groups = np.array_split(np.arange(masses.shape[1]), min(extent, masses.shape[1]))
    reduced = np.array(
        [[masses[np.ix_(rows, cols)].sum() for cols in col_groups] for rows in row_groups]
    )
    return reduced / reduced.sum()


def validate() -> None:
    # Non-square orientation also catches x/y transposition in the sampler.
    masses = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float64)
    masses /= masses.sum()
    marginal, conditional = image_distribution(masses)
    assert marginal.dtype == np.float32 and conditional.dtype == np.float32
    assert marginal.shape == (2,) and conditional.shape == (2, 3)
    assert marginal[-1] == 1.0 and np.all(np.diff(marginal) >= 0)
    assert np.all(conditional[:, -1] == 1.0) and np.all(np.diff(conditional, axis=1) >= 0)

    # Zero-mass row: the safe-divide must avoid 0/0, and the row's marginal
    # interval collapses to zero width so it can never be selected.
    zero_row_masses = np.array([[0.0, 0.0], [1.0, 1.0]])
    zero_marginal, zero_conditional = image_distribution(zero_row_masses)
    assert np.all(np.isfinite(zero_conditional)) and zero_marginal[0] == 0.0

    # Two-stage inversion mirror of the GPU sampler (upperBound == np.searchsorted
    # side="right"): confirms it selects only the three non-zero texels with the
    # right row/col orientation -- this is what currently guards against x/y
    # transposition.
    rng = np.random.default_rng(9)
    u_row, u_col = rng.random(4096), rng.random(4096)
    rows = np.clip(np.searchsorted(marginal, u_row, side="right"), 0, 1)
    cols = np.empty(4096, dtype=np.int64)
    for r in np.unique(rows):
        mask = rows == r
        cols[mask] = np.clip(np.searchsorted(conditional[r], u_col[mask], side="right"), 0, 2)
    assert set(np.unique(rows * 3 + cols)) == {1, 3, 5}

    grid = texel_centers(3, 2).reshape(2, 3, 2)
    assert np.allclose(grid[0, :, 1], 0.25) and np.allclose(grid[:, 0, 0], 1 / 6)
    density = masses * 6
    assert np.isclose(density.sum() / 6, 1.0)
    assert np.count_nonzero(masses) == 3  # zero mass is excluded from NLL reduction


def buffer(device: spy.Device, data: np.ndarray | None, nbytes: int, *, rw: bool = True):
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return (
        device.create_buffer(size=nbytes, usage=usage, data=data)
        if data is not None
        else device.create_buffer(size=nbytes, usage=usage)
    )


def package_versions() -> dict[str, str]:
    answer = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("slangpy", "Pillow", "matplotlib"):
        try:
            answer[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            answer[name] = "unavailable"
    return answer


@dataclass
class Runner:
    device: spy.Device
    kernels: dict[str, object]
    params: object
    master: object
    grads: object
    m1: object
    m2: object
    samples: object
    marginal: object
    conditional: object
    grid: object
    eval_out: object
    sample_out: object
    elements_by_arch: dict[str, int]
    padded_by_arch: dict[str, int]
    layout_by_arch: dict[str, object]
    dispatch_threads: int
    grid_count: int
    width: int
    height: int
    batch_size: int
    sample_out_count: int

    def reset(self, arch: str) -> None:
        enc = self.device.create_command_encoder()
        for item in (self.params, self.master, self.grads, self.m1, self.m2):
            enc.clear_buffer(item)
        self.kernels[REGISTRY[arch]["init"]].dispatch(
            thread_count=[self.dispatch_threads, 1, 1],
            vars={
                "gParamsInit": self.params,
                "gParamsMasterInit": self.master,
                "gLayout": self.layout_by_arch[arch],
                "InitCB": {"gInitThreadCount": self.dispatch_threads},
            },
            command_encoder=enc,
        )
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()

    def eval_grid(self, arch: str) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        self.kernels[REGISTRY[arch]["eval"]].dispatch(
            thread_count=[self.grid_count, 1, 1],
            vars={
                "gParams": self.params,
                "gSamples": self.grid,
                "gLogPDFs": self.eval_out,
                "gLayout": self.layout_by_arch[arch],
            },
        )
        self.device.wait()
        elapsed = time.perf_counter() - started
        values = np.frombuffer(self.eval_out.to_numpy(), dtype=np.float32).copy()
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("model produced non-finite logPDF values")
        return values, elapsed

    def generate(self, count: int, stream_offset: int, seed: int, encoder) -> None:
        self.kernels["sampleImage"].dispatch(
            thread_count=[count, 1, 1],
            vars={
                "gMarginalCDF": self.marginal,
                "gConditionalCDF": self.conditional,
                "gSamples": self.samples,
                "ImageSampleCB": {
                    "gWidth": self.width,
                    "gHeight": self.height,
                    "gCount": count,
                    # gStreamOffset is a uint32 cbuffer field
                    # (ImageSample.slang); callers compute it as
                    # step * batch_size, which the equal-GPU-time training
                    # budget can now run far enough (hundreds of thousands of
                    # steps for a cheap-per-step architecture) to overflow
                    # 2**32 -- slangpy's cursor write then raises
                    # `std::bad_cast` instead of silently truncating. Wrap
                    # explicitly: it is only a PCG32 stream-decorrelation
                    # offset (PCG32(seed, gStreamOffset + tid.x) in
                    # ImageSample.slang), so wrapping after ~4 billion
                    # samples just means two steps very far apart in an
                    # already-long run reuse the same stream slice --
                    # inconsequential next to the alternative of crashing.
                    "gStreamOffset": stream_offset & 0xFFFFFFFF,
                    "gSeed": seed,
                },
            },
            command_encoder=encoder,
        )

    def train_group(
        self,
        arch: str,
        step: int,
        group_size: int,
        lr: float,
        loss_scale: float,
        clip: float,
        seed: int,
    ) -> dict:
        """Runs `group_size` train+optimize steps for `arch`, batched into
        one command encoder with a GPU-timestamp query pool spanning the
        whole group (generate/clear/forward/optimize each bracketed by a
        timestamp write) -- this is the exact structure the equal-time
        training loop in `run_architecture` measures per checkpoint, and
        `warm()` below calls this too instead of its own one-command-buffer-
        per-step loop it used to run. Batching group_size dispatches (with
        their query-pool writes) into a single encoder is a different code
        path from `warm()`'s old per-step submission pattern, and whatever
        the driver needs to JIT/allocate for it apparently isn't triggered
        by the old pattern: the first *measured* checkpoint group of every
        run cost 1.3-2x more GPU time than every later group at the same
        size, which is what made the KL-vs-GPU-seconds plot's curves appear
        to start at scattered, offset x-positions (see FINDING.md). Sharing
        this method means warmup and measurement can no longer drift apart
        the same way again.

        Returns this group's new `step`, cumulative `samples_consumed`, and
        its own (generate, clear, forward, optimizer, total) GPU-second
        deltas -- `run_architecture`'s checkpoint() wants the running totals,
        `warm()` only cares that the call happened.
        """
        q = self.device.create_query_pool(spy.QueryType.timestamp, group_size * 5)
        enc = self.device.create_command_encoder()
        for i in range(group_size):
            base = i * 5
            enc.write_timestamp(q, base)
            self.generate(self.batch_size, step * self.batch_size, seed, enc)
            enc.write_timestamp(q, base + 1)
            enc.clear_buffer(self.grads)
            enc.write_timestamp(q, base + 2)
            step += 1
            self.kernels[REGISTRY[arch]["train"]].dispatch(
                thread_count=[self.batch_size, 1, 1],
                vars={
                    "gParams": self.params,
                    "gParamGrads": self.grads,
                    "gSamples": self.samples,
                    "gLayout": self.layout_by_arch[arch],
                    "TrainCB": {"gCount": self.batch_size, "gWeight": loss_scale / self.batch_size},
                },
                command_encoder=enc,
            )
            enc.write_timestamp(q, base + 3)
            self.kernels["optimize"].dispatch(
                thread_count=[self.dispatch_threads, 1, 1],
                vars=self.optimize_vars(arch, step, lr, loss_scale, clip),
                command_encoder=enc,
            )
            enc.write_timestamp(q, base + 4)
        self.device.submit_command_buffer(enc.finish())
        self.device.wait()
        stamps = (
            np.asarray(q.get_results(0, group_size * 5), dtype=np.uint64).reshape(-1, 5)
            / self.device.info.timestamp_frequency
        )
        return {
            "step": step,
            "samples_consumed": step * self.batch_size,
            "generate": float(np.sum(stamps[:, 1] - stamps[:, 0])),
            "clear": float(np.sum(stamps[:, 2] - stamps[:, 1])),
            "forward": float(np.sum(stamps[:, 3] - stamps[:, 2])),
            "optimizer": float(np.sum(stamps[:, 4] - stamps[:, 3])),
            "total": float(np.sum(stamps[:, 4] - stamps[:, 0])),
        }

    def warm(
        self,
        arch: str,
        lr: float,
        loss_scale: float,
        clip: float,
        count: int,
        group_size: int,
        seed: int,
    ) -> None:
        # At least one full group_size-sized group, batched via train_group
        # (see its doc comment) -- quantizing warmup to whole groups matters
        # more than hitting `count` exactly, since the point is exercising
        # the real loop's own checkpoint-group shape at production size, not
        # a specific step count (warm()'s work is thrown away by reset()
        # below regardless).
        step = 0
        while step < max(count, group_size):
            step = self.train_group(arch, step, group_size, lr, loss_scale, clip, seed)["step"]
        self.eval_grid(arch)
        self.kernels[REGISTRY[arch]["sample"]].dispatch(
            thread_count=[self.batch_size, 1, 1],
            vars={
                "gParams": self.params,
                "gSamples": self.sample_out,
                "gLayout": self.layout_by_arch[arch],
                "SampleCB": {"gSeed": 1},
            },
        )
        self.device.wait()
        self.reset(arch)

    def optimize_vars(
        self, arch: str, step: int, lr: float, loss_scale: float, clip: float
    ) -> dict:
        return {
            "gParams": self.params,
            "gParamsMaster": self.master,
            "gParamGrads": self.grads,
            "gMoments1": self.m1,
            "gMoments2": self.m2,
            "CB": {
                "gLearningRate": lr,
                "gCurrentStep": float(step),
                "gLossScale": loss_scale,
                "gGradClip": clip,
                "gParamElementCount": self.padded_by_arch[arch],
                "gDispatchThreadCount": self.dispatch_threads,
            },
        }


def make_runner(
    marginal: np.ndarray,
    conditional: np.ndarray,
    width: int,
    height: int,
    grid: np.ndarray,
    batch_size: int,
    timing_workload: int,
    architectures: list[str],
) -> Runner:
    options = spy.SlangCompilerOptions(
        {
            "include_paths": [HERE, ROOT, ROOT / "TSNN", spy.SHADER_PATH],
            "defines": {"DEBUG_COUNTERS": "0"},
        }
    )
    device = spy.Device(compiler_options=options)

    def load(module: str, entry: str):
        return device.create_compute_kernel(
            device.load_program(module_name=module, entry_point_names=[entry])
        )

    kernels = {
        "optimize": load("Optimize.slang", "optimize"),
        "sampleImage": load("ImageSample.slang", "sampleImage"),
    }
    for arch in architectures:
        for kind, entry in REGISTRY[arch].items():
            kernels[entry] = load(MODULE_BY_KIND[kind], entry)

    # Each architecture's per-layer buffer offsets (see compute_layout's doc
    # comment) are computed host-side, once, from the device's own reported
    # TrainingOptimal matrix sizes -- Slang has no way to query these itself.
    layout_by_arch: dict[str, object] = {}
    elements_by_layout: dict[str, int] = {}
    for arch in architectures:
        flat, elements = compute_layout(device, MLP_LAYOUTS[arch])
        layout_by_arch[arch] = buffer(device, flat, flat.nbytes, rw=False)
        elements_by_layout[arch] = elements

    # Each architecture owns its own parameter-element count; buffers are
    # sized to the largest one so every architecture's real weights fit,
    # while gParamElementCount (in optimize_vars) stays per-architecture so
    # the Adam sweep never walks past a smaller architecture's own tail.
    # The metadata kernel re-derives the same count from `gLayout` on the
    # shader side (T.getParamCount(layout), see IArchitecture.slang) -- kept
    # as a live cross-check that MLP_LAYOUTS above hasn't drifted from the
    # corresponding architectures/*.slang file's own MLP<> typealiases.
    meta = buffer(device, np.zeros(1, np.uint32), 4)
    elements_by_arch: dict[str, int] = {}
    padded_by_arch: dict[str, int] = {}
    for arch in architectures:
        kernels[REGISTRY[arch]["metadata"]].dispatch(
            thread_count=[1, 1, 1], vars={"gMetadata": meta, "gLayout": layout_by_arch[arch]}
        )
        device.wait()
        elements = int(np.frombuffer(meta.to_numpy(), dtype=np.uint32)[0])
        if elements <= 0:
            raise RuntimeError(f"architecture {arch!r} metadata returned no parameters")
        if elements != elements_by_layout[arch]:
            raise RuntimeError(
                f"architecture {arch!r}: MLP_LAYOUTS gives {elements_by_layout[arch]} fp16 "
                f"elements but the shader's own getParamCount(gLayout) reports {elements} -- "
                "MLP_LAYOUTS has drifted from this architecture's .slang file's MLP<> shapes"
            )
        elements_by_arch[arch] = elements
        padded_by_arch[arch] = aligned4(elements)

    padded = max(padded_by_arch.values())
    fp16_bytes = padded * 2
    fp32_bytes = padded * 4
    sample_out_count = max(batch_size, timing_workload)
    return Runner(
        device,
        kernels,
        buffer(device, None, fp16_bytes),
        buffer(device, None, fp32_bytes),
        buffer(device, None, fp16_bytes),
        buffer(device, None, fp32_bytes),
        buffer(device, None, fp32_bytes),
        buffer(device, None, batch_size * 8),
        buffer(device, np.ascontiguousarray(marginal), marginal.nbytes, rw=False),
        buffer(device, np.ascontiguousarray(conditional), conditional.nbytes, rw=False),
        buffer(device, np.ascontiguousarray(grid), grid.nbytes, rw=False),
        buffer(device, None, grid.shape[0] * 4),
        buffer(device, None, sample_out_count * 8),
        elements_by_arch,
        padded_by_arch,
        layout_by_arch,
        256 * 8,
        grid.shape[0],
        width,
        height,
        batch_size,
        sample_out_count,
    )


def verify_sampler(r: Runner, masses: np.ndarray, seed: int) -> None:
    """One-shot GPU dispatch of sampleImage checked against the target masses.

    The only end-to-end check that the GPU sampler reproduces the target
    distribution (as opposed to the CPU-side image_distribution() asserts,
    which never touch the GPU sampler itself).
    """
    count = 1 << 20
    height, width = masses.shape
    out = buffer(r.device, None, count * 8)
    r.kernels["sampleImage"].dispatch(
        thread_count=[count, 1, 1],
        vars={
            "gMarginalCDF": r.marginal,
            "gConditionalCDF": r.conditional,
            "gSamples": out,
            "ImageSampleCB": {
                "gWidth": width,
                "gHeight": height,
                "gCount": count,
                "gStreamOffset": 0,
                "gSeed": seed,
            },
        },
    )
    r.device.wait()
    points = np.frombuffer(out.to_numpy(), dtype=np.float32).reshape(-1, 2)
    hist, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=[height, width], range=[[0.0, 1.0], [0.0, 1.0]]
    )
    hist /= hist.sum()
    tv = 0.5 * float(np.abs(hist - masses).sum())
    if tv >= 0.05:
        raise FloatingPointError(f"GPU image sampler total-variation distance {tv:.4f} >= 0.05")
    print(f"GPU sampler smoke check passed (total-variation distance {tv:.4f})")


def timestamped_inference(
    r: Runner, arch: str, kind: str, workload: int, repeats: int, seed: int
) -> dict:
    key = REGISTRY[arch][kind]
    input_buf = r.grid if kind == "eval" else r.sample_out
    output_buf = r.eval_out if kind == "eval" else r.sample_out
    layout = r.layout_by_arch[arch]
    for i in range(5):
        vars_ = (
            {"gParams": r.params, "gSamples": input_buf, "gLogPDFs": output_buf, "gLayout": layout}
            if kind == "eval"
            else {
                "gParams": r.params,
                "gSamples": output_buf,
                "gLayout": layout,
                "SampleCB": {"gSeed": seed + i},
            }
        )
        r.kernels[key].dispatch(thread_count=[workload, 1, 1], vars=vars_)
    r.device.wait()
    q = r.device.create_query_pool(spy.QueryType.timestamp, repeats * 2)
    for i in range(repeats):
        vars_ = (
            {"gParams": r.params, "gSamples": input_buf, "gLogPDFs": output_buf, "gLayout": layout}
            if kind == "eval"
            else {
                "gParams": r.params,
                "gSamples": output_buf,
                "gLayout": layout,
                "SampleCB": {"gSeed": seed + i},
            }
        )
        r.kernels[key].dispatch(
            thread_count=[workload, 1, 1],
            vars=vars_,
            query_pool=q,
            query_index_before=2 * i,
            query_index_after=2 * i + 1,
        )
    r.device.wait()
    stamps = np.asarray(q.get_results(0, repeats * 2), dtype=np.uint64).reshape(-1, 2)
    seconds = (stamps[:, 1] - stamps[:, 0]) / r.device.info.timestamp_frequency
    mean = float(seconds.mean())
    return {
        "workload": workload,
        "repetitions": repeats,
        "mean_ms": mean * 1000,
        "throughput_per_s": workload / mean,
    }


def sample_importance_ratio_variance(
    r: Runner, arch: str, masses: np.ndarray, width: int, height: int, seed: int, count: int
) -> tuple[float, float, float]:
    """Variance (and mean) of the importance-sampling ratio p_ref(x)/p_model(x)
    for x drawn from the trained model's own sample() -- the standard
    diagnostic for how well a learned sampling distribution matches its
    target: a model that is locally under-confident where the target has
    mass produces a long-tailed ratio and a large variance, even if its NLL
    looks reasonable on average.

    p_ref is the same piecewise-constant-per-texel density
    (masses*width*height) reference_logpdf uses, looked up by nearest texel
    at each continuous sample coordinate.

    A sample the model itself assigns exactly zero density (model_logpdf ==
    -inf) makes the ratio a genuine +inf, not a numerical artifact of this
    function -- observed in practice for a small fraction (~0.2-0.3%) of
    TMM's own samples (TruncatedGMM.sample()'s inverse-normal-CDF step,
    unrelated to this session's DFN/DFL work; not fixed here, flagged in
    FINDING.md). One +inf silently NaNs np.var's mean-of-squares, which would
    make the WHOLE architecture's reported variance meaningless instead of
    just the offending samples'. Mirrors plot.py's own saturated-pixel-
    fraction precedent: exclude non-finite ratios from the statistic and
    report what fraction were excluded, rather than let a handful of
    degenerate draws erase the signal from the rest.
    """
    n = min(count, r.sample_out_count)
    layout = r.layout_by_arch[arch]
    r.kernels[REGISTRY[arch]["sample"]].dispatch(
        thread_count=[n, 1, 1],
        vars={
            "gParams": r.params,
            "gSamples": r.sample_out,
            "gLayout": layout,
            "SampleCB": {"gSeed": seed},
        },
    )
    r.device.wait()
    r.kernels[REGISTRY[arch]["eval"]].dispatch(
        thread_count=[n, 1, 1],
        vars={"gParams": r.params, "gSamples": r.sample_out, "gLogPDFs": r.eval_out, "gLayout": layout},
    )
    r.device.wait()
    xy = np.frombuffer(r.sample_out.to_numpy(), dtype=np.float32).reshape(-1, 2)[:n]
    model_logpdf = np.frombuffer(r.eval_out.to_numpy(), dtype=np.float32).copy()[:n]
    # Clamp to [0, 1) BEFORE scaling by width/height and casting to int: a
    # handful of samples can be a degenerate +-inf/NaN (see this function's
    # docstring), and scaling an inf coordinate first, then casting that
    # still-inf-or-overflowing float to int64, is undefined behavior (numpy
    # warns "invalid value encountered in cast"). Clamping the coordinate
    # itself first keeps every cast in-range.
    unit = np.nextafter(1.0, 0.0, dtype=np.float32)
    xy_clamped = np.nan_to_num(np.clip(xy, 0.0, unit), nan=0.0, posinf=unit, neginf=0.0)
    cols = np.clip((xy_clamped[:, 0] * width).astype(np.int64), 0, width - 1)
    rows = np.clip((xy_clamped[:, 1] * height).astype(np.int64), 0, height - 1)
    ref_density = masses[rows, cols] * (width * height)
    model_density = np.exp(model_logpdf.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref_density > 0, ref_density / model_density, 0.0)
    finite = np.isfinite(ratio)
    degenerate_fraction = 1.0 - float(np.count_nonzero(finite)) / n
    if not np.any(finite):
        return float("nan"), float("nan"), degenerate_fraction
    return (
        float(np.var(ratio[finite], dtype=np.float64)),
        float(np.mean(ratio[finite], dtype=np.float64)),
        degenerate_fraction,
    )


def run_architecture(
    r: Runner,
    arch: str,
    masses: np.ndarray,
    args: argparse.Namespace,
    output: Path,
    reference_entropy: float,
) -> dict:
    r.warm(
        arch,
        args.learning_rate,
        args.loss_scale,
        args.gradient_clip,
        args.warmup_count,
        args.evaluation_interval,
        args.seed,
    )
    initial, grid_time = r.eval_grid(arch)
    h, w = masses.shape

    # Started here, after warmup and this architecture's initial grid eval --
    # the clock every checkpoint's "wall_seconds" (below) and the equal-time
    # training budget (further down) are both measured against. This is the
    # ONLY seconds figure that actually reflects the equal-time budget: see
    # checkpoint()'s docstring-style comment for why
    # "cumulative_training_gpu_seconds" is a different, smaller number.
    wall_started = time.perf_counter()

    def checkpoint(step: int, consumed: int, timing: float, eval_seconds: float) -> dict:
        logpdf, seconds = r.eval_grid(arch)
        nll = -float(
            np.sum(masses[masses > 0] * logpdf.reshape(h, w)[masses > 0], dtype=np.float64)
        )
        return {
            "step": step,
            "samples_consumed": consumed,
            "nll": nll,
            # Sum of pure GPU-kernel dispatch time for the train-loop kernels
            # only (generate+clear+forward+optimize), measured via GPU
            # timestamp queries. This IS the quantity --seconds-per-method
            # budgets equally across architectures (see the training-loop
            # comment below for why GPU time rather than wall-clock).
            "cumulative_training_gpu_seconds": timing,
            "grid_evaluation_seconds": seconds + eval_seconds,
            # Wall-clock time, purely informational: how much real time this
            # checkpoint's slice of training + evaluation actually took,
            # Python/driver overhead and sync waits included. Not what the
            # equal-time budget targets -- see the training-loop comment.
            "wall_seconds": time.perf_counter() - wall_started,
        }, logpdf

    rows, final = [], initial
    initial_nll = -float(
        np.sum(masses[masses > 0] * initial.reshape(h, w)[masses > 0], dtype=np.float64)
    )
    rows.append(
        {
            "step": 0,
            "samples_consumed": 0,
            "nll": initial_nll,
            "cumulative_training_gpu_seconds": 0.0,
            "grid_evaluation_seconds": grid_time,
            "wall_seconds": 0.0,
        }
    )
    total_generate = total_train = total_clear = total_forward = total_opt = 0.0
    step = 0
    total_consumed = 0

    # Equal-time mode (the default: args.steps left unset) bounds each
    # architecture's training loop by GPU-measured training seconds
    # (total_train, the same GPU-timestamp sum stored per checkpoint as
    # "cumulative_training_gpu_seconds") instead of a fixed step count, so
    # every method gets the same GPU compute budget regardless of its
    # per-step cost -- fairer for comparing architectures whose per-step
    # time varies a lot (see the timing table: NSFRQS/HGGrid run ~4-5x the
    # per-update cost of TMM/DFL). Deliberately GPU time, not wall-clock:
    # wall-clock also counts Python/driver dispatch overhead and CPU<->GPU
    # sync waits, which are a per-call constant this benchmark's small
    # per-step batches make disproportionately large -- not representative
    # of a production inference/training loop's actual GPU cost, and it
    # would unfairly penalize an architecture just for issuing more, smaller
    # dispatches. total_train excludes checkpoint eval_grid() calls (a
    # separate "grid_evaluation_seconds" figure) and, since it only starts
    # accumulating once the training loop below begins -- after
    # make_runner's one-time shared kernel compilation and after this
    # architecture's own warmup -- it never counts compile time either.
    # Checked at the top of the loop using the previous iteration's
    # accumulated total, since a group's own GPU time is only known after it
    # runs. Only `--smoke`'s forced `args.steps = 2` (and an explicit
    # `--steps` override) still use the old fixed-step-count path.
    use_step_budget = args.steps is not None
    total_steps = args.steps if use_step_budget else None
    pbar_kwargs = (
        {"total": total_steps, "unit": "step"}
        if use_step_budget
        else {"total": args.seconds_per_method, "unit": "gpu-s"}
    )
    with tqdm(desc=f"{arch} train", **pbar_kwargs) as pbar:
        while True:
            if use_step_budget:
                if step >= total_steps:
                    break
                group_size = min(args.evaluation_interval, total_steps - step)
            else:
                if total_train >= args.seconds_per_method:
                    break
                group_size = args.evaluation_interval
            result = r.train_group(
                arch,
                step,
                group_size,
                args.learning_rate,
                args.loss_scale,
                args.gradient_clip,
                args.seed,
            )
            step = result["step"]
            total_consumed = result["samples_consumed"]
            total_generate += result["generate"]
            total_clear += result["clear"]
            total_forward += result["forward"]
            total_opt += result["optimizer"]
            total_train += result["total"]
            row, final = checkpoint(step, total_consumed, total_train, 0.0)
            rows.append(row)
            if use_step_budget:
                pbar.update(group_size)
            else:
                elapsed = min(args.seconds_per_method, total_train)
                pbar.update(max(0.0, elapsed - pbar.n))
            pbar.set_postfix(nll=f"{row['nll']:.4f}")
    final_image = final.reshape(h, w)
    np.save(output / f"{arch}_final_logpdf.npy", final_image)
    display = np.maximum(final_image, -12.0)
    display = (255 * (display - display.min()) / max(float(np.ptp(display)), 1e-12)).astype(
        np.uint8
    )
    Image.fromarray(display, mode="L").save(output / f"{arch}_final_logpdf.png")

    final_nll = rows[-1]["nll"]
    ratio_variance, ratio_mean, ratio_degenerate_fraction = sample_importance_ratio_variance(
        r, arch, masses, w, h, args.seed, args.variance_sample_count
    )
    return {
        "metadata": {
            "parameter_elements_fp16": r.elements_by_arch[arch],
            "padded_parameter_elements": r.padded_by_arch[arch],
        },
        "checkpoints": rows,
        "final_metrics": {
            # KL(target || model) = H(target, model) - H(target); final_nll
            # IS H(target, model) (the cross-entropy checkpoint() already
            # computes), so this needs no extra GPU evaluation.
            "kl_divergence": final_nll - reference_entropy,
            # Var[p_ref(x)/p_model(x)] for x ~ model.sample() -- the standard
            # importance-sampling diagnostic: large even when NLL looks fine
            # if the model is locally under-confident somewhere the target
            # has mass. See sample_importance_ratio_variance()'s docstring.
            "sample_importance_ratio_variance": ratio_variance,
            "sample_importance_ratio_mean": ratio_mean,
            # Fraction of the variance-estimate samples the model itself
            # assigned exactly zero density (ratio == +inf), excluded from
            # the two stats above rather than left in to NaN them.
            "sample_importance_ratio_degenerate_fraction": ratio_degenerate_fraction,
        },
        "timing": {
            "mean_training_ms_per_update": total_train / step * 1000,
            "sample_generation_seconds": total_generate,
            "gradient_clear_seconds": total_clear,
            "forward_backward_seconds": total_forward,
            "optimizer_seconds": total_opt,
            "cumulative_training_seconds": total_train,
            "pdf_evaluation": timestamped_inference(
                r,
                arch,
                "eval",
                min(args.timing_workload, r.grid_count),
                args.timing_repetitions,
                args.seed,
            ),
            "sampling": timestamped_inference(
                r, arch, "sample", args.timing_workload, args.timing_repetitions, args.seed
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, default=ROOT / "examples/einstein.png")
    p.add_argument("--batch-size", type=int, default=262144)  # Small batch size underoccupies GPU
    p.add_argument(
        "--architectures",
        default=",".join(ARCHITECTURES),
        help=f"comma-separated architecture names; default is all: {', '.join(ARCHITECTURES)}",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--loss-scale", type=float, default=128.0)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="fixed step count per architecture; overrides --seconds-per-method when set",
    )
    p.add_argument(
        "--seconds-per-method",
        type=float,
        default=5.0,
        # Default deliberately much smaller than a wall-clock budget would be:
        # GPU-measured seconds only count actual kernel time, so reaching even
        # a modest target can take much longer in real wall-clock time for a
        # cheap-per-step architecture -- e.g. an earlier 60-wall-second run's
        # GPU-measured total ranged from ~16% (HGGrid) down to under 1% (DFL)
        # of that wall time, so a 60 GPU-second target could take the better
        # part of an hour of real time for the fastest architectures.
        help="equal-time training budget per architecture, in GPU-measured seconds "
        "(cumulative_training_gpu_seconds, not wall-clock -- production-representative GPU "
        "compute, excluding Python/driver overhead and the one-time shared kernel-compile "
        "cost); ignored when --steps is set",
    )
    p.add_argument("--evaluation-interval", type=int, default=64)
    p.add_argument(
        "--warmup-count",
        type=int,
        default=5,
        help="minimum warmup steps before measurement starts; rounded up to a whole "
        "multiple of --evaluation-interval, since warm() batches steps into "
        "--evaluation-interval-sized groups (matching the measured training loop's own "
        "checkpoint-group structure, so the driver's first-use cost for that batched-"
        "dispatch-with-timestamp-queries shape is paid here instead of during the first "
        "measured checkpoint)",
    )
    p.add_argument("--timing-repetitions", type=int, default=100)
    p.add_argument("--timing-workload", type=int, default=4096)
    p.add_argument(
        "--variance-sample-count",
        type=int,
        default=65536,
        help="samples drawn from each trained model to estimate the final importance-ratio "
        "variance (clamped to the sample buffer's own capacity)",
    )
    p.add_argument("--output-directory", type=Path, default=HERE / "output")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="skip automatically running plot.py on the freshly written results",
    )
    args = p.parse_args()
    validate() if args.validate else None
    if args.validate and not args.smoke:
        print("CPU validation passed")
        return
    names = [x.strip() for x in args.architectures.split(",") if x.strip()]
    unknown = sorted(set(names) - REGISTRY.keys())
    if unknown:
        p.error(f"unknown architectures {unknown}; supported: {', '.join(REGISTRY)}")
    counts = [
        args.batch_size,
        args.evaluation_interval,
        args.timing_repetitions,
        args.timing_workload,
        args.variance_sample_count,
    ]
    if args.steps is not None:
        counts.append(args.steps)
    if min(counts) < 1:
        p.error("counts must be positive")
    if args.seconds_per_method <= 0:
        p.error("--seconds-per-method must be positive")
    if args.smoke:
        args.batch_size, args.evaluation_interval = 16, 1
        args.timing_repetitions, args.timing_workload = 2, 16
        args.variance_sample_count = 16
        args.steps = 2
    masses, width, height = load_image(args.image)
    if args.smoke:
        masses = smoke_reduce(masses)
        height, width = masses.shape
    cdf_started = time.perf_counter()
    marginal, conditional = image_distribution(masses)
    cdf_precompute_seconds = time.perf_counter() - cdf_started
    grid = texel_centers(width, height)
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    with np.errstate(divide="ignore"):
        reference_logpdf = np.where(masses > 0, np.log(masses * width * height), -np.inf)
    # H(target): entropy of the reference distribution, needed to turn each
    # architecture's cross-entropy (final NLL) into KL(target || model)
    # without re-evaluating anything on the GPU (see run_architecture).
    reference_entropy = -float(
        np.sum(masses[masses > 0] * reference_logpdf[masses > 0], dtype=np.float64)
    )
    np.save(output / "reference_masses.npy", masses)
    np.save(output / "reference_logpdf.npy", reference_logpdf)
    r = make_runner(
        marginal, conditional, width, height, grid, args.batch_size, args.timing_workload, names
    )
    if args.smoke:
        verify_sampler(r, masses, args.seed)
    report = {
        "configuration": vars(args) | {"image": str(args.image), "architectures": names},
        "image": {
            "width": width,
            "height": height,
            "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        },
        "device": {"adapter": r.device.info.adapter_name, "backend": r.device.info.api_name},
        "versions": package_versions(),
        "cdf_precompute_seconds": cdf_precompute_seconds,
        "reference_entropy": reference_entropy,
        "architectures": {},
    }
    for name in names:
        report["architectures"][name] = run_architecture(
            r, name, masses, args, output, reference_entropy
        )
    (output / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {output / 'results.json'}")

    if not args.no_plot:
        subprocess.run([sys.executable, str(HERE / "plot.py"), str(output)], check=True)


if __name__ == "__main__":
    main()
