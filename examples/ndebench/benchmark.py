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
ARCHITECTURES = ("TMM", "HGGrid")

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
    dispatch_threads: int
    grid_count: int
    width: int
    height: int
    batch_size: int

    def reset(self, arch: str) -> None:
        enc = self.device.create_command_encoder()
        for item in (self.params, self.master, self.grads, self.m1, self.m2):
            enc.clear_buffer(item)
        self.kernels[REGISTRY[arch]["init"]].dispatch(
            thread_count=[self.dispatch_threads, 1, 1],
            vars={
                "gParamsInit": self.params,
                "gParamsMasterInit": self.master,
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
            vars={"gParams": self.params, "gSamples": self.grid, "gLogPDFs": self.eval_out},
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
                    "gStreamOffset": stream_offset,
                    "gSeed": seed,
                },
            },
            command_encoder=encoder,
        )

    def warm(
        self,
        arch: str,
        lr: float,
        loss_scale: float,
        clip: float,
        count: int,
        seed: int,
    ) -> None:
        n = self.batch_size
        for step in range(1, count + 1):
            enc = self.device.create_command_encoder()
            self.generate(n, (step - 1) * n, seed, enc)
            enc.clear_buffer(self.grads)
            self.kernels[REGISTRY[arch]["train"]].dispatch(
                thread_count=[n, 1, 1],
                vars={
                    "gParams": self.params,
                    "gParamGrads": self.grads,
                    "gSamples": self.samples,
                    "TrainCB": {"gCount": n, "gWeight": loss_scale / n},
                },
                command_encoder=enc,
            )
            self.kernels["optimize"].dispatch(
                thread_count=[self.dispatch_threads, 1, 1],
                vars=self.optimize_vars(arch, step, lr, loss_scale, clip),
                command_encoder=enc,
            )
            self.device.submit_command_buffer(enc.finish())
        self.eval_grid(arch)
        self.kernels[REGISTRY[arch]["sample"]].dispatch(
            thread_count=[n, 1, 1],
            vars={"gParams": self.params, "gSamples": self.sample_out, "SampleCB": {"gSeed": 1}},
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

    # Each architecture owns its own parameter-element count; buffers are
    # sized to the largest one so every architecture's real weights fit,
    # while gParamElementCount (in optimize_vars) stays per-architecture so
    # the Adam sweep never walks past a smaller architecture's own tail.
    meta = buffer(device, np.zeros(1, np.uint32), 4)
    elements_by_arch: dict[str, int] = {}
    padded_by_arch: dict[str, int] = {}
    for arch in architectures:
        kernels[REGISTRY[arch]["metadata"]].dispatch(
            thread_count=[1, 1, 1], vars={"gMetadata": meta}
        )
        device.wait()
        elements = int(np.frombuffer(meta.to_numpy(), dtype=np.uint32)[0])
        if elements <= 0:
            raise RuntimeError(f"architecture {arch!r} metadata returned no parameters")
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
        256 * 8,
        grid.shape[0],
        width,
        height,
        batch_size,
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
    for i in range(5):
        vars_ = (
            {"gParams": r.params, "gSamples": input_buf, "gLogPDFs": output_buf}
            if kind == "eval"
            else {"gParams": r.params, "gSamples": output_buf, "SampleCB": {"gSeed": seed + i}}
        )
        r.kernels[key].dispatch(thread_count=[workload, 1, 1], vars=vars_)
    r.device.wait()
    q = r.device.create_query_pool(spy.QueryType.timestamp, repeats * 2)
    for i in range(repeats):
        vars_ = (
            {"gParams": r.params, "gSamples": input_buf, "gLogPDFs": output_buf}
            if kind == "eval"
            else {"gParams": r.params, "gSamples": output_buf, "SampleCB": {"gSeed": seed + i}}
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


def run_architecture(
    r: Runner, arch: str, masses: np.ndarray, args: argparse.Namespace, output: Path
) -> dict:
    r.warm(
        arch,
        args.learning_rate,
        args.loss_scale,
        args.gradient_clip,
        args.warmup_count,
        args.seed,
    )
    initial, grid_time = r.eval_grid(arch)
    h, w = masses.shape

    def checkpoint(step: int, consumed: int, timing: float, eval_seconds: float) -> dict:
        logpdf, seconds = r.eval_grid(arch)
        nll = -float(
            np.sum(masses[masses > 0] * logpdf.reshape(h, w)[masses > 0], dtype=np.float64)
        )
        return {
            "step": step,
            "samples_consumed": consumed,
            "nll": nll,
            "cumulative_training_gpu_seconds": timing,
            "grid_evaluation_seconds": seconds + eval_seconds,
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
        }
    )
    total_generate = total_train = total_clear = total_forward = total_opt = 0.0
    step = 0
    total_consumed = 0
    total_steps = args.steps
    with tqdm(total=total_steps, desc=f"{arch} train", unit="step") as pbar:
        while step < total_steps:
            group_size = min(args.evaluation_interval, total_steps - step)
            q = r.device.create_query_pool(spy.QueryType.timestamp, group_size * 5)
            enc = r.device.create_command_encoder()
            for i in range(group_size):
                base = i * 5
                enc.write_timestamp(q, base)
                r.generate(r.batch_size, step * r.batch_size, args.seed, enc)
                enc.write_timestamp(q, base + 1)
                enc.clear_buffer(r.grads)
                enc.write_timestamp(q, base + 2)
                step += 1
                total_consumed = step * r.batch_size
                r.kernels[REGISTRY[arch]["train"]].dispatch(
                    thread_count=[r.batch_size, 1, 1],
                    vars={
                        "gParams": r.params,
                        "gParamGrads": r.grads,
                        "gSamples": r.samples,
                        "TrainCB": {
                            "gCount": r.batch_size,
                            "gWeight": args.loss_scale / r.batch_size,
                        },
                    },
                    command_encoder=enc,
                )
                enc.write_timestamp(q, base + 3)
                r.kernels["optimize"].dispatch(
                    thread_count=[r.dispatch_threads, 1, 1],
                    vars=r.optimize_vars(
                        arch, step, args.learning_rate, args.loss_scale, args.gradient_clip
                    ),
                    command_encoder=enc,
                )
                enc.write_timestamp(q, base + 4)
            r.device.submit_command_buffer(enc.finish())
            r.device.wait()
            stamps = (
                np.asarray(q.get_results(0, group_size * 5), dtype=np.uint64).reshape(-1, 5)
                / r.device.info.timestamp_frequency
            )
            total_generate += float(np.sum(stamps[:, 1] - stamps[:, 0]))
            total_clear += float(np.sum(stamps[:, 2] - stamps[:, 1]))
            total_forward += float(np.sum(stamps[:, 3] - stamps[:, 2]))
            total_opt += float(np.sum(stamps[:, 4] - stamps[:, 3]))
            total_train += float(np.sum(stamps[:, 4] - stamps[:, 0]))
            row, final = checkpoint(step, total_consumed, total_train, 0.0)
            rows.append(row)
            pbar.update(group_size)
            pbar.set_postfix(nll=f"{row['nll']:.4f}")
    final_image = final.reshape(h, w)
    np.save(output / f"{arch}_final_logpdf.npy", final_image)
    display = np.maximum(final_image, -12.0)
    display = (255 * (display - display.min()) / max(float(np.ptp(display)), 1e-12)).astype(
        np.uint8
    )
    Image.fromarray(display, mode="L").save(output / f"{arch}_final_logpdf.png")
    return {
        "metadata": {
            "parameter_elements_fp16": r.elements_by_arch[arch],
            "padded_parameter_elements": r.padded_by_arch[arch],
        },
        "checkpoints": rows,
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
    p.add_argument("--steps", type=int, default=8192)
    p.add_argument("--evaluation-interval", type=int, default=64)
    p.add_argument("--warmup-count", type=int, default=5)
    p.add_argument("--timing-repetitions", type=int, default=100)
    p.add_argument("--timing-workload", type=int, default=4096)
    p.add_argument("--output-directory", type=Path, default=HERE / "output")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    validate() if args.validate else None
    if args.validate and not args.smoke:
        print("CPU validation passed")
        return
    names = [x.strip() for x in args.architectures.split(",") if x.strip()]
    unknown = sorted(set(names) - REGISTRY.keys())
    if unknown:
        p.error(f"unknown architectures {unknown}; supported: {', '.join(REGISTRY)}")
    if (
        min(
            args.batch_size,
            args.steps,
            args.evaluation_interval,
            args.timing_repetitions,
            args.timing_workload,
        )
        < 1
    ):
        p.error("counts must be positive")
    if args.smoke:
        args.batch_size, args.evaluation_interval = 16, 1
        args.timing_repetitions, args.timing_workload = 2, 16
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
        "architectures": {},
    }
    for name in names:
        report["architectures"][name] = run_architecture(r, name, masses, args, output)
    (output / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {output / 'results.json'}")


if __name__ == "__main__":
    main()
