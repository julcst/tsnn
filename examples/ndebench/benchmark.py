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
REGISTRY = {
    "TMM": {
        "init": "init_TMM",
        "train": "trainNLL_TMM",
        "eval": "inferEval_TMM",
        "sample": "inferSample_TMM",
        "metadata": "metadata_TMM",
    }
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


def sample_stream(masses: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("sample count must be positive")
    cdf = np.ascontiguousarray(np.cumsum(masses.ravel(), dtype=np.float64))
    cdf[-1] = 1.0
    if not np.isclose(cdf[-1], 1.0) or np.any(np.diff(cdf) < 0):
        raise ValueError("invalid normalized CDF")
    rng = np.random.default_rng(seed)
    chosen = np.searchsorted(cdf, rng.random(count), side="right")
    chosen = np.minimum(chosen, masses.size - 1)
    row, col = np.divmod(chosen, masses.shape[1])
    jitter = rng.random((count, 2))
    points = np.empty((count, 2), dtype=np.float32)
    points[:, 0] = (col + jitter[:, 0]) / masses.shape[1]
    points[:, 1] = (row + jitter[:, 1]) / masses.shape[0]
    # The cast must not convert a valid coordinate to precisely one.
    return np.minimum(points, np.nextafter(np.float32(1), np.float32(0)))


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


def batches(total: int, batch_size: int):
    for offset in range(0, total, batch_size):
        yield offset, min(batch_size, total - offset)


def validate() -> None:
    # Non-square orientation also catches x/y transposition in the sampler.
    masses = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=np.float64)
    masses /= masses.sum()
    points = sample_stream(masses, 4096, 9)
    assert points.dtype == np.float32 and np.all((points >= 0) & (points < 1))
    assert np.array_equal(points, sample_stream(masses, 4096, 9))
    cdf = np.cumsum(masses.ravel())
    cdf[-1] = 1.0
    assert cdf[-1] == 1.0 and np.searchsorted(cdf, 0.0, side="right") == 1
    grid = texel_centers(3, 2).reshape(2, 3, 2)
    assert np.allclose(grid[0, :, 1], 0.25) and np.allclose(grid[:, 0, 0], 1 / 6)
    density = masses * 6
    assert np.isclose(density.sum() / 6, 1.0)
    assert np.count_nonzero(masses) == 3  # zero mass is excluded from NLL reduction
    assert list(batches(10, 4)) == [(0, 4), (4, 4), (8, 2)]
    assert list(batches(3, 8)) == [(0, 3)]


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
    grid: object
    eval_out: object
    sample_out: object
    elements: int
    padded: int
    dispatch_threads: int
    grid_count: int
    sample_count: int

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

    def warm(
        self,
        arch: str,
        batch: int,
        lr: float,
        loss_scale: float,
        clip: float,
        count: int,
    ) -> None:
        n = min(batch, self.sample_count)
        for step in range(1, count + 1):
            enc = self.device.create_command_encoder()
            enc.clear_buffer(self.grads)
            self.kernels[REGISTRY[arch]["train"]].dispatch(
                thread_count=[n, 1, 1],
                vars={
                    "gParams": self.params,
                    "gParamGrads": self.grads,
                    "gSamples": self.samples,
                    "TrainCB": {"gOffset": 0, "gCount": n, "gWeight": loss_scale / n},
                },
                command_encoder=enc,
            )
            self.kernels["optimize"].dispatch(
                thread_count=[self.dispatch_threads, 1, 1],
                vars=self.optimize_vars(step, lr, loss_scale, clip),
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

    def optimize_vars(self, step: int, lr: float, loss_scale: float, clip: float) -> dict:
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
                "gParamElementCount": self.padded,
                "gDispatchThreadCount": self.dispatch_threads,
            },
        }


def make_runner(samples: np.ndarray, grid: np.ndarray) -> Runner:
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
        entry: load(module, entry)
        for module, entry in [
            ("Metadata.slang", "metadata_TMM"),
            ("Optimize.slang", "init_TMM"),
            ("Optimize.slang", "optimize"),
            ("TrainNLL.slang", "trainNLL_TMM"),
            ("InferEval.slang", "inferEval_TMM"),
            ("InferSample.slang", "inferSample_TMM"),
        ]
    }
    meta = buffer(device, np.zeros(1, np.uint32), 4)
    kernels["metadata_TMM"].dispatch(thread_count=[1, 1, 1], vars={"gMetadata": meta})
    device.wait()
    elements = int(np.frombuffer(meta.to_numpy(), dtype=np.uint32)[0])
    if elements <= 0:
        raise RuntimeError("architecture metadata returned no parameters")
    padded = aligned4(elements)
    fp16_bytes = padded * 2
    fp32_bytes = padded * 4
    return Runner(
        device,
        kernels,
        buffer(device, None, fp16_bytes),
        buffer(device, None, fp32_bytes),
        buffer(device, None, fp16_bytes),
        buffer(device, None, fp32_bytes),
        buffer(device, None, fp32_bytes),
        buffer(device, np.ascontiguousarray(samples), samples.nbytes, rw=False),
        buffer(device, np.ascontiguousarray(grid), grid.nbytes, rw=False),
        buffer(device, None, grid.shape[0] * 4),
        buffer(device, None, max(samples.shape[0], grid.shape[0]) * 8),
        elements,
        padded,
        256 * 8,
        grid.shape[0],
        samples.shape[0],
    )


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
        args.batch_size,
        args.learning_rate,
        args.loss_scale,
        args.gradient_clip,
        args.warmup_count,
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
    total_train = total_clear = total_forward = total_opt = 0.0
    step = 0
    total_consumed = 0
    # `all_batches` slices the fixed, CPU-generated sample pool; training steps
    # cycle through it rather than growing 1:1 with --samples, since
    # sample_stream (numpy CDF search) is far slower than a GPU train step and
    # would otherwise become the dominant cost as steps scale up (see the
    # "data generation" diagnostic printed at the end of main()).
    all_batches = list(batches(args.samples, args.batch_size))
    total_steps = args.steps
    with tqdm(total=total_steps, desc=f"{arch} train", unit="step") as pbar:
        while step < total_steps:
            group_size = min(args.evaluation_interval, total_steps - step)
            group = [all_batches[(step + i) % len(all_batches)] for i in range(group_size)]
            q = r.device.create_query_pool(spy.QueryType.timestamp, len(group) * 4)
            enc = r.device.create_command_encoder()
            for i, (offset, count) in enumerate(group):
                step += 1
                total_consumed += count
                base = i * 4
                enc.write_timestamp(q, base)
                enc.clear_buffer(r.grads)
                enc.write_timestamp(q, base + 1)
                r.kernels[REGISTRY[arch]["train"]].dispatch(
                    thread_count=[count, 1, 1],
                    vars={
                        "gParams": r.params,
                        "gParamGrads": r.grads,
                        "gSamples": r.samples,
                        "TrainCB": {
                            "gOffset": offset,
                            "gCount": count,
                            "gWeight": args.loss_scale / count,
                        },
                    },
                    command_encoder=enc,
                )
                enc.write_timestamp(q, base + 2)
                r.kernels["optimize"].dispatch(
                    thread_count=[r.dispatch_threads, 1, 1],
                    vars=r.optimize_vars(
                        step, args.learning_rate, args.loss_scale, args.gradient_clip
                    ),
                    command_encoder=enc,
                )
                enc.write_timestamp(q, base + 3)
            r.device.submit_command_buffer(enc.finish())
            r.device.wait()
            stamps = (
                np.asarray(q.get_results(0, len(group) * 4), dtype=np.uint64).reshape(-1, 4)
                / r.device.info.timestamp_frequency
            )
            total_clear += float(np.sum(stamps[:, 1] - stamps[:, 0]))
            total_forward += float(np.sum(stamps[:, 2] - stamps[:, 1]))
            total_opt += float(np.sum(stamps[:, 3] - stamps[:, 2]))
            total_train += float(np.sum(stamps[:, 3] - stamps[:, 0]))
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
        "metadata": {"parameter_elements_fp16": r.elements, "padded_parameter_elements": r.padded},
        "checkpoints": rows,
        "timing": {
            "mean_training_ms_per_update": total_train / step * 1000,
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
    p.add_argument("--samples", type=int, default=1 << 20)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--architectures", default="TMM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-2)
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
            args.samples,
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
        args.samples, args.batch_size, args.evaluation_interval = 32, 16, 1
        args.timing_repetitions, args.timing_workload = 2, 16
        args.steps = 2
    masses, width, height = load_image(args.image)
    if args.smoke:
        masses = smoke_reduce(masses)
        height, width = masses.shape
    gen_started = time.perf_counter()
    stream = sample_stream(masses, args.samples, args.seed)
    data_generation_seconds = time.perf_counter() - gen_started
    grid = texel_centers(width, height)
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    with np.errstate(divide="ignore"):
        reference_logpdf = np.where(masses > 0, np.log(masses * width * height), -np.inf)
    np.save(output / "reference_masses.npy", masses)
    np.save(output / "reference_logpdf.npy", reference_logpdf)
    r = make_runner(stream, grid)
    report = {
        "configuration": vars(args) | {"image": str(args.image), "architectures": names},
        "image": {
            "width": width,
            "height": height,
            "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        },
        "device": {"adapter": r.device.info.adapter_name, "backend": r.device.info.api_name},
        "versions": package_versions(),
        "data_generation_seconds": data_generation_seconds,
        "architectures": {},
    }
    for name in names:
        report["architectures"][name] = run_architecture(r, name, masses, args, output)
    (output / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {output / 'results.json'}")

    total_gpu_train = sum(
        a["timing"]["cumulative_training_seconds"] for a in report["architectures"].values()
    )
    print(
        f"data generation (CPU): {data_generation_seconds * 1000:.1f} ms for {args.samples} "
        f"samples ({args.samples / max(data_generation_seconds, 1e-9):.0f} samples/s) vs "
        f"{total_gpu_train * 1000:.1f} ms cumulative GPU training time across {len(names)} "
        f"architecture(s)"
    )
    if data_generation_seconds > total_gpu_train:
        print(
            "  -> CPU sample_stream() dominates GPU training time; --steps cycles the fixed "
            "sample pool instead of regenerating data, so raising --steps does not add to "
            "this cost. Raise --samples only if more unique data is actually needed."
        )


if __name__ == "__main__":
    main()
