#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy", "numpy", "tqdm"]
# ///
"""Benchmark harness for the TSNN (Slang/slangpy) image-learning example.
Trains the same hash-grid + MLP configuration as benchmark_tcnn.py and dumps a
JSON result file with per-kernel GPU device time (training, optimizer,
inference) and PSNR, for direct comparison.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import slangpy as spy
from tqdm import trange

# Shared across examples/image_learn and examples/nde.
DEFAULT_IMAGE = Path(__file__).parent.parent / "einstein.png"


def load_image(path: str, res: int) -> np.ndarray:
    bmp = spy.Bitmap(str(path)).convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.float32,
        srgb_gamma=False,
    )
    return np.array(bmp.resample(res, res), copy=False)


# ─── Layout constants (must match Network.slang / MLP.slang) ───────────────
# Kept identical to benchmark_tcnn.py's tcnn config so both sides train the
# same encoding + network shape with the same optimizer hyperparameters.

HASH_LEVELS = 16
HASH_FEATURES = 2
HASH_LOG2_TABLE = 19
HASH_TABLE_SIZE = 1 << HASH_LOG2_TABLE
HASH_BASE_RES = 16
HASH_SCALE = 1.5

HIDDEN_SIZE = 64
INPUT_SIZE = HASH_LEVELS * HASH_FEATURES  # 32
OUTPUT_SIZE = 3
HIDDEN_LAYERS = 4
TRANSITIONS = HIDDEN_LAYERS + 1

BATCH_SIZE = 1 << 14  # 16 384 pixels per step
STEPS = 5_000
LR = 1e-3
EVAL_EVERY = 100
RESOLUTION = 512

# Weight gradients are accumulated via fp16 hardware coopvec atomics
# (CoopVecComponentType::Float16 in MLP.slang). A relative-L2-luminance grad,
# divided by BATCH_SIZE and backpropagated through 4 hidden layers, lands well
# inside fp16's subnormal range (< 6e-5) and loses most of its precision on
# the atomic add. Pre-scaling by LOSS_SCALE before backward keeps gradients in
# fp16's normal range; Adam's invLossScale (Optimizers/Adam.slang) divides it
# back out during the update -- same trick as NISTracer's mLossScale.
LOSS_SCALE = 1024.0


def _align4(x: int) -> int:
    return (x + 3) & ~3


def layer_input(i):
    return INPUT_SIZE if i == 0 else HIDDEN_SIZE


def layer_output(i):
    return OUTPUT_SIZE if i == TRANSITIONS - 1 else HIDDEN_SIZE


def param_element_count() -> int:
    return sum(
        layer_input(i) * layer_output(i) + layer_output(i) for i in range(TRANSITIONS)
    )


def encoding_param_element_count() -> int:
    return HASH_LEVELS * HASH_TABLE_SIZE * HASH_FEATURES


PARAM_COUNT = param_element_count()
ENC_PARAM_COUNT = encoding_param_element_count()

PARAM_BYTES = _align4(PARAM_COUNT * 2)  # float16
ENC_PARAM_BYTES = _align4(ENC_PARAM_COUNT * 2)
MOMENT_BYTES = PARAM_COUNT * 4  # float32
ENC_GRAD_BYTES = _align4(ENC_PARAM_COUNT * 4)
ENC_MOMENT_BYTES = _align4(ENC_PARAM_COUNT * 4)

# Reset/optimize dispatch thread count, scaled to the (much larger) encoding
# param count instead of a small fixed size -- mirrors SlangNRC.cpp's
# kOptimizeDispatchThreads, which is what lets the production NRC pass beat
# tcnn. A fixed small dispatch here left every thread grid-striding through
# ~2048 elements each, making the Adam sweep the per-step bottleneck.
DISPATCH_THREAD_COUNT = (min(1 << 19, ENC_PARAM_COUNT) + 255) // 256 * 256


def create_buffer(device, size_bytes: int, is_rw: bool = False) -> spy.Buffer:
    usage = spy.BufferUsage.shader_resource
    if is_rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=size_bytes, usage=usage)


def create_texture(device, width: int, height: int, data: np.ndarray = None) -> spy.Texture:
    return device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        data=data,
    )


def upload_texture(device, data: np.ndarray) -> spy.Texture:
    H, W = data.shape[:2]
    rgba = np.concatenate([data, np.ones((H, W, 1), dtype=np.float32)], axis=-1)
    return create_texture(device, W, H, data=rgba)


class ImageLearner:
    def __init__(self, device, target: np.ndarray):
        self.device = device
        self.H, self.W = target.shape[:2]

        self.target_tex = upload_texture(device, target)

        self.params = create_buffer(device, PARAM_BYTES, is_rw=True)
        self.params_master = create_buffer(device, MOMENT_BYTES, is_rw=True)  # float32, same size as moments
        self.param_grads = create_buffer(device, PARAM_BYTES, is_rw=True)
        self.moments1 = create_buffer(device, MOMENT_BYTES, is_rw=True)
        self.moments2 = create_buffer(device, MOMENT_BYTES, is_rw=True)
        self.enc_params = create_buffer(device, ENC_PARAM_BYTES, is_rw=True)
        self.enc_grads = create_buffer(device, ENC_GRAD_BYTES, is_rw=True)
        self.enc_moments1 = create_buffer(device, ENC_MOMENT_BYTES, is_rw=True)
        self.enc_moments2 = create_buffer(device, ENC_MOMENT_BYTES, is_rw=True)
        self.output_tex = create_texture(device, self.W, self.H)

        self.reset_kernel = device.create_compute_kernel(
            device.load_program(module_name="Optimize.cs.slang", entry_point_names=["resetMain"])
        )
        self.train_kernel = device.create_compute_kernel(
            device.load_program(module_name="Train.cs.slang", entry_point_names=["trainMain"])
        )
        self.optimize_kernel = device.create_compute_kernel(
            device.load_program(module_name="Optimize.cs.slang", entry_point_names=["optimizeMain"])
        )
        self.infer_kernel = device.create_compute_kernel(
            device.load_program(module_name="Infer.cs.slang", entry_point_names=["inferMain"])
        )

        self.reset()

    def reset(self):
        dispatch_count = DISPATCH_THREAD_COUNT
        self.reset_kernel.dispatch(
            thread_count=[dispatch_count, 1, 1],
            vars={
                "gParams": self.params,
                "gParamsMaster": self.params_master,
                "gEncodingParams": self.enc_params,
                "CB": {
                    "gLearningRate": LR,
                    "gCurrentStep": 1.0,
                    "gDispatchThreadCount": dispatch_count,
                },
            },
        )
        encoder = self.device.create_command_encoder()
        for buf in (self.moments1, self.moments2, self.enc_moments1, self.enc_moments2):
            encoder.clear_buffer(buf)
        self.device.submit_command_buffer(encoder.finish())
        self.device.wait_for_idle()

    def zero_gradients(self, encoder=None):
        owns = encoder is None
        encoder = encoder or self.device.create_command_encoder()
        encoder.clear_buffer(self.param_grads)
        encoder.clear_buffer(self.enc_grads)
        if owns:
            self.device.submit_command_buffer(encoder.finish())

    def train_step(self, step: int, encoder=None):
        self.zero_gradients(encoder)
        self.train_kernel.dispatch(
            thread_count=[BATCH_SIZE, 1, 1],
            vars={
                "gTarget": self.target_tex,
                "gParams": self.params,
                "gParamGrads": self.param_grads,
                "gEncodingParams": self.enc_params,
                "gEncodingParamGrads": self.enc_grads,
                "CB": {
                    "gFrameDim": [self.W, self.H],
                    "gBatchSize": BATCH_SIZE,
                    "gCurrentStep": step,
                    "gFrameIndex": step,
                    "gLossScale": LOSS_SCALE,
                },
            },
            command_encoder=encoder,
        )

    def optimize_step(self, step: int, lr: float, encoder=None):
        dispatch_count = DISPATCH_THREAD_COUNT
        self.optimize_kernel.dispatch(
            thread_count=[dispatch_count, 1, 1],
            vars={
                "gParams": self.params,
                "gParamsMaster": self.params_master,
                "gParamGrads": self.param_grads,
                "gMoments1": self.moments1,
                "gMoments2": self.moments2,
                "gEncodingParams": self.enc_params,
                "gEncodingParamGrads": self.enc_grads,
                "gEncodingMoments1": self.enc_moments1,
                "gEncodingMoments2": self.enc_moments2,
                "CB": {
                    "gLearningRate": lr,
                    "gCurrentStep": float(step),
                    "gDispatchThreadCount": dispatch_count,
                    "gLossScale": LOSS_SCALE,
                },
            },
            command_encoder=encoder,
        )

    def dispatch_infer(self, encoder=None):
        self.infer_kernel.dispatch(
            thread_count=[self.W, self.H, 1],
            frameDim=[self.W, self.H],
            vars={"gOutput": self.output_tex, "gParams": self.params, "gEncodingParams": self.enc_params},
            command_encoder=encoder,
        )

    def read_output(self) -> np.ndarray:
        return self.output_tex.to_numpy().view(np.float32)[..., :3]

    def infer(self) -> np.ndarray:
        self.dispatch_infer()
        return self.read_output()

    def mse(self, pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))


def run_inference_benchmark(learner: ImageLearner, device, iters: int, warmup: int) -> dict:
    """Dedicated inference-only GPU throughput for the fused inference kernel,
    run after training completes: this example trains once then infers from
    the fixed result, unlike an online setup (e.g. NRC) where training and
    inference are interleaved every frame -- so inference is benchmarked as
    its own back-to-back loop, not mixed into the training loop's timing."""
    for _ in range(warmup):
        learner.dispatch_infer()
    device.wait_for_idle()

    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=2 * iters)
    device.wait_for_idle()

    encoder = device.create_command_encoder()
    for i in range(iters):
        encoder.write_timestamp(query_pool, 2 * i)
        learner.dispatch_infer(encoder=encoder)
        encoder.write_timestamp(query_pool, 2 * i + 1)
    device.submit_command_buffer(encoder.finish())
    device.wait_for_idle()

    ts = query_pool.get_timestamp_results(0, 2 * iters)
    gpu_time = sum(ts[2 * i + 1] - ts[2 * i] for i in range(iters))

    pixels = learner.W * learner.H * iters
    return {
        "iters": iters,
        "gpu_time_s": gpu_time,
        "mpixels_per_s": pixels / gpu_time / 1e6,
    }


def run(target: np.ndarray, device, steps: int, lr: float, eval_every: int, warmup: int) -> dict:
    learner = ImageLearner(device, target)

    # Slang pipelines are lazily compiled to native GPU ISA on first dispatch
    # (load_program only gets to SPIRV) -- run a few throwaway steps through
    # every kernel so that one-time driver compilation doesn't land inside the
    # timed loop below, then reset weights so training starts from scratch.
    # Mirrors the WARMUP convention used by scripts/nis_profile.py etc.
    for step in range(1, warmup + 1):
        learner.train_step(step)
        learner.optimize_step(step, lr)
    learner.infer()  # to_numpy() readback already blocks until the GPU is done
    learner.reset()

    # GPU device time for each kernel separately, bracketed with its own
    # timestamp pair per step: train_step is the fused forward+backward
    # kernel (gradient computation), optimize_step is the fused Adam kernel.
    # Both exclude wait_for_idle() and the periodic infer()/PSNR eval;
    # inference itself is benchmarked separately after training, below.
    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=4 * eval_every)

    history = []
    device.wait_for_idle()
    train_gpu_time = 0.0
    optimize_gpu_time = 0.0

    t = trange(1, steps + 1, desc="tsnn")
    step = 1
    while step <= steps:
        block_end = min(step + eval_every - 1, steps)
        query_pool.reset()

        encoder = device.create_command_encoder()
        for i, s in enumerate(range(step, block_end + 1)):
            encoder.write_timestamp(query_pool, 4 * i)
            learner.train_step(s, encoder=encoder)
            encoder.write_timestamp(query_pool, 4 * i + 1)
            encoder.write_timestamp(query_pool, 4 * i + 2)
            learner.optimize_step(s, lr, encoder=encoder)
            encoder.write_timestamp(query_pool, 4 * i + 3)
        device.submit_command_buffer(encoder.finish())
        device.wait_for_idle()

        n = block_end - step + 1
        ts = query_pool.get_timestamp_results(0, 4 * n)
        for i in range(n):
            train_gpu_time += ts[4 * i + 1] - ts[4 * i]
            optimize_gpu_time += ts[4 * i + 3] - ts[4 * i + 2]
        t.update(n)

        pred = learner.infer()  # to_numpy() readback already blocks until the GPU is done
        mse = learner.mse(pred, target)
        psnr = -10 * math.log10(max(mse, 1e-10))
        history.append({
            "step": block_end,
            "train_gpu_time_s": train_gpu_time,
            "optimize_gpu_time_s": optimize_gpu_time,
            "psnr": psnr,
        })
        t.set_postfix({"PSNR (dB)": f"{psnr:.2f}"})

        step = block_end + 1
    t.close()

    inference = run_inference_benchmark(learner, device, iters=200, warmup=warmup)
    return {
        "backend": "tsnn",
        "gpu": device.info.adapter_name,
        "param_count": PARAM_COUNT,
        "enc_param_count": ENC_PARAM_COUNT,
        "config": {
            "hash_levels": HASH_LEVELS,
            "hash_features": HASH_FEATURES,
            "hash_log2_table": HASH_LOG2_TABLE,
            "hash_base_res": HASH_BASE_RES,
            "hash_scale": HASH_SCALE,
            "hidden_size": HIDDEN_SIZE,
            "hidden_layers": HIDDEN_LAYERS,
            "activation": "LeakyReLU",
            "batch_size": BATCH_SIZE,
            "steps": steps,
            "lr": lr,
            "resolution": target.shape[0],
        },
        "history": history,
        "train_gpu_time_s": train_gpu_time,
        "optimize_gpu_time_s": optimize_gpu_time,
        "final_psnr": history[-1]["psnr"] if history else None,
        "inference": inference,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--res", type=int, default=RESOLUTION)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--warmup", type=int, default=10, help="Throwaway steps to force pipeline compile before timing")
    parser.add_argument("--out", default="bench_tsnn.json")
    args = parser.parse_args()

    target = load_image(args.image, args.res)
    print(f"Target: {args.image} ({target.shape[1]}x{target.shape[0]})")

    device = spy.create_device(
        include_paths=[
            Path(__file__).parent.absolute(),
            Path(__file__).parent.parent.parent.absolute(),
            Path(__file__).parent.parent.parent.absolute() / "TSNN",
        ]
    )
    print(f"GPU: {device.info.adapter_name}")
    print(f"Parameters: {PARAM_COUNT:,} (MLP) + {ENC_PARAM_COUNT:,} (hash grid)")

    result = run(target, device, args.steps, args.lr, args.eval_every, args.warmup)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[train]    gpu_time={result['train_gpu_time_s']:.3f}s")
    print(f"[optimize] gpu_time={result['optimize_gpu_time_s']:.3f}s")
    print(f"[psnr]     final={result['final_psnr']:.2f}dB")
    inf = result["inference"]
    print(f"[infer]    gpu_time={inf['gpu_time_s']:.3f}s ({inf['iters']} full-image passes, "
          f"{inf['mpixels_per_s']:.1f} Mpixels/s)")
    print(f"Result saved -> {args.out}")


if __name__ == "__main__":
    main()
