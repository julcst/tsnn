#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy>=0.42,<0.43", "numpy", "tqdm"]
# ///

import argparse
import math
from pathlib import Path
import slangpy as spy
from layout import InferenceWeights, compute_layout, static_layouts_source

import numpy as np
from tqdm import trange

# ─── Layout constants (must match MLP.slang) ───────────────────────────────

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

# Weight gradients are accumulated via fp16 hardware coopvec atomics
# (CoopVecComponentType::Float16 in MLP.slang); pre-scaling by LOSS_SCALE
# before backward keeps them out of fp16's subnormal range. Adam's
# invLossScale divides it back out during the update. See benchmark.py.
LOSS_SCALE = 1024.0

BATCH_SIZE = 1 << 14  # 16 384 pixels per step
STEPS = 5_000
LR = 1e-3
DISPLAY_EVERY = 200
RESOLUTION = 512


# ─── Encoding buffer sizes (MLP sizes are queried from the device) ─────────────


def _align4(x: int) -> int:
    return (x + 3) & ~3


ENC_PARAM_COUNT = HASH_LEVELS * HASH_TABLE_SIZE * HASH_FEATURES

ENC_PARAM_BYTES = _align4(ENC_PARAM_COUNT * 2)
ENC_GRAD_BYTES = _align4(ENC_PARAM_COUNT * 4)
ENC_MOMENT_BYTES = _align4(ENC_PARAM_COUNT * 4)

# Reset/optimize dispatch thread count, scaled to the (much larger) encoding
# param count instead of a small fixed size -- mirrors SlangNRC.cpp's
# kOptimizeDispatchThreads.
DISPATCH_THREAD_COUNT = (min(1 << 19, ENC_PARAM_COUNT) + 255) // 256 * 256


# ─── Target image helpers ─────────────────────────────────────────────────────

# Shared across examples/image_learn and examples/nde.
DEFAULT_IMAGE = Path(__file__).parent.parent / "einstein.png"


def load_image(path: str, res: int) -> np.ndarray:
    bmp = spy.Bitmap(str(path)).convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.float32,
        srgb_gamma=False,
    )
    return np.array(bmp.resample(res, res), copy=False)


# ─── Buffer utilities ─────────────────────────────────────────────────────────


def create_buffer(device, size_bytes: int, is_rw: bool = False) -> spy.Buffer:
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.copy_source
    if is_rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=size_bytes, usage=usage)


def create_texture(
    device, width: int, height: int, data: np.ndarray = None
) -> spy.Texture:
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
    tex = create_texture(device, W, H, data=rgba)
    return tex


# ─── Kernel helpers ───────────────────────────────────────────────────────────


class ImageLearner:
    def __init__(self, device, target: np.ndarray):
        self.device = device
        self.H, self.W = target.shape[:2]

        # Load and upload target image
        self.target_tex = upload_texture(device, target)

        # GPU buffers
        offsets, self.param_count = compute_layout(
            device, INPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE
        )
        self.layout_offsets = offsets
        param_bytes = self.param_count * 2
        moment_bytes = self.param_count * 4
        self.params = create_buffer(device, param_bytes, is_rw=True)
        self.inference_weights = InferenceWeights(
            device, self.params, offsets, INPUT_SIZE, HIDDEN_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE
        )
        layout_source = static_layouts_source(
            offsets, self.param_count,
            self.inference_weights.offsets, self.inference_weights.param_count,
        )
        self.params_master = create_buffer(device, moment_bytes, is_rw=True)  # float32, same size as moments
        self.param_grads = create_buffer(device, param_bytes, is_rw=True)
        self.moments1 = create_buffer(device, moment_bytes, is_rw=True)
        self.moments2 = create_buffer(device, moment_bytes, is_rw=True)
        self.enc_params = create_buffer(device, ENC_PARAM_BYTES, is_rw=True)
        self.enc_grads = create_buffer(device, ENC_GRAD_BYTES, is_rw=True)
        self.enc_moments1 = create_buffer(device, ENC_MOMENT_BYTES, is_rw=True)
        self.enc_moments2 = create_buffer(device, ENC_MOMENT_BYTES, is_rw=True)
        self.output_tex = create_texture(device, self.W, self.H)

        # Load compute kernels
        self.reset_kernel = device.create_compute_kernel(
            device.load_program(
                module_name="Optimize.cs.slang",
                entry_point_names=["resetMain"],
                additional_source=layout_source,
            )
        )
        self.train_kernel = device.create_compute_kernel(
            device.load_program(
                module_name="Train.cs.slang",
                entry_point_names=["trainMain"],
                additional_source=layout_source,
            )
        )
        self.optimize_kernel = device.create_compute_kernel(
            device.load_program(
                module_name="Optimize.cs.slang",
                entry_point_names=["optimizeMain"],
                additional_source=layout_source,
            )
        )
        self.infer_kernel = device.create_compute_kernel(
            device.load_program(
                module_name="Infer.cs.slang",
                entry_point_names=["inferMain"],
                additional_source=layout_source,
            )
        )

        # Reset parameters (weights + hash grid). Adam moments are just zeroed.
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
        encoder = device.create_command_encoder()
        for buf in (self.moments1, self.moments2, self.enc_moments1, self.enc_moments2):
            encoder.clear_buffer(buf)
        device.submit_command_buffer(encoder.finish())

    def zero_gradients(self):
        encoder = self.device.create_command_encoder()
        encoder.clear_buffer(self.param_grads)
        encoder.clear_buffer(self.enc_grads)
        self.device.submit_command_buffer(encoder.finish())

    def train_step(self, step: int):
        self.zero_gradients()
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
        )

    def optimize_step(self, step: int, lr: float):
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
        )

    def infer(self) -> np.ndarray:
        encoder = self.device.create_command_encoder()
        self.inference_weights.convert(encoder)
        self.infer_kernel.dispatch(
            thread_count=[self.W, self.H, 1],
            frameDim=[self.W, self.H],
            vars={
                "gOutput": self.output_tex,
                "gParams": self.inference_weights.params,
                "gEncodingParams": self.enc_params,
            },
            command_encoder=encoder,
        )
        self.device.submit_command_buffer(encoder.finish())
        return self.output_tex.to_numpy().view(np.float32)[..., :3]

    def mse(self, pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean((pred - target) ** 2))


# ─── Training loop ────────────────────────────────────────────────────────────


def train(target: np.ndarray, device, steps: int, lr: float):
    learner = ImageLearner(device, target)
    print(f"MLP storage: {learner.param_count:,} half slots; hash grid: {ENC_PARAM_COUNT:,} parameters")
    print(f"Training for {steps} steps, batch size {BATCH_SIZE}")

    for step in (t := trange(1, steps + 1)):
        learner.train_step(step)
        learner.optimize_step(step, lr)

        if step % DISPLAY_EVERY == 0 or step == 1:
            pred = learner.infer()
            mse = learner.mse(pred, target)
            psnr = -10 * math.log10(max(mse, 1e-10))
            t.set_postfix({"PSNR (dB)": f"{psnr:.2f}"})

    return learner


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--res", type=int, default=RESOLUTION)
    parser.add_argument("--out", default="result.exr")
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
    learner = train(target, device, args.steps, args.lr)
    pred = learner.infer()
    # Save using slangpy.Bitmap
    bmp = spy.Bitmap(pred, spy.Bitmap.PixelFormat.rgb)
    spy.tev.show(bmp, name="prediction")
    bmp.write(Path(args.out))
    print(f"Result saved → {args.out}")


if __name__ == "__main__":
    main()
