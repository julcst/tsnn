#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy", "numpy", "tqdm"]
# ///
"""
Neural Density Estimation — main training script.

Trains an 8-layer coupling normalizing flow to fit a 2-D probability
distribution.  The coupling transform is affine by default, or rational-
quadratic splines (set USE_RQS below AND #define USE_RQS 1 in Network.slang).

Everything lives on the normalized domain [-DOMAIN, DOMAIN]² (DOMAIN = 1); the
data generators scale into it, so there is no per-dataset tail bound to manage.

Three GPU kernels mirror the image_learn pattern:
  1. trainMain    – compute NLL gradients for a random mini-batch
  2. optimizeMain – Adam step (float32 master weights, float16 mirror)
  3. evalMain     – render the learned density to a texture

Evaluation uses Jensen-Shannon divergence between the model's predicted PDF
and an analytic histogram of the training data.  Density fields are written as
float .exr (via spy.Bitmap) for easy inspection.

Usage:
  python main.py                        # default: synthetic 2-D Gaussian mixture
  python main.py --image galaxy.png     # fit image-as-PDF
  python main.py --steps 10000 --lr 3e-4
"""

import argparse
import math
from pathlib import Path

import numpy as np
import slangpy as spy
from tqdm import trange

# ─── Architecture constants (must match Network.slang) ───────────────────────

HIDDEN = 64
COND_DEPTH = 2  # hidden layers in conditioner MLP
NUM_FLOW_LAYERS = 8
NUM_BINS = 16  # RQS bins (USE_RQS only)
DOMAIN = 1.0  # normalized data/latent domain [-1, 1]

# Must match `#define USE_RQS` in Network.slang. Affine → 2 params/layer (s,t);
# RQS → 3K-1 packed spline params/layer.
USE_RQS = True
PARAMS_PER_LAYER = (3 * NUM_BINS - 1) if USE_RQS else 2


# ─── Parameter-count mirror of Network.slang constexprs ──────────────────────


def _align4(x: int) -> int:
    return (x + 3) & ~3


def _mlp_byte_size(input_dim: int, hidden: int, depth: int, output_dim: int) -> int:
    """Byte size of one MLP block (float16 weights), mirroring MLP.slang __init."""
    byte_off = 0
    for l in range(depth + 1):
        in_size = input_dim if l == 0 else hidden
        out_size = output_dim if l == depth else hidden
        byte_off = _align4(byte_off + 2 * in_size * out_size)  # weights (half)
        byte_off = _align4(byte_off + 2 * out_size)  # biases  (half)
    return byte_off


MLP_BLOCK_BYTES = _mlp_byte_size(1, HIDDEN, COND_DEPTH, PARAMS_PER_LAYER)
PARAM_BYTE_COUNT = MLP_BLOCK_BYTES * NUM_FLOW_LAYERS
PARAM_ELEM_COUNT = PARAM_BYTE_COUNT // 2  # float16 elements
MOMENT_BYTE_COUNT = PARAM_ELEM_COUNT * 4  # float32 moments


# ─── Training hyper-parameters ───────────────────────────────────────────────

BATCH_SIZE = 1 << 14  # 16 384 data samples per step
N_DATA = 1 << 20  # pre-generated pool: 1 M samples
STEPS = 36_000
LR = 3e-5   # float32 master makes Adam usable; >5e-5 lets weights grow into the
            # imprecise float16-mirror range and destabilises late training
LR_WARMUP = 500

DISPLAY_EVERY = 3_000
GRID_RES = 256  # density evaluation grid


# ─── 2-D data generation (all outputs scaled to the normalized domain) ────────


def image_to_samples(img_path: str, n: int, bound: float = DOMAIN) -> np.ndarray:
    """Sample 2-D points from a greyscale image used as an unnormalised PDF."""
    bmp = np.array(spy.Bitmap(str(img_path))).astype(np.float32)
    img = bmp.mean(axis=2) if bmp.ndim == 3 else bmp  # luminance
    pdf = img / img.sum()

    H, W = pdf.shape
    p_row = pdf.sum(axis=1)
    cdf_row = np.cumsum(p_row)

    rows = np.searchsorted(cdf_row, np.random.rand(n))
    rows = np.clip(rows, 0, H - 1)

    cols = np.zeros(n, dtype=np.int32)
    for r in np.unique(rows):
        mask = rows == r
        p_col = pdf[r]
        s = p_col.sum()
        if s < 1e-12:
            continue
        cdf_col = np.cumsum(p_col / s)
        cols[mask] = np.searchsorted(cdf_col, np.random.rand(mask.sum()))
    cols = np.clip(cols, 0, W - 1)

    jitter = np.random.rand(n, 2) - 0.5
    x = ((cols + 0.5 + jitter[:, 0]) / W) * 2.0 * bound - bound
    y = ((rows + 0.5 + jitter[:, 1]) / H) * 2.0 * bound - bound
    return np.stack([x, y], axis=1).astype(np.float32)


def gaussian_mixture_samples(n: int, bound: float = DOMAIN) -> np.ndarray:
    """5-component Gaussian mixture inside [-bound, bound] — canonical NF benchmark."""
    centres = bound * np.array(
        [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
        dtype=np.float32,
    )
    scales = bound * np.array([0.16, 0.12, 0.12, 0.12, 0.12], dtype=np.float32)

    idx = np.random.randint(0, len(centres), size=n)
    pts = centres[idx] + np.random.randn(n, 2).astype(np.float32) * scales[idx, None]
    return np.clip(pts, -bound + 1e-3, bound - 1e-3)


def analytic_histogram(samples: np.ndarray, grid_res: int, bound: float) -> np.ndarray:
    """2-D histogram of samples, normalised to sum to 1.
    Output shape (grid_res, grid_res) with [row, col] = [y-bin, x-bin].
    """
    edges = np.linspace(-bound, bound, grid_res + 1)
    hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges, edges])
    hist = hist.T.astype(np.float64)  # transpose → [y-bin, x-bin]
    return (hist / hist.sum()).astype(np.float32)


# ─── I/O helpers ──────────────────────────────────────────────────────────────


def save_field(arr: np.ndarray, path) -> None:
    """Write a 2-D float field as an RGBA float .exr (rows flipped so y is up)."""
    a = np.ascontiguousarray(arr[::-1].astype(np.float32))
    rgba = np.stack([a, a, a, np.ones_like(a)], axis=-1)
    spy.Bitmap(rgba, spy.Bitmap.PixelFormat.rgba).write(Path(path))


def make_buf(device, size_bytes: int, rw: bool = False) -> spy.Buffer:
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=size_bytes, usage=usage)


def make_tex(device, w: int, h: int) -> spy.Texture:
    return device.create_texture(
        width=w,
        height=h,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )


# ─── Main learner class ───────────────────────────────────────────────────────


class NDELearner:
    def __init__(self, device, samples: np.ndarray):
        self.device = device
        self.n_data = len(samples)

        # Upload training samples (float32 x,y pairs)
        samples_f32 = samples.astype(np.float32).flatten()
        self.data_buf = device.create_buffer(
            size=int(samples_f32.nbytes),
            usage=spy.BufferUsage.shader_resource,
            data=samples_f32,
        )

        # Analytic histogram for PDF distance evaluation
        self.analytic_hist = analytic_histogram(samples, GRID_RES, DOMAIN)

        # Parameter buffers: float16 mirror (GPU compute) + float32 master (Adam)
        self.params = make_buf(device, PARAM_BYTE_COUNT, rw=True)
        self.params_master = make_buf(device, MOMENT_BYTE_COUNT, rw=True)  # float32
        self.param_grads = make_buf(device, PARAM_BYTE_COUNT, rw=True)
        self.moments1 = make_buf(device, MOMENT_BYTE_COUNT, rw=True)
        self.moments2 = make_buf(device, MOMENT_BYTE_COUNT, rw=True)

        # Output texture for density visualisation
        self.density_tex = make_tex(device, GRID_RES, GRID_RES)

        def load(module, entry):
            return device.create_compute_kernel(
                device.load_program(module_name=module, entry_point_names=[entry])
            )

        self.reset_k = load("Optimize.cs.slang", "resetMain")
        self.train_k = load("Train.cs.slang", "trainMain")
        self.optimize_k = load("Optimize.cs.slang", "optimizeMain")
        self.eval_k = load("Infer.cs.slang", "evalMain")
        self.sample_k = load("Infer.cs.slang", "sampleMain")

        self._reset()

    def _reset(self):
        n = 256 * 8
        self.reset_k.dispatch(
            thread_count=[n, 1, 1],
            vars={
                "gParams": self.params,
                "gParamsMaster": self.params_master,
                "gParamGrads": self.param_grads,
                "gMoments1": self.moments1,
                "gMoments2": self.moments2,
                "CB": {
                    "gLearningRate": LR,
                    "gCurrentStep": 1.0,
                    "gDispatchThreadCount": n,
                },
            },
        )

    def zero_grads(self):
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.param_grads)
        self.device.submit_command_buffer(enc.finish())

    def train_step(self, step: int):
        self.zero_grads()
        self.train_k.dispatch(
            thread_count=[BATCH_SIZE, 1, 1],
            vars={
                "gSamples": self.data_buf,
                "gParams": self.params,
                "gParamGrads": self.param_grads,
                "CB": {
                    "gNumSamples": self.n_data,
                    "gBatchSize": BATCH_SIZE,
                    "gCurrentStep": step,
                },
            },
        )

    def optimize_step(self, step: int, lr: float):
        n = 256 * 8
        self.optimize_k.dispatch(
            thread_count=[n, 1, 1],
            vars={
                "gParams": self.params,
                "gParamsMaster": self.params_master,
                "gParamGrads": self.param_grads,
                "gMoments1": self.moments1,
                "gMoments2": self.moments2,
                "CB": {
                    "gLearningRate": lr,
                    "gCurrentStep": float(step),
                    "gDispatchThreadCount": n,
                },
            },
        )

    def eval_density(self) -> np.ndarray:
        """Render the learned density to a [GRID_RES, GRID_RES] numpy array."""
        self.eval_k.dispatch(
            thread_count=[GRID_RES, GRID_RES, 1],
            frameDim=[GRID_RES, GRID_RES],
            vars={"gParams": self.params, "gDensityTex": self.density_tex},
        )
        arr = self.density_tex.to_numpy().view(np.float32)[..., 0]  # R channel
        return arr

    def compute_jsd(self, density: np.ndarray) -> float:
        """Jensen-Shannon divergence in [0, log 2] between model PDF and analytic histogram.

        JSD(p, q) = 0.5*KL(p||m) + 0.5*KL(q||m), m = (p+q)/2.
        Symmetric, bounded, and well-defined even when one distribution has zero bins.
        """
        p = self.analytic_hist.astype(np.float64)
        q = density.astype(np.float64)
        q /= max(q.sum(), 1e-10)
        m = 0.5 * (p + q)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        return 0.5 * (kl(p, m) + kl(q, m))

    def generate_samples(self, n: int) -> np.ndarray:
        """Generate n samples from the learned distribution."""
        out_buf = make_buf(self.device, n * 8, rw=True)
        self.sample_k.dispatch(
            thread_count=[n, 1, 1],
            vars={
                "gParams": self.params,
                "gSampleOut": out_buf,
                "SampleCB": {"gNumSamples": n, "gSeed": 42},
            },
        )
        raw = np.frombuffer(out_buf.to_numpy(), dtype=np.float32)
        return raw.reshape(n, 2)


# ─── Training loop ────────────────────────────────────────────────────────────


def train(learner: NDELearner, steps: int, lr: float):
    flow = "RQS" if USE_RQS else "affine"
    print(f"Flow: {flow}, {PARAM_ELEM_COUNT:,} float16 params ({PARAM_BYTE_COUNT / 1024:.1f} KB)")
    print(f"Training {steps} steps, batch {BATCH_SIZE}, lr={lr}")

    best_jsd = float("inf")
    for step in (bar := trange(1, steps + 1)):
        # Linear warmup then cosine decay
        if step <= LR_WARMUP:
            current_lr = lr * step / LR_WARMUP
        else:
            t = (step - LR_WARMUP) / max(steps - LR_WARMUP, 1)
            current_lr = lr * (0.5 * (1.0 + math.cos(math.pi * t)))

        learner.train_step(step)
        learner.optimize_step(step, current_lr)

        if step % DISPLAY_EVERY == 0 or step == 1:
            density = learner.eval_density()
            jsd = learner.compute_jsd(density)
            best_jsd = min(best_jsd, jsd)
            bar.set_postfix({"JSD": f"{jsd:.4f}", "best": f"{best_jsd:.4f}"})
            save_field(density, f"model_step{step:06d}.exr")

    return learner


# ─── Entry point ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Greyscale image to fit as PDF")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--out", default="density.exr", help="Output density .exr")
    parser.add_argument(
        "--out-samples", default=None, help="Save generated samples (.npy)"
    )
    args = parser.parse_args()

    np.random.seed(0)
    if args.image:
        print(f"Sampling {args.n_data:,} points from {args.image}")
        samples = image_to_samples(args.image, args.n_data)
    else:
        print(f"Using 5-component Gaussian mixture ({args.n_data:,} samples)")
        samples = gaussian_mixture_samples(args.n_data)

    device = spy.create_device(
        include_paths=[
            Path(__file__).parent.absolute(),
            Path(__file__).parent.parent.parent.absolute(),
            Path(__file__).parent.parent.parent.absolute() / "TSNN",
        ]
    )

    learner = NDELearner(device, samples)
    save_field(learner.analytic_hist, "target_hist.exr")
    print("Target histogram saved → target_hist.exr")
    train(learner, args.steps, args.lr)

    density = learner.eval_density()
    save_field(density, args.out)
    print(f"Density saved → {args.out}")

    if args.out_samples:
        pts = learner.generate_samples(10_000)
        np.save(args.out_samples, pts)
        print(f"Samples saved → {args.out_samples}")


if __name__ == "__main__":
    main()
