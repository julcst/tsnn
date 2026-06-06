#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy", "numpy", "tqdm"]
# ///
"""
Neural Density Estimation — main training script.

Trains an 8-layer coupling normalizing flow to fit a 2-D probability
distribution.  The flow is configured at the command line (--flow / --prior /
--rotations); those choices are passed to the Slang compiler as preprocessor
#defines, so no shader editing is needed to try different variants.

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
  python main.py                              # affine + normal + rotations (default)
  python main.py --flow rqs --prior uniform   # bounded spline flow, clean sampling
  python main.py --flow rqs --no-rotations    # spline flow, axis-aligned
  python main.py --image galaxy.png --steps 10000 --lr 3e-4
"""

import argparse
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

# Conditioner-MLP activation → ACTIVATION #define in Network.slang.
ACTIVATIONS = {
    "relu": 0,
    "leakyrelu": 1,
    "swish": 2,
    "silu": 2,
    "gelu": 3,
    "elu": 4,
    "tanh": 5,
    "mish": 6,
}

# Stability / mixed-precision knobs (see Train.cs / Optimize.cs).
LOSS_SCALE = 128.0  # gradient pre-scale to keep float16 accumulation off the floor
GRAD_CLIP = 1.0  # per-element gradient clip, in TRUE (unscaled) units
WEIGHT_DECAY = 1e-2  # AdamW decoupled weight decay (ignored by Adam)

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


def build_config(
    flow: str,
    prior: str,
    rotations: bool,
    activation: str = "leakyrelu",
    optimizer: str = "adam",
    debug: bool = False,
) -> dict:
    """Resolve a flow/prior/rotation/activation/optimizer choice into Slang
    #defines + buffer sizes.

    The #defines are passed to the Slang compiler (see main); the buffer sizes
    mirror Network.slang's constexpr parameter layout for the chosen flow.
    """
    use_rqs = flow == "rqs"
    use_uniform = prior == "uniform"
    params_per_layer = (3 * NUM_BINS - 1) if use_rqs else 2
    block = _mlp_byte_size(1, HIDDEN, COND_DEPTH, params_per_layer)
    param_bytes = block * NUM_FLOW_LAYERS
    elem = param_bytes // 2
    opt_label = "adamw" if optimizer == "adamw" else "adam"
    return {
        "defines": {
            "USE_RQS": str(int(use_rqs)),
            "USE_ROTATIONS": str(int(rotations)),
            "USE_UNIFORM_PRIOR": str(int(use_uniform)),
            "ACTIVATION": str(ACTIVATIONS[activation]),
            "DEBUG_COUNTERS": str(int(debug)),
        },
        "debug": debug,
        "param_bytes": param_bytes,
        "moment_bytes": elem * 4,  # float32 master + 2 Adam moments share this size
        "param_elems": elem,
        "label": (
            f"{flow} + {prior}"
            + (" + rotations" if rotations else "")
            + f" + {activation} + {opt_label}"
        ),
    }


# ─── Training hyper-parameters ───────────────────────────────────────────────

BATCH_SIZE = 1 << 14  # 16 384 data samples per step
N_DATA = 1 << 20  # pre-generated pool: 1 M samples
STEPS = 24_000
LR = 3e-4  # constant LR (no schedule). The float32 master + loss-scale-corrected
# clip make this stable for the whole run; lr up to 1e-3 also converges without
# late-stage divergence. rqs+uniform reaches JSD≈0.0067 (near the noise floor) by
# ~24k steps at this LR.

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
    a = a / a.max()  # normalise for better contrast
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
    def __init__(
        self,
        device,
        samples: np.ndarray,
        cfg: dict,
        loss_scale: float = LOSS_SCALE,
        grad_clip: float = GRAD_CLIP,
        weight_decay: float = WEIGHT_DECAY,
    ):
        self.device = device
        self.cfg = cfg
        self.n_data = len(samples)
        self.loss_scale = loss_scale
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.debug = cfg.get("debug", False)

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
        param_bytes = cfg["param_bytes"]
        moment_bytes = cfg["moment_bytes"]
        self.params = make_buf(device, param_bytes, rw=True)
        self.params_master = make_buf(device, moment_bytes, rw=True)  # float32
        self.param_grads = make_buf(device, param_bytes, rw=True)
        self.moments1 = make_buf(device, moment_bytes, rw=True)
        self.moments2 = make_buf(device, moment_bytes, rw=True)

        # Debug under/overflow tally: 4 × uint32 (only used when DEBUG_COUNTERS).
        self.debug_counters = make_buf(device, 16, rw=True) if self.debug else None

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
                    "gLossScale": self.loss_scale,
                    "gGradClip": self.grad_clip,
                    "gWeightDecay": self.weight_decay,
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
                    "gLossScale": self.loss_scale,
                },
            },
        )

    def optimize_step(self, step: int, lr: float):
        n = 256 * 8
        vars = {
            "gParams": self.params,
            "gParamsMaster": self.params_master,
            "gParamGrads": self.param_grads,
            "gMoments1": self.moments1,
            "gMoments2": self.moments2,
            "CB": {
                "gLearningRate": lr,
                "gCurrentStep": float(step),
                "gDispatchThreadCount": n,
                "gLossScale": self.loss_scale,
                "gGradClip": self.grad_clip,
                "gWeightDecay": self.weight_decay,
            },
        }
        if self.debug:
            vars["gDebugCounters"] = self.debug_counters
        self.optimize_k.dispatch(thread_count=[n, 1, 1], vars=vars)

    def read_debug_counters(self) -> dict:
        """Read and zero the under/overflow tally. Returns fractions of elements."""
        raw = np.frombuffer(self.debug_counters.to_numpy(), dtype=np.uint32)
        over, clip, zero, tot = (int(x) for x in raw[:4])
        tot = max(tot, 1)
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.debug_counters)
        self.device.submit_command_buffer(enc.finish())
        return {
            "overflow": over / tot,
            "clipped": clip / tot,
            "zero": zero / tot,
        }

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
    cfg = learner.cfg
    print(
        f"Flow: {cfg['label']}, {cfg['param_elems']:,} float16 params "
        f"({cfg['param_bytes'] / 1024:.1f} KB)"
    )
    print(f"Training {steps} steps, batch {BATCH_SIZE}, lr={lr}")

    best_jsd = float("inf")
    for step in (bar := trange(1, steps + 1)):
        # Constant learning rate (no warmup/decay schedule).
        learner.train_step(step)
        learner.optimize_step(step, lr)

        if step % DISPLAY_EVERY == 0 or step == 1:
            density = learner.eval_density()
            jsd = learner.compute_jsd(density)
            best_jsd = min(best_jsd, jsd)
            post = {"JSD": f"{jsd:.4f}", "best": f"{best_jsd:.4f}"}
            if learner.debug:
                d = learner.read_debug_counters()
                # over = float16 accumulator overflowed (lower loss-scale);
                # zero = gradient underflowed/dead (raise loss-scale);
                # clip = clip saturating (raise grad-clip or lower loss-scale).
                post["of/uf/clip"] = (
                    f"{d['overflow']:.0e}/{d['zero']:.2f}/{d['clipped']:.0e}"
                )
            bar.set_postfix(post)
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
    # Flow configuration — compiled into the shaders via Slang #defines.
    parser.add_argument(
        "--flow", choices=["affine", "rqs"], default="rqs", help="Coupling transform"
    )
    # Note: Normal prior does not work well here because the domain is bounded and the normal distribution only has 47% of its mass inside [-1, 1]²
    parser.add_argument(
        "--prior",
        choices=["normal", "uniform"],
        default="uniform",
        help="Base distribution",
    )
    parser.add_argument(
        "--rotations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interleave Rotation2D mixing layers between coupling blocks",
    )
    parser.add_argument(
        "--activation",
        choices=sorted(ACTIVATIONS.keys()),
        default="leakyrelu",
        help="Conditioner-MLP activation (try swish/gelu/mish for stability)",
    )
    parser.add_argument(
        "--optimizer", choices=["adam", "adamw"], default="adam", help="Optimizer"
    )
    parser.add_argument("--loss-scale", type=float, default=LOSS_SCALE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Count gradient under/overflows each eval (to tune --loss-scale)",
    )
    args = parser.parse_args()

    rotations = args.rotations
    if args.prior == "uniform" and rotations:
        print(
            "note: rotations push latents outside the uniform support and can lead to artifacts with uniform prior"
        )
    cfg = build_config(
        args.flow, args.prior, rotations, args.activation, args.optimizer, args.debug
    )

    np.random.seed(0)
    if args.image:
        print(f"Sampling {args.n_data:,} points from {args.image}")
        samples = image_to_samples(args.image, args.n_data)
    else:
        print(f"Using 5-component Gaussian mixture ({args.n_data:,} samples)")
        samples = gaussian_mixture_samples(args.n_data)

    # Use the low-level device so we can pass the flow #defines straight to the
    # Slang compiler (these reach the imported Network module, unlike per-program
    # additional_source).  Must add slangpy.SHADER_PATH to the include paths.
    here = Path(__file__).parent.absolute()
    compiler_options = spy.SlangCompilerOptions(
        {
            "include_paths": [
                here,
                here.parent.parent,
                here.parent.parent / "TSNN",
                spy.SHADER_PATH,
            ],
            "defines": cfg["defines"],
        }
    )
    device = spy.Device(compiler_options=compiler_options)

    # Adam and AdamW share one kernel now; the only difference is the decoupled
    # weight-decay term, gated on gWeightDecay. Plain Adam = zero decay.
    weight_decay = args.weight_decay if args.optimizer == "adamw" else 0.0
    learner = NDELearner(
        device,
        samples,
        cfg,
        loss_scale=args.loss_scale,
        grad_clip=args.grad_clip,
        weight_decay=weight_decay,
    )
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
