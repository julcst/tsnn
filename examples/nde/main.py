#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy", "numpy", "tqdm"]
# ///
"""
Neural Density Estimation — two training modes for a 2-D normalizing flow.

MLE mode (--mode mle, default):
  Maximize log-likelihood of data samples drawn from the target distribution.
  Works with any 2-D distribution: a greyscale image or the built-in Gaussian
  mixture.  Samples are generated once on the CPU and stored in a GPU buffer.

KL mode (--mode kl):
  Minimize KL(p_target ‖ q_θ) via importance-weighted model samples, as in
  Mueller et al. "Neural Importance Sampling" (2019), §3.1.  Each step:
    1. Draw x ~ q_θ by sampling the prior and running the inverse flow on GPU.
    2. Look up unnormalized target f(x) from an image luminance buffer.
    3. Backpropagate with gradient weight  -f(x) / q_θ(x).
  Requires --image.  The global scale of f cancels in Adam.

Both modes produce density .exr fields and Jensen-Shannon divergence logs.

Usage:
  python main.py                                             # MLE, Gaussian mixture
  python main.py --image ../einstein.png                     # MLE, image target
  python main.py --image ../einstein.png --mode kl           # KL, image target
  python main.py --image ../einstein.png --mode kl --flow rqs --prior uniform
  python main.py --image ../einstein.png --steps 10000 --lr 3e-4
"""

import argparse
from pathlib import Path

import numpy as np
import slangpy as spy
from tqdm import trange

# ─── Architecture constants (must match Network.slang constexprs exactly) ──────
#
# GPU timeout (NVRM XID 109 / CTX SWITCH TIMEOUT) root cause:
#   RQS.slang uses [ForceUnroll] on all loops sized by kNumBins, so kNumBins is a
#   hard multiplier on per-thread shader complexity.  The bwd_diff(forward) kernel
#   runs one full forward+backward pass per sample; with 16 384 samples per step
#   the total work scales as:  kNumBins × kCondDepth × NUM_FLOW_LAYERS.
#
#   kNumBins=64, kCondDepth=3, 8 layers  →  ~8× original  →  XID 109 timeout
#   kNumBins=16, kCondDepth=2, 4 layers  →  1× original   →  safe
#   kNumBins=32, kCondDepth=2, 2 layers  →  ~1×            →  safe (Mueller et al. literal)
#
# Mueller et al. use 2 coupling layers with 32-bin piecewise-quadratic and a deep
# U-net conditioner.  Without the U-net / one-blob encoding, 4 layers + 16-bin RQS
# is the closest equivalent that stays within GPU timeout bounds.
#
# IF you increase any of these, update Network.slang to match; mismatches silently
# over/under-size the parameter buffer and produce wrong results or GPU faults.

HIDDEN = 64          # kHidden   in Network.slang
COND_DEPTH = 2       # kCondDepth in Network.slang
NUM_FLOW_LAYERS = 4  # number of kMLP* blocks in Network.slang  (currently 4: kMLP0..kMLP3)
NUM_BINS = 32        # kNumBins  in Network.slang
DOMAIN = 1.0

ACTIVATIONS = {
    "relu": 0, "leakyrelu": 1, "swish": 2, "silu": 2,
    "gelu": 3, "elu": 4, "tanh": 5, "mish": 6,
}

LOSS_SCALE = 128.0
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-2

# ─── Parameter layout (mirrors Network.slang constexprs) ──────────────────────


def _align4(x: int) -> int:
    return (x + 3) & ~3


def _mlp_byte_size(input_dim: int, hidden: int, depth: int, output_dim: int) -> int:
    byte_off = 0
    for l in range(depth + 1):
        in_size = input_dim if l == 0 else hidden
        out_size = output_dim if l == depth else hidden
        byte_off = _align4(byte_off + 2 * in_size * out_size)
        byte_off = _align4(byte_off + 2 * out_size)
    return byte_off


def build_config(
    flow: str,
    prior: str,
    rotations: bool,
    activation: str = "leakyrelu",
    optimizer: str = "adam",
    debug: bool = False,
) -> dict:
    use_rqs = flow == "rqs"
    use_uniform = prior == "uniform"
    params_per_layer = (3 * NUM_BINS - 1) if use_rqs else 2
    block = _mlp_byte_size(1, HIDDEN, COND_DEPTH, params_per_layer)
    param_bytes = block * NUM_FLOW_LAYERS
    elem = param_bytes // 2
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
        "moment_bytes": elem * 4,
        "param_elems": elem,
        "label": (
            f"{flow} + {prior}"
            + (" + rotations" if rotations else "")
            + f" + {activation} + {'adamw' if optimizer == 'adamw' else 'adam'}"
        ),
    }


# ─── Training hyper-parameters ────────────────────────────────────────────────

BATCH_SIZE = 1 << 14
N_DATA = 1 << 20
STEPS = 24_000
LR = 3e-4
DISPLAY_EVERY = 3_000
GRID_RES = 256

# ─── Data / target generation ─────────────────────────────────────────────────


def load_luminance(img_path: str) -> np.ndarray:
    """Load a greyscale luminance map from an image file.  Returns float32 [H, W]."""
    bmp = np.array(spy.Bitmap(str(img_path))).astype(np.float32)
    return bmp.mean(axis=2) if bmp.ndim == 3 else bmp


def image_to_samples(img_path: str, n: int, bound: float = DOMAIN) -> np.ndarray:
    """Sample 2-D points from a greyscale image used as an unnormalised PDF."""
    img = load_luminance(img_path)
    pdf = img / img.sum()
    H, W = pdf.shape
    p_row = pdf.sum(axis=1)
    cdf_row = np.cumsum(p_row)
    rows = np.clip(np.searchsorted(cdf_row, np.random.rand(n)), 0, H - 1)
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
    # Row 0 is the top of the image.  save_field(arr[::-1]) places y=+1 at the
    # top of the output, so top-of-image must map to y=+1 (not y=-1).
    y = bound - ((rows + 0.5 + jitter[:, 1]) / H) * 2.0 * bound
    return np.stack([x, y], axis=1).astype(np.float32)


def gaussian_mixture_samples(n: int, bound: float = DOMAIN) -> np.ndarray:
    centres = bound * np.array(
        [[0.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]],
        dtype=np.float32,
    )
    scales = bound * np.array([0.16, 0.12, 0.12, 0.12, 0.12], dtype=np.float32)
    idx = np.random.randint(0, len(centres), size=n)
    pts = centres[idx] + np.random.randn(n, 2).astype(np.float32) * scales[idx, None]
    return np.clip(pts, -bound + 1e-3, bound - 1e-3)


def analytic_histogram(samples: np.ndarray, grid_res: int, bound: float) -> np.ndarray:
    edges = np.linspace(-bound, bound, grid_res + 1)
    hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges, edges])
    hist = hist.T.astype(np.float64)
    return (hist / hist.sum()).astype(np.float32)


# ─── I/O helpers ──────────────────────────────────────────────────────────────


def save_field(arr: np.ndarray, path) -> None:
    """Write a 2-D float field as an RGBA float .exr (rows flipped so y is up)."""
    a = np.ascontiguousarray(arr[::-1].astype(np.float32))
    a = a / a.max() if a.max() > 0 else a
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


# ─── Learner ──────────────────────────────────────────────────────────────────


class NDELearner:
    """
    Two training modes:

    mle — maximize log-likelihood of pre-sampled data.
          samples is the CPU training set; also used for JSD evaluation.

    kl  — minimize KL(target ‖ model) via importance-weighted model samples
          (Mueller et al. 2019 §3.1).  target_lum is the float32 [H, W]
          luminance array; samples is used only for JSD evaluation histogram.
    """

    def __init__(
        self,
        device,
        cfg: dict,
        samples: np.ndarray,
        mode: str = "mle",
        target_lum: np.ndarray = None,
        loss_scale: float = LOSS_SCALE,
        grad_clip: float = GRAD_CLIP,
        weight_decay: float = WEIGHT_DECAY,
    ):
        assert mode in ("mle", "kl"), f"Unknown mode {mode!r}"
        if mode == "kl":
            assert target_lum is not None, "KL mode requires target_lum"

        self.device = device
        self.cfg = cfg
        self.mode = mode
        self.loss_scale = loss_scale
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.debug = cfg.get("debug", False)

        # Analytic histogram used for JSD evaluation in both modes.
        self.analytic_hist = analytic_histogram(samples, GRID_RES, DOMAIN)

        # MLE: upload training samples (float32 x,y pairs).
        if mode == "mle":
            self.n_data = len(samples)
            samples_f32 = samples.astype(np.float32).flatten()
            self.data_buf = device.create_buffer(
                size=int(samples_f32.nbytes),
                usage=spy.BufferUsage.shader_resource,
                data=samples_f32,
            )

        # KL: upload target luminance as a flat float32 buffer.
        # Row 0 = top of image = world-space y = -DOMAIN, matching evalMain.
        if mode == "kl":
            lum = target_lum.astype(np.float32)
            self.tex_height, self.tex_width = lum.shape
            lum_flat = lum.flatten()
            self.target_buf = device.create_buffer(
                size=int(lum_flat.nbytes),
                usage=spy.BufferUsage.shader_resource,
                data=lum_flat,
            )

        # Parameter buffers: float16 mirror (GPU compute) + float32 master (Adam).
        param_bytes = cfg["param_bytes"]
        moment_bytes = cfg["moment_bytes"]
        self.params = make_buf(device, param_bytes, rw=True)
        self.params_master = make_buf(device, moment_bytes, rw=True)
        self.param_grads = make_buf(device, param_bytes, rw=True)
        self.moments1 = make_buf(device, moment_bytes, rw=True)
        self.moments2 = make_buf(device, moment_bytes, rw=True)

        self.debug_counters = make_buf(device, 16, rw=True) if self.debug else None
        self.density_tex = make_tex(device, GRID_RES, GRID_RES)

        def load(module, entry):
            return device.create_compute_kernel(
                device.load_program(module_name=module, entry_point_names=[entry])
            )

        self.reset_k = load("Optimize.cs.slang", "resetMain")
        self.train_k = load("Train.cs.slang", "trainMain")
        self.train_kl_k = load("Train.cs.slang", "trainKLMain")
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
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.moments1)
        enc.clear_buffer(self.moments2)
        self.device.submit_command_buffer(enc.finish())

    def zero_grads(self):
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.param_grads)
        self.device.submit_command_buffer(enc.finish())

    def train_step(self, step: int):
        self.zero_grads()
        if self.mode == "mle":
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
        else:  # kl
            self.train_kl_k.dispatch(
                thread_count=[BATCH_SIZE, 1, 1],
                vars={
                    "gTargetBuf": self.target_buf,
                    "gParams": self.params,
                    "gParamGrads": self.param_grads,
                    "KLCB": {
                        "gKLBatchSize": BATCH_SIZE,
                        "gKLCurrentStep": step,
                        "gKLLossScale": self.loss_scale,
                        "gTexWidth": self.tex_width,
                        "gTexHeight": self.tex_height,
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
        raw = np.frombuffer(self.debug_counters.to_numpy(), dtype=np.uint32)
        over, clip, zero, tot = (int(x) for x in raw[:4])
        tot = max(tot, 1)
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.debug_counters)
        self.device.submit_command_buffer(enc.finish())
        return {"overflow": over / tot, "clipped": clip / tot, "zero": zero / tot}

    def eval_density(self) -> np.ndarray:
        self.eval_k.dispatch(
            thread_count=[GRID_RES, GRID_RES, 1],
            frameDim=[GRID_RES, GRID_RES],
            vars={"gParams": self.params, "gDensityTex": self.density_tex},
        )
        arr = self.density_tex.to_numpy().view(np.float32)[..., 0]
        return arr

    def compute_jsd(self, density: np.ndarray) -> float:
        """Jensen-Shannon divergence in [0, log 2] between model PDF and analytic histogram."""
        p = self.analytic_hist.astype(np.float64)
        q = density.astype(np.float64)
        q /= max(q.sum(), 1e-10)
        m = 0.5 * (p + q)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

        return 0.5 * (kl(p, m) + kl(q, m))

    def generate_samples(self, n: int) -> np.ndarray:
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


def train(learner: NDELearner, steps: int, lr: float, prefix: str = ""):
    cfg = learner.cfg
    print(
        f"Mode: {learner.mode.upper()}  |  Flow: {cfg['label']}"
        f"  |  {cfg['param_elems']:,} float16 params ({cfg['param_bytes'] / 1024:.1f} KB)"
    )
    print(f"Training {steps} steps, batch {BATCH_SIZE}, lr={lr}")

    best_jsd = float("inf")
    for step in (bar := trange(1, steps + 1)):
        learner.train_step(step)
        learner.optimize_step(step, lr)

        if step % DISPLAY_EVERY == 0 or step == 1:
            density = learner.eval_density()
            jsd = learner.compute_jsd(density)
            best_jsd = min(best_jsd, jsd)
            post = {"JSD": f"{jsd:.4f}", "best": f"{best_jsd:.4f}"}
            if learner.debug:
                d = learner.read_debug_counters()
                post["of/uf/clip"] = (
                    f"{d['overflow']:.0e}/{d['zero']:.2f}/{d['clipped']:.0e}"
                )
            bar.set_postfix(post)
            tag = f"{prefix}_step{step:06d}" if prefix else f"model_step{step:06d}"
            save_field(density, f"{tag}.exr")

    return learner


# ─── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Greyscale image to fit as PDF")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--out", default="density.exr", help="Output density .exr")
    parser.add_argument("--out-samples", default=None, help="Save generated samples (.npy)")
    parser.add_argument(
        "--mode",
        choices=["mle", "kl"],
        default="mle",
        help=(
            "mle: maximize log-likelihood of data samples (standard NLL). "
            "kl:  minimize KL(target‖model) via importance-weighted model samples "
            "(Mueller et al. 2019).  Requires --image."
        ),
    )
    parser.add_argument(
        "--flow", choices=["affine", "rqs"], default="rqs", help="Coupling transform"
    )
    # Note: Normal prior does not work well with a bounded domain — only 47% of
    # Normal mass falls inside [-1,1]².  Use uniform for bounded RQS flows.
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
        help="Conditioner-MLP activation",
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

    if args.mode == "kl" and args.image is None:
        parser.error("--mode kl requires --image")

    rotations = args.rotations
    if args.prior == "uniform" and rotations:
        print(
            "note: rotations push latents outside the uniform support and can "
            "lead to artifacts with uniform prior"
        )
    cfg = build_config(
        args.flow, args.prior, rotations, args.activation, args.optimizer, args.debug
    )

    np.random.seed(0)

    # Build evaluation samples (used for JSD histogram in both modes).
    if args.image:
        print(f"Sampling {args.n_data:,} evaluation points from {args.image}")
        samples = image_to_samples(args.image, args.n_data)
    else:
        print(f"Using 5-component Gaussian mixture ({args.n_data:,} samples)")
        samples = gaussian_mixture_samples(args.n_data)

    # KL mode: also load the raw luminance map for the GPU training buffer.
    target_lum = None
    if args.mode == "kl":
        target_lum = load_luminance(args.image)
        print(
            f"KL target: {args.image}  "
            f"({target_lum.shape[1]}×{target_lum.shape[0]} px, "
            f"max={target_lum.max():.1f})"
        )

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

    weight_decay = args.weight_decay if args.optimizer == "adamw" else 0.0
    learner = NDELearner(
        device,
        cfg,
        samples,
        mode=args.mode,
        target_lum=target_lum,
        loss_scale=args.loss_scale,
        grad_clip=args.grad_clip,
        weight_decay=weight_decay,
    )

    save_field(learner.analytic_hist, "target_hist.exr")
    print("Target histogram saved → target_hist.exr")

    prefix = args.mode  # "mle_step…" or "kl_step…"
    train(learner, args.steps, args.lr, prefix=prefix)

    density = learner.eval_density()
    save_field(density, args.out)
    print(f"Density saved → {args.out}")

    if args.out_samples:
        pts = learner.generate_samples(10_000)
        np.save(args.out_samples, pts)
        print(f"Samples saved → {args.out_samples}")


if __name__ == "__main__":
    main()
