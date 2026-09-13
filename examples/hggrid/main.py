#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy>=0.42,<0.43", "numpy", "tqdm"]
# ///
"""
HGGrid — a hierarchical grid of truncated Gaussians for 2-D density estimation.

TSNN/Mixtures/TruncatedGMM.slang fits a target well but every loop in its
eval()/sample() is [ForceUnroll]ed over the component count, so cost *and*
reverse-mode shader size grow linearly with it.  HGGrid instead learns a
coarse GRID_DIM x GRID_DIM categorical (Histogram) that picks a cell, and a
small MLP -- driven by a one-hot encoding of that cell -- predicts a few
truncated Gaussians over the cell's *local* coordinates.  Expressiveness now
scales with the number of cells (cheap: one softmax) instead of with the
unrolled mixture size.

Two training modes (mirrors examples/nde):

MLE mode (--mode mle, default):
  Maximize log-likelihood of data samples drawn from einstein.png.

KL mode (--mode kl):
  Minimize KL(target ‖ q_θ) via importance-weighted model samples, as in
  Mueller et al. "Neural Importance Sampling" (2019), §3.1:
    1. Draw x ~ q_θ via sampleModel's single ancestral pass (GPU).
    2. Look up unnormalized target f(x) from the image luminance buffer.
    3. Backpropagate with gradient weight  -f(x) / q_θ(x).

Domain is the unit square [0,1]^2 (TruncatedGMM's native truncation), so
unlike examples/nde there is no ±domain rescale and no row flip anywhere in
this file: row 0 of the image is x.y = 0, matching evalMain in Infer.cs.slang.

Usage:
  uv run main.py                                    # MLE fit of einstein.png
  uv run main.py --mode kl --steps 10000
  uv run main.py --steps 500 --debug --check-grad    # smoke test + grad check
"""

import argparse
from pathlib import Path

import numpy as np
import slangpy as spy
from tqdm import trange

# ─── Architecture constants (must match Network.slang constexprs exactly) ─────

GRID_DIM = 8
NUM_GAUSSIANS = 4
HIDDEN = 64
DEPTH = 3

ACTIVATIONS = {
    "relu": 0, "leakyrelu": 1, "swish": 2, "silu": 2,
    "gelu": 3, "elu": 4, "tanh": 5, "mish": 6,
}

LOSS_SCALE = 128.0
GRAD_CLIP = 1.0

IMAGE_PATH = Path(__file__).parent.parent / "einstein.png"
OUT_DIR = Path(__file__).parent / "output"

# ─── Parameter layout (mirrors Network.slang / TSNN/Utils/MLP.slang) ──────────


def _align4(x: int) -> int:
    return (x + 3) & ~3


def _mlp_byte_size(input_dim: int, hidden: int, depth: int, output_dim: int) -> int:
    byte_off = 0
    for l in range(depth + 1):
        in_size = input_dim if l == 0 else hidden
        out_size = output_dim if l == depth else hidden
        byte_off = _align4(byte_off + 2 * in_size * out_size)  # half weights
        byte_off = _align4(byte_off + 2 * out_size)             # half bias
    return byte_off


def build_config(
    grid_dim: int,
    num_gaussians: int,
    hidden: int,
    depth: int,
    activation: str = "leakyrelu",
    debug: bool = False,
) -> dict:
    num_cells = grid_dim * grid_dim
    encoded_dim = 2 * grid_dim
    fine_param_count = num_gaussians * 5  # TruncatedGMM<N>::STRIDE = 1 + 2*2
    param_bytes = _mlp_byte_size(encoded_dim, hidden, depth, fine_param_count)
    param_elems = param_bytes // 2  # half elements
    return {
        "defines": {
            "GRID_DIM": str(grid_dim),
            "NUM_GAUSSIANS": str(num_gaussians),
            "HG_HIDDEN": str(hidden),
            "HG_DEPTH": str(depth),
            "ACTIVATION": str(ACTIVATIONS[activation]),
            "DEBUG_COUNTERS": str(int(debug)),
        },
        "debug": debug,
        "grid_dim": grid_dim,
        "num_cells": num_cells,
        "hist_bytes": num_cells * 4,       # float32
        "param_bytes": param_bytes,        # half mirror
        "param_elems": param_elems,
        "moment_bytes": param_elems * 4,   # float32 master / moments
        "label": (
            f"{grid_dim}x{grid_dim} grid, {num_gaussians} gaussians/cell, "
            f"{hidden}x{depth} {activation} MLP"
        ),
    }


# ─── Training hyper-parameters ────────────────────────────────────────────────

BATCH_SIZE = 1 << 14
N_DATA = 1 << 20
STEPS = 10_000
LR = 3e-3
DISPLAY_EVERY = 1_000
GRID_RES = 256

# ─── Data / target generation ──────────────────────────────────────────────────


def load_luminance(img_path) -> np.ndarray:
    """Load a greyscale luminance map, linear (no sRGB gamma) -- matches
    einstein_density in tests/truncGMM/test_truncated_gmm.py so JSD numbers
    are directly comparable to the 16-component TruncatedGMM baseline."""
    bitmap = spy.Bitmap(str(img_path)).convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.float32,
        srgb_gamma=False,
    )
    return np.asarray(bitmap, dtype=np.float32).mean(axis=2)


def image_to_samples(density: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw samples from an image treated as an unnormalised PDF.  Mirrors
    einstein_samples in tests/truncGMM/test_truncated_gmm.py; points are
    already in [0,1]^2, HGGrid's native domain -- no rescale needed."""
    weights = density.reshape(-1).astype(np.float64)
    weights /= weights.sum()
    idx = rng.choice(len(weights), size=n, p=weights)
    y, x = np.divmod(idx, density.shape[1])
    jitter = rng.random((n, 2), dtype=np.float32)
    return np.column_stack(
        ((x + jitter[:, 0]) / density.shape[1], (y + jitter[:, 1]) / density.shape[0])
    ).astype(np.float32)


def analytic_histogram(samples: np.ndarray, grid_res: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, grid_res + 1)
    hist, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=[edges, edges])
    hist = hist.T.astype(np.float64)  # [row=y, col=x]
    return (hist / hist.sum()).astype(np.float32)


# ─── I/O helpers ────────────────────────────────────────────────────────────────


def save_field(arr: np.ndarray, name: str) -> Path:
    """Write a 2-D float field as an RGBA float .exr under OUT_DIR.  Row 0 is
    x.y = 0, which is also the top of the source image, so -- unlike
    examples/nde -- no vertical flip is needed for correct orientation."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr.astype(np.float32))
    a = a / a.max() if a.max() > 0 else a
    rgba = np.stack([a, a, a, np.ones_like(a)], axis=-1)
    path = OUT_DIR / name
    spy.Bitmap(rgba, spy.Bitmap.PixelFormat.rgba).write(path)
    return path


def make_buf(device, size_bytes: int, rw: bool = False, data=None) -> spy.Buffer:
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    if data is not None:
        return device.create_buffer(size=size_bytes, usage=usage, data=data)
    return device.create_buffer(size=size_bytes, usage=usage)


def make_tex(device, w: int, h: int) -> spy.Texture:
    return device.create_texture(
        width=w,
        height=h,
        format=spy.Format.rgba32_float,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )


# ─── Learner ────────────────────────────────────────────────────────────────────


class HGGridLearner:
    """
    mle — maximize log-likelihood of pre-sampled image data.
    kl  — minimize KL(target ‖ model) via importance-weighted model samples
          (Mueller et al. 2019 §3.1).  target_lum is the float32 [H, W]
          luminance array; samples is used only for the JSD histogram.
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
    ):
        assert mode in ("mle", "kl"), f"Unknown mode {mode!r}"
        if mode == "kl":
            assert target_lum is not None, "KL mode requires target_lum"

        self.device = device
        self.cfg = cfg
        self.mode = mode
        self.loss_scale = loss_scale
        self.grad_clip = grad_clip
        self.debug = cfg.get("debug", False)
        self.samples = samples  # kept around for JSD eval and --check-grad

        self.analytic_hist = analytic_histogram(samples, GRID_RES)

        if mode == "mle":
            self.n_data = len(samples)
            samples_f32 = samples.astype(np.float32).flatten()
            self.data_buf = make_buf(device, int(samples_f32.nbytes), data=samples_f32)

        if mode == "kl":
            lum = target_lum.astype(np.float32)
            self.tex_height, self.tex_width = lum.shape
            self.target_buf = make_buf(device, int(lum.nbytes), data=lum.flatten())

        hist_bytes = cfg["hist_bytes"]
        param_bytes = cfg["param_bytes"]
        moment_bytes = cfg["moment_bytes"]

        self.hist_logits = make_buf(device, hist_bytes, rw=True)
        self.hist_logit_grads = make_buf(device, hist_bytes, rw=True)
        self.hist_moments1 = make_buf(device, hist_bytes, rw=True)
        self.hist_moments2 = make_buf(device, hist_bytes, rw=True)

        self.params = make_buf(device, param_bytes, rw=True)
        self.params_master = make_buf(device, moment_bytes, rw=True)
        self.param_grads = make_buf(device, param_bytes, rw=True)
        self.moments1 = make_buf(device, moment_bytes, rw=True)
        self.moments2 = make_buf(device, moment_bytes, rw=True)

        # Two 4xuint32 blocks back to back: [0:16)=histogram, [16:32)=MLP.
        self.debug_counters = make_buf(device, 32, rw=True) if self.debug else None
        self.density_tex = make_tex(device, GRID_RES, GRID_RES)

        def load(module, entry):
            return device.create_compute_kernel(
                device.load_program(module_name=module, entry_point_names=[entry])
            )

        self.reset_k = load("Optimize.cs.slang", "resetMain")
        self.train_k = load("Train.cs.slang", "trainMain")
        self.train_kl_k = load("Train.cs.slang", "trainKLMain")
        self.train_check_k = load("Train.cs.slang", "trainCheckMain")
        self.optimize_k = load("Optimize.cs.slang", "optimizeMain")
        self.eval_k = load("Infer.cs.slang", "evalMain")
        self.eval_nll_k = load("Infer.cs.slang", "evalNLLMain")
        self.sample_k = load("Infer.cs.slang", "sampleMain")

        self._reset()

    def _reset(self):
        n = 256 * 8
        self.reset_k.dispatch(
            thread_count=[n, 1, 1],
            vars={
                "gHistLogits": self.hist_logits,
                "gParams": self.params,
                "gParamsMaster": self.params_master,
                "CB": {
                    "gLearningRate": LR,
                    "gCurrentStep": 1.0,
                    "gDispatchThreadCount": n,
                    "gLossScale": self.loss_scale,
                    "gGradClip": self.grad_clip,
                    "gWeightDecay": 0.0,
                },
            },
        )
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.hist_moments1)
        enc.clear_buffer(self.hist_moments2)
        enc.clear_buffer(self.moments1)
        enc.clear_buffer(self.moments2)
        self.device.submit_command_buffer(enc.finish())

    def zero_grads(self):
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.hist_logit_grads)
        enc.clear_buffer(self.param_grads)
        self.device.submit_command_buffer(enc.finish())

    def train_step(self, step: int):
        self.zero_grads()
        if self.mode == "mle":
            self.train_k.dispatch(
                thread_count=[BATCH_SIZE, 1, 1],
                vars={
                    "gSamples": self.data_buf,
                    "gHistLogits": self.hist_logits,
                    "gHistLogitGrads": self.hist_logit_grads,
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
                    "gHistLogits": self.hist_logits,
                    "gHistLogitGrads": self.hist_logit_grads,
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
            "gHistLogits": self.hist_logits,
            "gHistLogitGrads": self.hist_logit_grads,
            "gHistMoments1": self.hist_moments1,
            "gHistMoments2": self.hist_moments2,
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
                "gWeightDecay": 0.0,
            },
        }
        if self.debug:
            vars["gDebugCounters"] = self.debug_counters
        self.optimize_k.dispatch(thread_count=[n, 1, 1], vars=vars)

    def read_debug_counters(self) -> dict:
        raw = np.frombuffer(self.debug_counters.to_numpy(), dtype=np.uint32)
        h_over, h_clip, h_zero, h_tot = (int(x) for x in raw[0:4])
        m_over, m_clip, m_zero, m_tot = (int(x) for x in raw[4:8])
        h_tot, m_tot = max(h_tot, 1), max(m_tot, 1)
        enc = self.device.create_command_encoder()
        enc.clear_buffer(self.debug_counters)
        self.device.submit_command_buffer(enc.finish())
        return {
            "hist": {"overflow": h_over / h_tot, "clipped": h_clip / h_tot, "zero": h_zero / h_tot},
            "mlp": {"overflow": m_over / m_tot, "clipped": m_clip / m_tot, "zero": m_zero / m_tot},
        }

    def eval_density(self) -> np.ndarray:
        self.eval_k.dispatch(
            thread_count=[GRID_RES, GRID_RES, 1],
            frameDim=[GRID_RES, GRID_RES],
            vars={"gHistLogits": self.hist_logits, "gParams": self.params, "gDensityTex": self.density_tex},
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
                "gHistLogits": self.hist_logits,
                "gParams": self.params,
                "gSampleOut": out_buf,
                "SampleCB": {"gNumSamples": n, "gSeed": 42},
            },
        )
        raw = np.frombuffer(out_buf.to_numpy(), dtype=np.float32)
        return raw.reshape(n, 2)

    # ─── --check-grad support ──────────────────────────────────────────────

    def eval_nll(self, batch: np.ndarray) -> np.ndarray:
        """log p(x) for each row of `batch` via evalNLLMain -- deterministic,
        no RNG -- so it can finite-difference the exact batch trainCheckMain
        differentiated."""
        n = len(batch)
        test_buf = make_buf(self.device, int(batch.nbytes), data=batch.astype(np.float32))
        out_buf = make_buf(self.device, n * 4, rw=True)
        self.eval_nll_k.dispatch(
            thread_count=[n, 1, 1],
            vars={
                "gHistLogits": self.hist_logits,
                "gParams": self.params,
                "gTestSamples": test_buf,
                "gLogProbsOut": out_buf,
                "NLLCB": {"gNLLCount": n},
            },
        )
        return np.frombuffer(out_buf.to_numpy(), dtype=np.float32).copy()

    def accumulate_check_grad(self, batch: np.ndarray, loss_scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic full-batch d(meanNLL)/dθ via trainCheckMain, split
        into (hist_grad[num_cells], mlp_grad[param_elems]), both already
        divided back down by loss_scale (see Optimize.cs.slang's
        invLossScale)."""
        n = len(batch)
        check_buf = make_buf(self.device, int(batch.nbytes), data=batch.astype(np.float32))
        self.zero_grads()
        self.train_check_k.dispatch(
            thread_count=[n, 1, 1],
            vars={
                "gCheckSamples": check_buf,
                "gHistLogits": self.hist_logits,
                "gHistLogitGrads": self.hist_logit_grads,
                "gParams": self.params,
                "gParamGrads": self.param_grads,
                "CheckCB": {"gCheckBatchSize": n, "gCheckLossScale": loss_scale},
            },
        )
        hist_grad = np.frombuffer(self.hist_logit_grads.to_numpy(), dtype=np.float32).copy() / loss_scale
        mlp_grad = (
            np.frombuffer(self.param_grads.to_numpy(), dtype=np.float16).astype(np.float64).copy() / loss_scale
        )
        return hist_grad, mlp_grad

    def read_hist_logits(self) -> np.ndarray:
        return np.frombuffer(self.hist_logits.to_numpy(), dtype=np.float32).copy()

    def write_hist_logits(self, values: np.ndarray):
        self.hist_logits.copy_from_numpy(np.ascontiguousarray(values, dtype=np.float32))

    def read_mlp_master(self) -> np.ndarray:
        return np.frombuffer(self.params_master.to_numpy(), dtype=np.float32).copy()

    def write_mlp_weight(self, index: int, value: float):
        """Write one MLP weight to both the float32 master and the float16
        compute mirror (mirrors Optimize.cs.slang's dual-write)."""
        master = self.read_mlp_master()
        master[index] = value
        self.params_master.copy_from_numpy(np.ascontiguousarray(master, dtype=np.float32))
        mirror = np.frombuffer(self.params.to_numpy(), dtype=np.float16).copy()
        mirror[index] = np.float16(value)
        self.params.copy_from_numpy(np.ascontiguousarray(mirror, dtype=np.float16))


# ─── Gradient check (--check-grad) ──────────────────────────────────────────────


def check_gradients(
    learner: HGGridLearner,
    batch_size: int = 512,
    n_mlp_probes: int = 32,
    eps_hist: float = 1e-3,
    eps_mlp: float = 2e-2,
    loss_scale: float = 128.0,
    seed: int = 123,
) -> dict:
    """Finite-difference the GPU backward pass against evalNLLMain, entirely
    on-device (no reimplementation of forward() in numpy).  Freezes a batch,
    reads the accumulated gradient from trainCheckMain (a deterministic,
    unhashed variant of trainMain), then for each probed parameter perturbs
    it by ±eps and re-measures mean NLL with evalNLLMain.  Same methodology
    as test_parameter_gradients in tests/truncGMM/test_truncated_gmm.py.
    """
    rng = np.random.default_rng(seed)
    n = min(batch_size, len(learner.samples))
    batch = learner.samples[rng.choice(len(learner.samples), size=n, replace=False)]

    hist_grad, mlp_grad = learner.accumulate_check_grad(batch, loss_scale)

    def mean_nll(b):
        return float(-learner.eval_nll(b).mean())

    # ─ Histogram branch: all num_cells logits ─
    num_cells = learner.cfg["num_cells"]
    orig_hist = learner.read_hist_logits()
    hist_fd = np.zeros(num_cells, dtype=np.float64)
    for j in range(num_cells):
        probe = orig_hist.copy()
        probe[j] = orig_hist[j] + eps_hist
        learner.write_hist_logits(probe)
        nll_plus = mean_nll(batch)
        probe[j] = orig_hist[j] - eps_hist
        learner.write_hist_logits(probe)
        nll_minus = mean_nll(batch)
        hist_fd[j] = (nll_plus - nll_minus) / (2.0 * eps_hist)
    learner.write_hist_logits(orig_hist)

    # accumulate_check_grad already returns d(meanNLL)/dθ (see its docstring),
    # matching hist_fd's central difference of mean_nll directly.
    hist_err = np.abs(hist_fd - hist_grad)
    hist_report = {
        "max_abs_err": float(hist_err.max()),
        "mean_abs_err": float(hist_err.mean()),
        "max_grad_mag": float(np.abs(hist_grad).max()),
    }

    # ─ Fine MLP branch: a random subset of weights ─
    param_elems = learner.cfg["param_elems"]
    n_probe = min(n_mlp_probes, param_elems)
    probe_idx = rng.choice(param_elems, size=n_probe, replace=False)
    orig_master = learner.read_mlp_master()
    mlp_fd = np.zeros(n_probe, dtype=np.float64)
    for k, j in enumerate(probe_idx):
        learner.write_mlp_weight(int(j), float(orig_master[j] + eps_mlp))
        nll_plus = mean_nll(batch)
        learner.write_mlp_weight(int(j), float(orig_master[j] - eps_mlp))
        nll_minus = mean_nll(batch)
        mlp_fd[k] = (nll_plus - nll_minus) / (2.0 * eps_mlp)
    for j in probe_idx:
        learner.write_mlp_weight(int(j), float(orig_master[j]))

    mlp_err = np.abs(mlp_fd - mlp_grad[probe_idx])
    mlp_report = {
        "max_abs_err": float(mlp_err.max()),
        "mean_abs_err": float(mlp_err.mean()),
        "max_grad_mag": float(np.abs(mlp_grad[probe_idx]).max()),
        "n_probed": int(n_probe),
    }

    return {"hist": hist_report, "mlp": mlp_report}


# ─── Training loop ────────────────────────────────────────────────────────────


def train(learner: HGGridLearner, steps: int, lr: float, prefix: str = ""):
    cfg = learner.cfg
    print(
        f"Mode: {learner.mode.upper()}  |  {cfg['label']}  |  "
        f"{cfg['num_cells']} cells + {cfg['param_elems']:,} float16 MLP params "
        f"({cfg['param_bytes'] / 1024:.1f} KB)"
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
                post["hist of/uf/clip"] = (
                    f"{d['hist']['overflow']:.0e}/{d['hist']['zero']:.2f}/{d['hist']['clipped']:.0e}"
                )
                post["mlp of/uf/clip"] = (
                    f"{d['mlp']['overflow']:.0e}/{d['mlp']['zero']:.2f}/{d['mlp']['clipped']:.0e}"
                )
            bar.set_postfix(post)
            tag = f"{prefix}_step{step:06d}" if prefix else f"model_step{step:06d}"
            save_field(density, f"{tag}.exr")

    return learner, best_jsd


# ─── Entry point ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["mle", "kl"], default="mle",
        help="mle: maximize log-likelihood of image samples. kl: minimize "
        "KL(target‖model) via importance-weighted model samples (Mueller et al. 2019).",
    )
    parser.add_argument("--grid-dim", type=int, default=GRID_DIM, help="Coarse categorical grid resolution")
    parser.add_argument("--num-gaussians", type=int, default=NUM_GAUSSIANS, help="Truncated Gaussians per cell")
    parser.add_argument("--hidden", type=int, default=HIDDEN, help="Fine-MLP hidden width")
    parser.add_argument("--depth", type=int, default=DEPTH, help="Fine-MLP depth")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS.keys()), default="leakyrelu")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-data", type=int, default=N_DATA)
    parser.add_argument("--loss-scale", type=float, default=LOSS_SCALE)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--out", default="density.exr", help="Output density .exr")
    parser.add_argument("--out-samples", default=None, help="Save generated samples (.npy)")
    parser.add_argument("--debug", action="store_true", help="Sanitizer under/overflow tallies per branch")
    parser.add_argument(
        "--check-grad", action="store_true",
        help="Finite-difference the GPU gradient against evalNLLMain after training",
    )
    args = parser.parse_args()

    np.random.seed(0)
    cfg = build_config(
        args.grid_dim, args.num_gaussians, args.hidden, args.depth, args.activation, args.debug
    )

    print(f"Sampling {args.n_data:,} evaluation points from {IMAGE_PATH.name}")
    target_lum = load_luminance(IMAGE_PATH)
    rng = np.random.default_rng(0)
    samples = image_to_samples(target_lum, args.n_data, rng)

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

    learner = HGGridLearner(
        device,
        cfg,
        samples,
        mode=args.mode,
        target_lum=target_lum if args.mode == "kl" else None,
        loss_scale=args.loss_scale,
        grad_clip=args.grad_clip,
    )

    target_path = save_field(learner.analytic_hist, "target_hist.exr")
    print(f"Target histogram saved → {target_path}")

    prefix = args.mode
    learner, best_jsd = train(learner, args.steps, args.lr, prefix=prefix)

    density = learner.eval_density()
    density_path = save_field(density, args.out)
    print(f"Density saved → {density_path}")
    print(f"pdf.mean() = {density.mean():.4f} (should be ≈ 1.0 -- validates the 2*log(GRID_DIM) Jacobian term)")

    if args.out_samples:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        samples_path = OUT_DIR / args.out_samples
        pts = learner.generate_samples(10_000)
        np.save(samples_path, pts)
        print(f"Samples saved → {samples_path}")

    try:
        spy.tev.show(spy.Bitmap(np.stack([density] * 3 + [np.ones_like(density)], axis=-1), spy.Bitmap.PixelFormat.rgba), name=f"hggrid_{prefix}")
    except Exception as e:
        print(f"(tev not available: {e})")

    print(f"\nFinal JSD: {learner.compute_jsd(density):.4f}  (best over run: {best_jsd:.4f})")
    print(f"Config: {cfg['label']}, {cfg['num_cells']} cells + {cfg['param_elems']:,} MLP params")

    if args.check_grad:
        print("\nRunning --check-grad (finite-difference vs. backward pass)...")
        report = check_gradients(learner, loss_scale=args.loss_scale)
        h, m = report["hist"], report["mlp"]
        print(
            f"  histogram (64 logits): max|err|={h['max_abs_err']:.2e}  "
            f"mean|err|={h['mean_abs_err']:.2e}  max|grad|={h['max_grad_mag']:.2e}"
        )
        print(
            f"  fine MLP ({m['n_probed']} probed weights): max|err|={m['max_abs_err']:.2e}  "
            f"mean|err|={m['mean_abs_err']:.2e}  max|grad|={m['max_grad_mag']:.2e}"
        )
        ok = h["max_abs_err"] < 5e-2 and m["max_abs_err"] < 5e-2
        print("  Gradient check: " + ("PASS" if ok else "FAIL -- see errors above"))


if __name__ == "__main__":
    main()
