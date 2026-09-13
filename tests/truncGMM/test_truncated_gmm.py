#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["slangpy>=0.42,<0.44", "numpy", "scipy", "matplotlib", "pytest", "tqdm"]
# ///
"""GPU unit tests for TSNN/Mixtures/TruncatedGMM.slang.

Run directly with:
    UV_CACHE_DIR=/tmp/tsnn-uv-cache uv run tests/truncGMM/test_truncated_gmm.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import slangpy as spy
from scipy.special import ndtr, ndtri
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
COMPONENTS = 16
PARAM_COUNT = COMPONENTS * 5
# The default is deliberately longer than the former exploratory probe. A
# reduced value is useful while iterating on the GPU kernel locally.
TRAIN_STEPS = int(os.environ.get("TSNN_GMM_TEST_STEPS", "10000"))
BATCH_SIZE = 256
REPORT_EVERY = 500


def buffer(device, values, rw=False):
    values = np.ascontiguousarray(values, dtype=np.float32)
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=values.nbytes, usage=usage, data=values)


class Probe:
    def __init__(self):
        self.device = spy.create_device(include_paths=[ROOT])
        self.kernels = {}
        for name in (
            "cdfMain",
            "nllGradMain",
            "evalMain",
            "halfEvalMain",
            "sampleMain",
            "adamMain",
        ):
            program = self.device.load_program(
                module_name="tests/truncGMM/TruncatedGMMTest.slang",
                entry_point_names=[name],
            )
            self.kernels[name] = self.device.create_compute_kernel(program)

    def run(self, name, inputs, output_count, params=None):
        inp = buffer(self.device, inputs.reshape(-1))
        out = buffer(self.device, np.zeros(output_count, np.float32), rw=True)
        vars = {"gInputs": inp, "gOutputs": out, "CB": {"gCount": len(inputs)}}
        if params is not None:
            vars["gParams"] = buffer(self.device, params, rw=True)
        self.kernels[name].dispatch(thread_count=[len(inputs), 1, 1], vars=vars)
        return out.to_numpy().view(np.float32).copy()

    def cdf(self, values):
        return self.run("cdfMain", values, len(values) * 2).reshape(-1, 2)

    def create_adam_state(self, params):
        return {
            "params": buffer(self.device, params, rw=True),
            "moment1": buffer(self.device, np.zeros(PARAM_COUNT, np.float32), rw=True),
            "moment2": buffer(self.device, np.zeros(PARAM_COUNT, np.float32), rw=True),
        }

    def nll_grad(self, samples, state, read_loss=False, read_grad=False):
        inp = buffer(self.device, samples.reshape(-1))
        losses = buffer(self.device, np.zeros(len(samples), np.float32), rw=True)
        grads = buffer(
            self.device, np.zeros((len(samples), PARAM_COUNT), np.float32), rw=True
        )
        self.kernels["nllGradMain"].dispatch(
            thread_count=[len(samples), 1, 1],
            vars={
                "gInputs": inp,
                "gOutputs": losses,
                "gParams": state["params"],
                "gParamGrads": grads,
                "CB": {"gCount": len(samples)},
            },
        )
        loss = float(losses.to_numpy().view(np.float32).mean()) if read_loss else None
        grad = (
            grads.to_numpy().view(np.float32).reshape(-1, PARAM_COUNT).mean(axis=0)
            if read_grad
            else None
        )
        return loss, grads, grad

    def adam_step(self, state, grads, batch_size, step, learning_rate):
        self.kernels["adamMain"].dispatch(
            thread_count=[PARAM_COUNT, 1, 1],
            vars={
                "gParamGrads": grads,
                "gTrainParams": state["params"],
                "gMoment1": state["moment1"],
                "gMoment2": state["moment2"],
                "AdamCB": {
                    "gBatchSize": batch_size,
                    "gCurrentStep": float(step),
                    "gLearningRate": learning_rate,
                },
            },
        )

    def parameters(self, state):
        return state["params"].to_numpy().view(np.float32).copy()

    def pdf(self, points, params):
        return self.run("evalMain", points, len(points), params)

    def half_pdf(self, points, params):
        return self.run("halfEvalMain", points, len(points), params)

    def sample(self, count, params):
        dummy = np.zeros((count, 2), np.float32)
        return self.run("sampleMain", dummy, count * 2, params).reshape(-1, 2)


@pytest.fixture(scope="module")
def probe():
    return Probe()


def test_normal_math(probe):
    rng = np.random.default_rng(4)
    x = rng.uniform(-4, 4, 4096).astype(np.float32)
    mean = rng.uniform(-1, 1, len(x)).astype(np.float32)
    log_sigma = rng.uniform(-1.5, 0.7, len(x)).astype(np.float32)
    cdf_gpu = probe.cdf(np.stack([x, mean, log_sigma], axis=1))[:, 0]
    cdf_ref = ndtr((x - mean) / np.exp(log_sigma))
    assert np.max(np.abs(cdf_gpu - cdf_ref)) < 3e-6

    p = rng.uniform(1e-5, 1 - 1e-5, 4096).astype(np.float32)
    inv_gpu = probe.cdf(np.stack([p, mean, log_sigma], axis=1))[:, 1]
    inv_ref = mean + np.exp(log_sigma) * ndtri(p)
    assert np.max(np.abs(inv_gpu - inv_ref)) < 4e-4


def einstein_density():
    bitmap = spy.Bitmap(str(ROOT / "examples/einstein.png")).convert(
        pixel_format=spy.Bitmap.PixelFormat.rgb,
        component_type=spy.Bitmap.ComponentType.float32,
        srgb_gamma=False,
    )
    image = np.asarray(bitmap, dtype=np.float32).mean(axis=2)
    return image / image.mean()


def einstein_samples(density, count, rng):
    weights = density.reshape(-1).copy()
    weights /= weights.sum()
    idx = rng.choice(len(weights), size=count, p=weights)
    y, x = np.divmod(idx, density.shape[1])
    jitter = rng.random((count, 2), dtype=np.float32)
    return np.column_stack(
        ((x + jitter[:, 0]) / density.shape[1], (y + jitter[:, 1]) / density.shape[0])
    ).astype(np.float32)


def initial_params():
    x = (np.arange(4, dtype=np.float32) + 0.5) / 4
    y = (np.arange(4, dtype=np.float32) + 0.5) / 4
    means = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    params = np.zeros((COMPONENTS, 5), np.float32)
    params[:, 1:3] = means
    params[:, 3:5] = -1.4
    return params.reshape(-1)


def save_comparison(reference, pdf, samples):
    OUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    vmax = max(float(reference.max()), float(pdf.max()))
    image_args = {
        "origin": "upper",
        "extent": (0, 1, 1, 0),
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": vmax,
    }
    reference_image = axes[0].imshow(reference, **image_args)
    axes[0].set(title="Reference density", xlabel="x", ylabel="y")
    model = axes[1].imshow(pdf, **image_args)
    axes[1].scatter(
        samples[:3000, 0], samples[:3000, 1], s=1, c="black", alpha=0.35, linewidths=0
    )
    axes[1].set(title=f"{COMPONENTS}-component GMM fit", xlabel="x", ylabel="y")
    fig.colorbar(reference_image, ax=axes, label="PDF")
    fig.savefig(OUT_DIR / "einstein_fit_comparison.png", dpi=160)
    plt.close(fig)


def test_truncated_gmm_fit_with_gpu_adam(probe):
    rng = np.random.default_rng(5)
    reference = einstein_density()
    data = einstein_samples(reference, 16_384, rng)
    state = probe.create_adam_state(initial_params())

    initial = None
    progress = trange(
        1, TRAIN_STEPS + 1, desc="GMM Adam", unit="step", dynamic_ncols=True
    )
    for step in progress:
        batch = data[rng.integers(0, len(data), BATCH_SIZE)]
        report = step == 1 or step % REPORT_EVERY == 0
        loss, grads, grad = probe.nll_grad(
            batch, state, read_loss=report, read_grad=step == 1
        )
        if initial is None:
            initial = loss
            assert grad is not None and np.max(np.abs(grad)) > 1e-4
        if loss is not None:
            progress.set_postfix(nll=f"{loss:.4f}")
        probe.adam_step(state, grads, len(batch), step, learning_rate=0.015)

    params = probe.parameters(state)
    final, _, _ = probe.nll_grad(data[:1024], state, read_loss=True)
    assert np.isfinite(final) and final < initial - 0.05, (
        f"NLL did not converge: {initial:.4f} -> {final:.4f}"
    )

    res = 256
    xy = (np.arange(res, dtype=np.float32) + 0.5) / res
    grid = np.stack(np.meshgrid(xy, xy), axis=-1).reshape(-1, 2)
    pdf = probe.pdf(grid, params).reshape(res, res)
    assert abs(float(pdf.mean()) - 1.0) < 2e-3

    half_pdf = probe.half_pdf(grid[:1024], params)
    assert np.all(np.isfinite(half_pdf))
    assert np.max(np.abs(half_pdf - pdf.reshape(-1)[:1024])) < 5e-3

    samples = probe.sample(65_536, params)
    assert np.all((samples >= 0) & (samples <= 1))
    bins = 32
    hist, _, _ = np.histogram2d(
        samples[:, 0], samples[:, 1], bins=bins, range=((0, 1), (0, 1))
    )
    expected = pdf.reshape(bins, res // bins, bins, res // bins).mean(axis=(1, 3))
    expected /= expected.sum()
    assert np.abs(hist / hist.sum() - expected.T).sum() < 0.12
    save_comparison(reference, pdf, samples)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))
