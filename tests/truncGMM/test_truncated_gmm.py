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
import sys
import json
import hashlib
from time import perf_counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import slangpy as spy
from scipy.special import ndtr, ndtri, logsumexp
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
        self.source_root = Path(os.environ.get("TSNN_GMM_SOURCE_ROOT", ROOT))
        self.device = spy.create_device(
            include_paths=[self.source_root, ROOT], enable_hot_reload=False
        )
        self.kernels = {}
        for name in (
            "cdfMain",
            "componentGradMain",
            "nllGradMain",
            "nllArrayGradMain",
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

    def create_workspace(self, count):
        return {
            "inputs": buffer(self.device, np.zeros((count, 2), np.float32)),
            "losses": buffer(self.device, np.zeros(count, np.float32), rw=True),
            "grads": buffer(
                self.device, np.zeros((PARAM_COUNT, count), np.float32), rw=True
            ),
        }

    def nll_grad(
        self,
        samples,
        state,
        read_loss=False,
        read_grad=False,
        workspace=None,
        kernel="nllArrayGradMain",
    ):
        if workspace is None:
            workspace = self.create_workspace(len(samples))
        inp, losses, grads = (workspace[key] for key in ("inputs", "losses", "grads"))
        inp.copy_from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        self.kernels[kernel].dispatch(
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
            grads.to_numpy().view(np.float32).reshape(PARAM_COUNT, -1).mean(axis=1)
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


def test_warm_throughput(probe):
    """GPU timestamps exclude compilation, allocation, upload and readback.

    Use TSNN_GMM_SOURCE_ROOT to benchmark an isolated source snapshot with
    this same harness. Wall time includes dispatch/submission and final wait.
    """
    print(f"GPU: {probe.device.info.adapter_name} ({probe.device.info.api_name})")
    results = []
    rng = np.random.default_rng(17)
    repeats = int(os.environ.get("TSNN_GMM_BENCH_REPEATS", "200"))
    assert repeats > 0
    for count in (256, 16384, 65536):
        inputs = buffer(probe.device, rng.random((count, 2), dtype=np.float32))
        outputs = buffer(probe.device, np.zeros(count, np.float32), rw=True)
        params = buffer(probe.device, initial_params())
        grads = buffer(
            probe.device, np.zeros((count, PARAM_COUNT), np.float32), rw=True
        )
        for name in ("evalMain", "nllGradMain", "nllArrayGradMain"):
            bindings = {
                "gInputs": inputs,
                "gOutputs": outputs,
                "gParams": params,
                "CB": {"gCount": count},
            }
            if name != "evalMain":
                bindings["gParamGrads"] = grads
            kernel = probe.kernels[name]
            warm_started = perf_counter()
            for _ in range(3):
                kernel.dispatch(thread_count=[count, 1, 1], vars=bindings)
            probe.device.wait()
            warm_seconds = perf_counter() - warm_started
            queries = probe.device.create_query_pool(
                spy.QueryType.timestamp, repeats * 2
            )
            started = perf_counter()
            for i in range(repeats):
                kernel.dispatch(
                    thread_count=[count, 1, 1],
                    vars=bindings,
                    query_pool=queries,
                    query_index_before=2 * i,
                    query_index_after=2 * i + 1,
                )
            probe.device.wait()
            wall = perf_counter() - started
            stamps = np.array(
                queries.get_results(0, repeats * 2), dtype=np.uint64
            ).reshape(-1, 2)
            durations = (
                stamps[:, 1] - stamps[:, 0]
            ) / probe.device.info.timestamp_frequency
            median = float(np.median(durations))
            row = {
                "kernel": name,
                "batch": count,
                "warmup_seconds": warm_seconds,
                "gpu_median_ms": median * 1000,
                "gpu_p95_ms": float(np.percentile(durations, 95)) * 1000,
                "gpu_samples_per_second": count / median,
                "wall_samples_per_second": count * repeats / wall,
            }
            results.append(row)
            print(json.dumps(row))
    OUT_DIR.mkdir(exist_ok=True)
    label = os.environ.get("TSNN_GMM_BENCH_LABEL", "working")
    report = {
        "adapter": probe.device.info.adapter_name,
        "api": probe.device.info.api_name,
        "repeats": repeats,
        "source_sha256": {
            name: hashlib.sha256(
                (probe.source_root / "TSNN/Mixtures" / name).read_bytes()
            ).hexdigest()
            for name in ("TruncatedGMM.slang", "GaussianHelpers.slang")
        },
        "measurements": results,
    }
    (OUT_DIR / f"benchmark_{label}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def reference_nll(params, points):
    p = np.asarray(params, np.float64).reshape(COMPONENTS, 5)
    sigma = np.exp(p[:, 3:5])
    z = (points[:, None, :] - p[:, 1:3]) / sigma
    mass = ndtr((1 - p[:, 1:3]) / sigma) - ndtr(-p[:, 1:3] / sigma)
    logits = p[:, 0] - (
        0.5 * z * z + p[:, 3:5] + 0.5 * np.log(2 * np.pi) + np.log(mass)
    ).sum(axis=2)
    return -(logsumexp(logits, axis=1) - logsumexp(p[:, 0])).mean()


@pytest.mark.parametrize("kernel", ["nllGradMain", "nllArrayGradMain"])
def test_parameter_gradients(probe, kernel):
    rng = np.random.default_rng(23)
    points = rng.random((257, 2), dtype=np.float32)
    params = initial_params().reshape(COMPONENTS, 5)
    params[:, 0] = rng.normal(size=COMPONENTS)
    params[:, 1:3] = rng.uniform(-0.1, 1.1, (COMPONENTS, 2))
    params[:, 3:5] = rng.uniform(-1.7, -0.2, (COMPONENTS, 2))
    params = params.reshape(-1)
    loss, _, grad = probe.nll_grad(
        points, probe.create_adam_state(params), True, True, kernel=kernel
    )
    expected = []
    for j in range(PARAM_COUNT):
        plus, minus = params.astype(np.float64), params.astype(np.float64)
        plus[j] += 1e-4
        minus[j] -= 1e-4
        expected.append(
            (reference_nll(plus, points) - reference_nll(minus, points)) / 2e-4
        )
    np.testing.assert_allclose(loss, reference_nll(params, points), atol=2e-6)
    np.testing.assert_allclose(grad, expected, atol=2e-5, rtol=2e-3)
    # A common large logit offset must not change either density or gradient.
    shifted = params.reshape(COMPONENTS, 5).copy()
    shifted[:, 0] += 1000
    shifted_loss, _, shifted_grad = probe.nll_grad(
        points, probe.create_adam_state(shifted.reshape(-1)), True, True, kernel=kernel
    )
    np.testing.assert_allclose(shifted_loss, loss, atol=1e-4)
    np.testing.assert_allclose(shifted_grad, grad, atol=1e-5, rtol=2e-3)


def test_component_pdf_derivative(probe):
    rng = np.random.default_rng(29)
    inputs = np.column_stack(
        (
            rng.uniform(0, 1, (128, 2)),
            rng.uniform(-0.1, 1.1, (128, 2)),
            rng.uniform(-1.5, 0.5, (128, 2)),
        )
    ).astype(np.float32)
    actual = probe.run("componentGradMain", inputs, len(inputs) * 7).reshape(-1, 7)

    def reference(values):
        x, mean, log_sigma = values[:, :2], values[:, 2:4], values[:, 4:]
        sigma = np.exp(log_sigma)
        mass = ndtr((1 - mean) / sigma) - ndtr(-mean / sigma)
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2 - log_sigma).prod(axis=1) / (
            2 * np.pi * mass.prod(axis=1)
        )

    values = inputs.astype(np.float64)
    np.testing.assert_allclose(actual[:, 0], reference(values), atol=3e-6, rtol=3e-5)
    for j in range(6):
        plus, minus = values.copy(), values.copy()
        plus[:, j] += 1e-5
        minus[:, j] -= 1e-5
        expected = (reference(plus) - reference(minus)) / 2e-5
        np.testing.assert_allclose(actual[:, j + 1], expected, atol=2e-5, rtol=3e-4)


def test_tail_diagnostics(probe):
    # Was diagnostic-only: the old erf-sum truncation mass in
    # logTruncGaussianPDF underflowed to exactly 0 once a component's
    # standardized interval sat entirely in one tail (mean/log_sigma pairs
    # below are exactly that regime), producing NaN gradients no gradient
    # guard could recover. logTruncMass (TSNN/Mixtures/TruncatedGMM.slang)
    # now routes both tails through a stable erfcx-based log-CDF instead of
    # clamping mean/log_sigma away from the collapse, so this is a real
    # regression assertion: every configuration below must stay finite.
    points = np.array([[0, 0], [0.5, 0.5], [1, 1]], np.float32)
    for mean, log_sigma in ((0.5, -3), (-0.5, -1), (-0.5, -3), (1.5, -3)):
        params = initial_params().reshape(COMPONENTS, 5)
        params[:, 1:3] = mean
        params[:, 3:5] = log_sigma
        loss, grads, _ = probe.nll_grad(
            points, probe.create_adam_state(params.reshape(-1)), True
        )
        values = grads.to_numpy().view(np.float32)
        nonfinite = np.count_nonzero(~np.isfinite(values))
        print(
            f"tail mean={mean} log_sigma={log_sigma}: loss={loss}, "
            f"nonfinite_gradients={nonfinite}/{values.size}"
        )
        assert np.isfinite(loss), f"non-finite loss at mean={mean} log_sigma={log_sigma}"
        assert nonfinite == 0, (
            f"{nonfinite} non-finite gradients at mean={mean} log_sigma={log_sigma}"
        )


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
    assert TRAIN_STEPS > 0
    rng = np.random.default_rng(5)
    reference = einstein_density()
    data = einstein_samples(reference, 16_384, rng)
    state = probe.create_adam_state(initial_params())

    # Compile and execute both training kernels before starting the clock.
    workspace = probe.create_workspace(BATCH_SIZE)
    _, warm_grads, _ = probe.nll_grad(data[:BATCH_SIZE], state, workspace=workspace)
    probe.adam_step(state, warm_grads, BATCH_SIZE, 1, learning_rate=0.015)
    probe.device.wait()
    state = probe.create_adam_state(initial_params())
    probe.device.wait()
    started = perf_counter()
    initial = None
    progress = trange(
        1, TRAIN_STEPS + 1, desc="GMM Adam", unit="step", dynamic_ncols=True
    )
    for step in progress:
        batch = data[rng.integers(0, len(data), BATCH_SIZE)]
        report = step == 1 or step % REPORT_EVERY == 0
        loss, grads, grad = probe.nll_grad(
            batch, state, read_loss=report, read_grad=step == 1, workspace=workspace
        )
        if initial is None:
            initial = loss
            assert grad is not None and np.max(np.abs(grad)) > 1e-4
        if loss is not None:
            progress.set_postfix(nll=f"{loss:.4f}")
        probe.adam_step(state, grads, len(batch), step, learning_rate=0.015)

    probe.device.wait()
    elapsed = perf_counter() - started
    print(
        f"\nTraining (warm): {elapsed / TRAIN_STEPS * 1e3:.3f} ms/step, "
        f"{TRAIN_STEPS * BATCH_SIZE / elapsed:,.0f} samples/s"
    )
    params = probe.parameters(state)
    final, _, _ = probe.nll_grad(data[:1024], state, read_loss=True)
    assert np.isfinite(final) and final < initial - 0.05, (
        f"NLL did not converge: {initial:.4f} -> {final:.4f}"
    )

    print(f"Fit NLL: {initial:.6f} -> {final:.6f}")
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "training.json").write_text(
        json.dumps(
            {
                "steps": TRAIN_STEPS,
                "batch": BATCH_SIZE,
                "seconds": elapsed,
                "samples_per_second": TRAIN_STEPS * BATCH_SIZE / elapsed,
                "initial_nll": initial,
                "final_nll": final,
            },
            indent=2,
        )
        + "\n"
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
    raise SystemExit(pytest.main(["-s", __file__, *sys.argv[1:]]))
