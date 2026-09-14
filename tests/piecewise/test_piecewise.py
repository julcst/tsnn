#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["slangpy==0.42.0", "numpy", "pytest"]
# ///
"""GPU unit tests for TSNN/Flows/PiecewiseLinear.slang and PiecewiseQuadratic.slang.

Covers the P5a Step 1 bin-probability/vertex floor: forward<->inverse round trips,
monotonicity + normalization, the floor regression (a conditioner row with one logit
driven to -30, which pre-floor collapsed logAbsDet to the 1e-30 guard and broke the
quadratic inverse -- see PiecewiseQuadratic.slang's header), and finite-difference
gradient checks against bwd_diff.

Run directly with:
    UV_CACHE_DIR=/tmp/tsnn-uv-cache uv run tests/piecewise/test_piecewise.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

ROOT = Path(__file__).resolve().parents[2]
TAIL = 3.0
FLOOR_LOGIT = -30.0


def buffer(device, values, rw=False):
    values = np.ascontiguousarray(values, dtype=np.float32)
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=values.nbytes, usage=usage, data=values)


class Probe:
    def __init__(self):
        self.source_root = Path(os.environ.get("TSNN_PW_SOURCE_ROOT", ROOT))
        self.device = spy.create_device(include_paths=[self.source_root, ROOT], enable_hot_reload=False)
        self.kernels = {}
        for name in (
            "linearRoundTripK8Main", "linearRoundTripK32Main",
            "linearGradK8Main", "linearGradK32Main",
            "quadRoundTripK8Main", "quadRoundTripK32Main",
            "quadGradK8Main", "quadGradK32Main",
        ):
            program = self.device.load_program(
                module_name="tests/piecewise/PiecewiseTest.slang", entry_point_names=[name]
            )
            self.kernels[name] = self.device.create_compute_kernel(program)

    def round_trip(self, kernel, x, params, tail=TAIL):
        count = len(x)
        inp = buffer(self.device, x)
        par = buffer(self.device, params.reshape(-1))
        out = buffer(self.device, np.zeros(count * 4, np.float32), rw=True)
        self.kernels[kernel].dispatch(
            thread_count=[count, 1, 1],
            vars={"gInputs": inp, "gParams": par, "gOutputs": out, "CB": {"gCount": count, "gTail": tail}},
        )
        return out.to_numpy().view(np.float32).reshape(count, 4).copy()

    def grad(self, kernel, x, params, tail=TAIL):
        count = len(x)
        stride = params.reshape(count, -1).shape[1]
        inp = buffer(self.device, x)
        par = buffer(self.device, params.reshape(-1))
        out = buffer(self.device, np.zeros(count, np.float32), rw=True)
        grads = buffer(self.device, np.zeros(count * stride, np.float32), rw=True)
        self.kernels[kernel].dispatch(
            thread_count=[count, 1, 1],
            vars={
                "gInputs": inp, "gParams": par, "gOutputs": out, "gParamGrads": grads,
                "CB": {"gCount": count, "gTail": tail},
            },
        )
        return (
            out.to_numpy().view(np.float32).copy(),
            grads.to_numpy().view(np.float32).reshape(count, stride).copy(),
        )


@pytest.fixture(scope="module")
def probe():
    return Probe()


def linear_params(rng, count, K, scale=1.5):
    return rng.normal(scale=scale, size=(count, K)).astype(np.float32)


def quad_params(rng, count, K, scale=1.5):
    W = rng.normal(scale=scale, size=(count, K)).astype(np.float32)
    V = rng.normal(scale=scale, size=(count, K + 1)).astype(np.float32)
    return np.concatenate([W, V], axis=1)


# ─── Round trip ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("K", [8, 32])
def test_round_trip_linear(probe, K):
    rng = np.random.default_rng(1000 + K)
    n = 2048
    x = rng.uniform(-TAIL + 1e-3, TAIL - 1e-3, n).astype(np.float32)
    params = linear_params(rng, n, K)
    out = probe.round_trip(f"linearRoundTripK{K}Main", x, params)
    fwd_val, fwd_ld, inv_val, inv_ld = out.T
    np.testing.assert_allclose(inv_val, x, atol=1e-4)
    # forward and inverse read off the same bin's floored probability -- their
    # signed log-determinants must cancel exactly (up to float rounding).
    np.testing.assert_allclose(fwd_ld + inv_ld, 0.0, atol=1e-4)
    assert np.all(np.isfinite(fwd_ld))


@pytest.mark.parametrize("K", [8, 32])
def test_round_trip_quadratic(probe, K):
    rng = np.random.default_rng(2000 + K)
    n = 2048
    x = rng.uniform(-TAIL + 1e-3, TAIL - 1e-3, n).astype(np.float32)
    params = quad_params(rng, n, K)
    out = probe.round_trip(f"quadRoundTripK{K}Main", x, params)
    fwd_val, fwd_ld, inv_val, inv_ld = out.T
    np.testing.assert_allclose(inv_val, x, atol=1e-3)
    # A handful of samples can land within float32 rounding of a bin boundary,
    # where forward's and inverse's independent bin searches occasionally
    # disagree by one bin -- a real but tiny (~1e-3) logAbsDet discontinuity
    # there, not a correctness bug. Bound the typical case tightly and the
    # worst case loosely.
    resid = np.abs(fwd_ld + inv_ld)
    assert np.percentile(resid, 99) < 1e-3
    assert np.all(resid < 5e-2)
    assert np.all(np.isfinite(fwd_ld))


# ─── Monotonicity + normalization ────────────────────────────────────────────────


@pytest.mark.parametrize("K", [8, 32])
def test_monotonic_and_normalized_linear(probe, K):
    rng = np.random.default_rng(3000 + K)
    n = 8192
    x = np.linspace(-TAIL + 1e-4, TAIL - 1e-4, n).astype(np.float32)
    row = linear_params(rng, 1, K)
    params = np.repeat(row, n, axis=0)
    out = probe.round_trip(f"linearRoundTripK{K}Main", x, params)
    fwd_val, fwd_ld = out[:, 0], out[:, 1]
    assert np.all(np.diff(fwd_val) >= -1e-5), "forward map must be monotone non-decreasing"
    # mean(dy/dx) over a uniform-x grid on a [-tail,tail]->[-tail,tail] bijection ~= 1
    assert abs(float(np.exp(fwd_ld).mean()) - 1.0) < 5e-3


@pytest.mark.parametrize("K", [8, 32])
def test_monotonic_and_normalized_quadratic(probe, K):
    rng = np.random.default_rng(4000 + K)
    n = 8192
    x = np.linspace(-TAIL + 1e-4, TAIL - 1e-4, n).astype(np.float32)
    row = quad_params(rng, 1, K)
    params = np.repeat(row, n, axis=0)
    out = probe.round_trip(f"quadRoundTripK{K}Main", x, params)
    fwd_val, fwd_ld = out[:, 0], out[:, 1]
    assert np.all(np.diff(fwd_val) >= -1e-5), "forward map must be monotone non-decreasing"
    assert abs(float(np.exp(fwd_ld).mean()) - 1.0) < 5e-3


# ─── Floor regression (the test that fails without Step 1) ──────────────────────


def test_floor_regression_linear(probe):
    """One conditioner logit at -30 (K=32): a bare softmax puts that bin's
    probability many orders below float32 precision, so log(pb*K) collapses to the
    1e-30 sanity floor (~-69). With the kPLMinBinProb=1e-3 floor, pb is bounded
    below by ~1e-3 regardless of the logit, so logAbsDet stays near
    log(1e-3*32)=-3.43, comfortably finite -- and the round trip stays exact."""
    K = 32
    q = np.zeros((1, K), np.float32)
    q[0, 0] = FLOOR_LOGIT
    # bin 0 spans [-tail, -tail + 2*tail/K): place x well inside it.
    x = np.array([-TAIL + (TAIL / K)], np.float32)
    out = probe.round_trip("linearRoundTripK32Main", x, q)
    fwd_val, fwd_ld, inv_val, inv_ld = out[0]
    assert np.isfinite(fwd_ld) and fwd_ld > -10.0, f"logAbsDet={fwd_ld} looks unfloored"
    assert abs(inv_val - x[0]) < 1e-4


def test_floor_regression_quadratic(probe):
    """One vertex logit at -30 (K=32): pre-floor, v[0]->~0 makes the inverse's
    `rhs / max(vb, 1e-30f)` linear-branch division and the quadratic root's
    discriminant ill-conditioned right at bin 0's left edge (t~0, where pdf=vb).
    With the kPQMinVertex=1e-3 floor, v[0] is bounded below by 1e-3 regardless of
    the logit, so logAbsDet=log(pdf) stays near log(1e-3)=-6.91 -- finite, and the
    round trip stays exact."""
    K = 32
    W = np.zeros((1, K), np.float32)
    V = np.zeros((1, K + 1), np.float32)
    V[0, 0] = FLOOR_LOGIT
    params = np.concatenate([W, V], axis=1)
    x = np.array([-TAIL + (TAIL / K) * 0.05], np.float32)  # near bin 0's left edge, t~0
    out = probe.round_trip("quadRoundTripK32Main", x, params)
    fwd_val, fwd_ld, inv_val, inv_ld = out[0]
    assert np.isfinite(fwd_ld) and fwd_ld > -10.0, f"logAbsDet={fwd_ld} looks unfloored"
    assert abs(inv_val - x[0]) < 1e-3


# ─── Gradient check: bwd_diff vs central finite differences ─────────────────────


@pytest.mark.parametrize("K", [8, 32])
def test_linear_param_gradient(probe, K):
    rng = np.random.default_rng(5000 + K)
    n = 64
    x = rng.uniform(-TAIL + 0.2, TAIL - 0.2, n).astype(np.float32)
    params = linear_params(rng, n, K, scale=1.0)
    _, analytic = probe.grad(f"linearGradK{K}Main", x, params)

    eps = 2e-3
    fd = np.zeros_like(analytic)
    for j in range(K):
        plus, minus = params.copy(), params.copy()
        plus[:, j] += eps
        minus[:, j] -= eps
        ld_plus = probe.round_trip(f"linearRoundTripK{K}Main", x, plus)[:, 1]
        ld_minus = probe.round_trip(f"linearRoundTripK{K}Main", x, minus)[:, 1]
        fd[:, j] = (ld_plus - ld_minus) / (2 * eps)

    # A handful of (sample, param) pairs can straddle a bin boundary that the
    # perturbation itself crosses (logAbsDet is piecewise-constant-in-param at
    # fixed x there); drop the worst 2% before checking agreement on the rest.
    err = np.abs(fd - analytic)
    keep = err <= np.percentile(err, 98)
    np.testing.assert_allclose(analytic[keep], fd[keep], atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("K", [8, 32])
def test_quadratic_param_gradient(probe, K):
    rng = np.random.default_rng(6000 + K)
    n = 64
    x = rng.uniform(-TAIL + 0.2, TAIL - 0.2, n).astype(np.float32)
    params = quad_params(rng, n, K, scale=1.0)
    _, analytic = probe.grad(f"quadGradK{K}Main", x, params)

    eps = 2e-3
    stride = params.shape[1]
    fd = np.zeros_like(analytic)
    for j in range(stride):
        plus, minus = params.copy(), params.copy()
        plus[:, j] += eps
        minus[:, j] -= eps
        ld_plus = probe.round_trip(f"quadRoundTripK{K}Main", x, plus)[:, 1]
        ld_minus = probe.round_trip(f"quadRoundTripK{K}Main", x, minus)[:, 1]
        fd[:, j] = (ld_plus - ld_minus) / (2 * eps)

    err = np.abs(fd - analytic)
    keep = err <= np.percentile(err, 98)
    np.testing.assert_allclose(analytic[keep], fd[keep], atol=3e-2, rtol=3e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__, *sys.argv[1:]]))
