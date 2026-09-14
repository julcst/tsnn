#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["slangpy==0.42.0", "numpy", "scipy", "mpmath", "pytest"]
# ///
"""GPU unit tests for TSNN/Mixtures/VonMisesFisher.slang.

References use mpmath (arbitrary precision) for the single-lobe normalizer/cap-mass,
so they stay exact even at kappa=1e4/1e6 where double precision itself would overflow
forming sinh/exp directly -- the same reason the GPU implementation never forms them
either. vmfHalfspaceLogMass (the Gauss-Legendre approximation) is checked against a
separate, dense (20000-point) double-precision quadrature using scipy's i0e, which is
the "dense numpy reference" the module's own header promises.

Run directly with:
    UV_CACHE_DIR=/tmp/tsnn-uv-cache uv run tests/vmf/test_vmf.py
"""

import os
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
import slangpy as spy
from scipy.special import i0e, logsumexp

ROOT = Path(__file__).resolve().parents[2]
mp.mp.dps = 50  # decimal digits of working precision


def buffer(device, values, rw=False):
    values = np.ascontiguousarray(values, dtype=np.float32)
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=values.nbytes, usage=usage, data=values)


class Probe:
    def __init__(self):
        self.source_root = Path(os.environ.get("TSNN_VMF_SOURCE_ROOT", ROOT))
        self.device = spy.create_device(include_paths=[self.source_root, ROOT], enable_hot_reload=False)
        self.kernels = {}
        for name in (
            "logNormalizerMain", "logPdfMain", "logPdfCapMain", "halfspaceLogMassMain",
            "sampleMain", "sampleCapMain", "logPdfGradMain", "logPdfCapGradMain",
            "mixtureEvalMain", "mixtureSampleMain",
        ):
            program = self.device.load_program(
                module_name="tests/vmf/VonMisesFisherTest.slang", entry_point_names=[name]
            )
            self.kernels[name] = self.device.create_compute_kernel(program)

    def run(self, name, inputs, output_count, params=None):
        count = inputs.shape[0] if inputs.ndim > 1 else len(inputs)
        inp = buffer(self.device, inputs.reshape(-1))
        out = buffer(self.device, np.zeros(output_count, np.float32), rw=True)
        vars = {"gInputs": inp, "gOutputs": out, "CB": {"gCount": count}}
        if params is not None:
            vars["gParams"] = buffer(self.device, params.reshape(-1))
        self.kernels[name].dispatch(thread_count=[count, 1, 1], vars=vars)
        return out.to_numpy().view(np.float32).copy()

    def log_normalizer(self, kappa):
        return self.run("logNormalizerMain", np.asarray(kappa, np.float32), len(kappa))

    def log_pdf(self, mean, kappa, dir_):
        rows = np.concatenate([mean, kappa[:, None], dir_], axis=1)
        return self.run("logPdfMain", rows, len(kappa))

    def log_pdf_cap(self, mean, kappa, dir_, cos_cap):
        rows = np.concatenate([mean, kappa[:, None], dir_, cos_cap[:, None]], axis=1)
        return self.run("logPdfCapMain", rows, len(kappa))

    def halfspace_log_mass(self, mean, kappa, axis):
        rows = np.concatenate([mean, kappa[:, None], axis], axis=1)
        return self.run("halfspaceLogMassMain", rows, len(kappa))

    def sample(self, mean, kappa, xi):
        rows = np.concatenate([mean, kappa[:, None], xi], axis=1)
        return self.run("sampleMain", rows, len(kappa) * 3).reshape(-1, 3)

    def sample_cap(self, mean, kappa, cos_cap, xi):
        rows = np.concatenate([mean, kappa[:, None], cos_cap[:, None], xi], axis=1)
        return self.run("sampleCapMain", rows, len(kappa) * 3).reshape(-1, 3)

    def log_pdf_grad(self, mean, kappa, dir_):
        rows = np.concatenate([mean, kappa[:, None], dir_], axis=1)
        out = self.run("logPdfGradMain", rows, len(kappa) * 5).reshape(-1, 5)
        return out[:, 0], out[:, 1:4], out[:, 4]

    def log_pdf_cap_grad(self, mean, kappa, dir_, cos_cap):
        rows = np.concatenate([mean, kappa[:, None], dir_, cos_cap[:, None]], axis=1)
        out = self.run("logPdfCapGradMain", rows, len(kappa) * 5).reshape(-1, 5)
        return out[:, 0], out[:, 1:4], out[:, 4]

    def mixture_eval(self, params, dir_):
        return self.run("mixtureEvalMain", dir_, len(dir_), params=np.repeat(params[None], len(dir_), axis=0))

    def mixture_sample(self, params, count):
        dummy = np.zeros((count, 3), np.float32)
        rows = np.repeat(params[None], count, axis=0)
        return self.run("mixtureSampleMain", dummy, count * 3, params=rows).reshape(-1, 3)


@pytest.fixture(scope="module")
def probe():
    return Probe()


def random_directions(rng, n):
    v = rng.normal(size=(n, 3))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


# ─── mpmath references ───────────────────────────────────────────────────────────


def log_normalizer_ref(kappa: float) -> float:
    k = mp.mpf(kappa)
    if k < mp.mpf("1e-9"):
        return float(-mp.log(4 * mp.pi))
    return float(mp.log(k) - mp.log(4 * mp.pi) - mp.log(mp.sinh(k)))


def log_cap_mass_ref(kappa: float, cos_cap: float) -> float:
    k, c = mp.mpf(kappa), mp.mpf(cos_cap)
    if k < mp.mpf("1e-9"):
        return float(mp.log(mp.mpf("0.5") * (1 - c)))
    num = mp.e**k - mp.e ** (k * c)
    den = mp.e**k - mp.e ** (-k)
    return float(mp.log(num / den))


def w_quantile_desc(kappa: float, q: float) -> float:
    """w such that P(W>=w)=q for the vMF marginal of w=dot(dir,mean) (mpmath-exact,
    inverting log_cap_mass_ref's own formula -- used to build equal-EXPECTED-mass
    bins for the chi-square goodness-of-fit test below, since kappa concentrates
    almost all mass near w=1 and equal-width bins leave most cells empty)."""
    k, qq = mp.mpf(kappa), mp.mpf(q)
    if k < mp.mpf("1e-9"):
        return float(1 - 2 * qq)
    num = mp.e**k - qq * (mp.e**k - mp.e ** (-k))
    return float(mp.log(num) / k)


def halfspace_log_mass_dense(mean, kappa, axis, n=20000):
    mean, axis = np.asarray(mean, np.float64), np.asarray(axis, np.float64)
    cos_gamma = float(np.clip(mean @ axis, -1.0, 1.0))
    sin_gamma = float(np.sqrt(max(0.0, 1.0 - cos_gamma**2)))
    theta = (np.arange(n) + 0.5) / n * (np.pi / 2)
    dtheta = (np.pi / 2) / n
    log_c = log_normalizer_ref(kappa)
    radial = kappa * sin_gamma * np.sin(theta)
    axial = kappa * cos_gamma * np.cos(theta)
    log_integrand = log_c + np.log(2 * np.pi) + axial + radial + np.log(i0e(radial)) + np.log(np.sin(theta))
    return logsumexp(log_integrand) + np.log(dtheta)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    golden = np.pi * (1 + 5**0.5)
    theta = golden * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float64)


# ─── Normalization ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kappa", [0.0, 1e-4, 1.0, 10.0, 100.0, 1e4])
def test_sphere_normalization(probe, kappa):
    n = 200_000
    dirs = fibonacci_sphere(n).astype(np.float32)
    mean = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n, 1))
    logpdf = probe.log_pdf(mean, np.full(n, kappa, np.float32), dirs)
    weight = 4.0 * np.pi / n
    integral = float(np.exp(logpdf.astype(np.float64)).sum() * weight)
    assert abs(integral - 1.0) < 1e-3, f"kappa={kappa}: integral={integral}"


@pytest.mark.parametrize("kappa,cos_cap", [(1.0, 0.0), (10.0, 0.3), (100.0, -0.5), (1e4, 0.9)])
def test_cap_normalization(probe, kappa, cos_cap):
    n = 400_000
    dirs = fibonacci_sphere(n)
    mean = np.array([0.0, 0.0, 1.0])
    inside = dirs @ mean >= cos_cap
    kept = dirs[inside].astype(np.float32)
    if len(kept) == 0:
        pytest.skip("cap too small for this Fibonacci grid")
    means = np.tile(mean.astype(np.float32), (len(kept), 1))
    logpdf = probe.log_pdf_cap(means, np.full(len(kept), kappa, np.float32), kept,
                                np.full(len(kept), cos_cap, np.float32))
    weight = 4.0 * np.pi / n
    integral = float(np.exp(logpdf.astype(np.float64)).sum() * weight)
    assert abs(integral - 1.0) < 2e-3, f"kappa={kappa} cosCap={cos_cap}: integral={integral}"


def test_halfspace_log_mass_vs_dense_quadrature(probe):
    rng = np.random.default_rng(11)
    kappas = [0.5, 2.0, 10.0, 50.0, 200.0]
    means = random_directions(rng, len(kappas))
    axes = random_directions(rng, len(kappas))
    gpu = probe.halfspace_log_mass(means, np.asarray(kappas, np.float32), axes)
    ref = np.array([halfspace_log_mass_dense(means[i], kappas[i], axes[i]) for i in range(len(kappas))])
    rel_err = np.abs(np.exp(gpu.astype(np.float64)) - np.exp(ref)) / np.exp(ref)
    print(f"vmfHalfspaceLogMass max relative error vs dense quadrature: {rel_err.max():.3e}")
    assert np.all(rel_err < 0.05), rel_err

    # axis == mean must match the exact own-mean-cap formula (cosCap=0) closely --
    # both the coarse GL rule and the dense reference should agree with the exact
    # closed form here, a cross-check the two approximate paths don't share a bug.
    exact = np.array([log_cap_mass_ref(k, 0.0) for k in kappas])
    gpu_axis_eq_mean = probe.halfspace_log_mass(means, np.asarray(kappas, np.float32), means)
    np.testing.assert_allclose(np.exp(gpu_axis_eq_mean.astype(np.float64)), np.exp(exact), rtol=0.03)


# ─── Density agreement with the mpmath twin ───────────────────────────────────────


def test_density_matches_reference(probe):
    rng = np.random.default_rng(21)
    kappas = [0.0, 1e-4, 0.5, 5.0, 50.0, 500.0]
    means = random_directions(rng, len(kappas))
    dirs = random_directions(rng, len(kappas))
    gpu = probe.log_pdf(means, np.asarray(kappas, np.float32), dirs)
    ref = np.array([
        kappas[i] * float(means[i] @ dirs[i]) + log_normalizer_ref(kappas[i]) for i in range(len(kappas))
    ])
    np.testing.assert_allclose(gpu.astype(np.float64), ref, atol=2e-3, rtol=2e-4)


def test_cap_density_matches_reference(probe):
    rng = np.random.default_rng(22)
    kappa, cos_cap = 8.0, 0.2
    n = 32
    means = random_directions(rng, n)
    # bias directions toward each mean so most fall inside its own cap
    dirs = (0.7 * means + 0.3 * random_directions(rng, n))
    dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)
    w = np.einsum("ij,ij->i", means, dirs)
    gpu = probe.log_pdf_cap(means, np.full(n, kappa, np.float32), dirs, np.full(n, cos_cap, np.float32))
    log_mass = log_cap_mass_ref(kappa, cos_cap)
    for i in range(n):
        if w[i] < cos_cap:
            assert gpu[i] < -1e20  # sentinel
        else:
            ref = kappa * float(w[i]) + log_normalizer_ref(kappa) - log_mass
            assert abs(gpu[i] - ref) < 2e-3, (gpu[i], ref)


# ─── Sampling ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kappa", [0.5, 2.0, 8.0, 50.0])
def test_sample_mean_resultant_length(probe, kappa):
    rng = np.random.default_rng(100 + int(kappa))
    n = 1_000_000
    mean = np.array([0.0, 0.0, 1.0], np.float32)
    means = np.tile(mean, (n, 1))
    xi = rng.random((n, 2)).astype(np.float32)
    samples = probe.sample(means, np.full(n, kappa, np.float32), xi)
    assert np.all(np.isfinite(samples))
    norms = np.linalg.norm(samples, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    w_mean = float((samples @ mean).mean())
    k = mp.mpf(float(kappa))
    expected = float(1 / mp.tanh(k) - 1 / k)
    # Monte-Carlo standard error of the mean at n=1e6 is a few times 1e-4 for
    # kappa in this range; 2e-3 is a generous (~5-10 sigma) margin, not a
    # correctness bound -- the point of this test is to catch a wrong sampler
    # formula (order-of-magnitude off), not to measure MC noise.
    assert abs(w_mean - expected) < 2e-3, f"kappa={kappa}: {w_mean} vs {expected}"


@pytest.mark.parametrize("kappa", [1.0, 6.0, 25.0])
def test_sample_matches_pdf_chi2_binning(probe, kappa):
    """Equal-EXPECTED-mass bins (not equal-width): kappa concentrates almost all
    mass near w=1, so fixed-width bins over [-1,1] leave most cells nearly empty
    at large kappa and any per-bin relative-tolerance check is dominated by
    Poisson noise there. Quantile bins give every cell the same expected count,
    making a real scipy.stats.chisquare goodness-of-fit test well-conditioned."""
    from scipy.stats import chisquare

    rng = np.random.default_rng(200 + int(kappa))
    n = 200_000
    mean = np.array([0.0, 0.0, 1.0], np.float32)
    means = np.tile(mean, (n, 1))
    xi = rng.random((n, 2)).astype(np.float32)
    samples = probe.sample(means, np.full(n, kappa, np.float32), xi)
    w = samples @ mean

    nbins = 10
    edges = np.array([w_quantile_desc(kappa, 1.0 - k / nbins) for k in range(nbins + 1)])
    edges[0], edges[-1] = -1.0, 1.0  # exact endpoints regardless of mpmath rounding
    observed, _ = np.histogram(w, bins=edges)
    expected = np.full(nbins, n / nbins)
    assert observed.sum() == n and abs(expected.sum() - n) < 1.0

    stat, pvalue = chisquare(observed, expected)
    # A very low bar (not ~0.05): this must reject a genuinely wrong sampler,
    # not flag ordinary sampling noise across CI runs/seeds.
    assert pvalue > 1e-4, f"kappa={kappa}: chi2={stat:.2f}, p={pvalue:.2e}, observed={observed.tolist()}"


def test_cap_sample_stays_in_cap_and_matches_density(probe):
    rng = np.random.default_rng(31)
    kappa, cos_cap = 3.0, 0.1
    n = 100_000
    mean = np.array([0.0, 0.0, 1.0], np.float32)
    means = np.tile(mean, (n, 1))
    xi = rng.random((n, 2)).astype(np.float32)
    samples = probe.sample_cap(means, np.full(n, kappa, np.float32), np.full(n, cos_cap, np.float32), xi)
    w = samples @ mean
    assert np.all(w >= cos_cap - 1e-4)

    nbins = 12
    edges = np.linspace(cos_cap, 1.0, nbins + 1)
    observed, _ = np.histogram(w, bins=edges)
    log_full = np.array([log_cap_mass_ref(kappa, float(e)) for e in edges])  # mass(w>=e) over FULL sphere
    frac = (np.exp(log_full[:-1]) - np.exp(log_full[1:])) / np.exp(log_cap_mass_ref(kappa, cos_cap))
    expected = frac * n
    rel = np.abs(observed - expected) / np.maximum(expected, 1.0)
    assert np.all(rel < 0.15), list(zip(observed, expected))


# ─── Gradients ─────────────────────────────────────────────────────────────────────

KAPPA_SWEEP = [1e-6, 5e-4, 2e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]  # spans both branches/crossover at 1e-3


def test_logpdf_gradient_fd(probe):
    rng = np.random.default_rng(41)
    n = len(KAPPA_SWEEP)
    means = random_directions(rng, n)
    dirs = random_directions(rng, n)
    kappas = np.asarray(KAPPA_SWEEP, np.float32)
    value, dmean, dkappa = probe.log_pdf_grad(means, kappas, dirs)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(dmean)) and np.all(np.isfinite(dkappa))

    eps = 1e-3
    fd_kappa = (probe.log_pdf(means, kappas + eps, dirs) - probe.log_pdf(means, kappas - eps, dirs)) / (2 * eps)
    np.testing.assert_allclose(dkappa, fd_kappa, atol=2e-2, rtol=2e-2)

    eps_m = 1e-3
    fd_mean = np.zeros_like(dmean)
    for d in range(3):
        plus, minus = means.copy(), means.copy()
        plus[:, d] += eps_m
        minus[:, d] -= eps_m
        fd_mean[:, d] = (probe.log_pdf(plus, kappas, dirs) - probe.log_pdf(minus, kappas, dirs)) / (2 * eps_m)
    np.testing.assert_allclose(dmean, fd_mean, atol=2e-2, rtol=2e-2)


def test_logpdfcap_gradient_fd(probe):
    rng = np.random.default_rng(42)
    n = len(KAPPA_SWEEP)
    means = random_directions(rng, n)
    # dirs sampled close to mean so they land inside a moderate cap
    dirs = (0.85 * means + 0.15 * random_directions(rng, n))
    dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(np.float32)
    kappas = np.asarray(KAPPA_SWEEP, np.float32)
    cos_cap = np.full(n, -0.5, np.float32)
    value, dmean, dkappa = probe.log_pdf_cap_grad(means, kappas, dirs, cos_cap)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(dmean)) and np.all(np.isfinite(dkappa))

    eps = 1e-3
    fd_kappa = (
        probe.log_pdf_cap(means, kappas + eps, dirs, cos_cap) - probe.log_pdf_cap(means, kappas - eps, dirs, cos_cap)
    ) / (2 * eps)
    np.testing.assert_allclose(dkappa, fd_kappa, atol=3e-2, rtol=3e-2)


def test_extreme_kappa_safety(probe):
    for kappa in (0.0, 1e6):
        mean = np.array([[0.3, 0.4, np.sqrt(1 - 0.09 - 0.16)]], np.float32)
        dir_ = np.array([[0.0, 0.0, 1.0]], np.float32)
        value, dmean, dkappa = probe.log_pdf_grad(mean, np.array([kappa], np.float32), dir_)
        assert np.all(np.isfinite(value)), (kappa, value)
        assert np.all(np.isfinite(dmean)), (kappa, dmean)
        assert np.all(np.isfinite(dkappa)), (kappa, dkappa)

        norm = probe.log_normalizer(np.array([kappa], np.float32))
        assert np.all(np.isfinite(norm)), (kappa, norm)


# ─── N-lobe mixture smoke test (sample<->pdf consistency) ────────────────────────


def test_mixture_eval_and_sample_are_consistent(probe):
    rng = np.random.default_rng(51)
    lobes, stride = 4, 4
    params = np.zeros(lobes * stride, np.float32)
    for lobe in range(lobes):
        params[lobe * stride + 0] = rng.normal()  # logWeight
        params[lobe * stride + 1] = rng.uniform(1.0, 2.5)  # kappaRaw -> moderate kappa
        params[lobe * stride + 2] = rng.normal(scale=0.5)  # thetaRaw
        params[lobe * stride + 3] = rng.normal(scale=0.5)  # phiRaw

    n_eval = 50_000
    dirs = fibonacci_sphere(n_eval).astype(np.float32)
    logdensity = probe.mixture_eval(params, dirs)
    assert np.all(np.isfinite(logdensity))
    weight = 4.0 * np.pi / n_eval
    integral = float(np.exp(logdensity.astype(np.float64)).sum() * weight)
    assert abs(integral - 1.0) < 5e-2, f"mixture solid-angle integral={integral}"

    samples = probe.mixture_sample(params, 200_000)
    assert np.all(np.isfinite(samples))
    assert np.allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-3)

    # coarse sample<->density consistency: bin samples on a cylindrical-equal-area
    # grid (uniform in z=dir.z, not in theta=acos(z) -- dOmega = dz*dphi exactly,
    # so uniform-z x uniform-phi bins have EQUAL solid angle; uniform-theta bins
    # would not, since dOmega = sin(theta)*dtheta*dphi shrinks near the poles) and
    # compare to the mean evaluated density per bin.
    z = np.clip(samples[:, 2], -1, 1)
    phi = np.arctan2(samples[:, 1], samples[:, 0]) % (2 * np.pi)
    tb, pb = 8, 16
    t_idx = np.clip(((z + 1.0) * 0.5 * tb).astype(int), 0, tb - 1)
    p_idx = np.clip((phi / (2 * np.pi) * pb).astype(int), 0, pb - 1)
    observed = np.zeros((tb, pb))
    np.add.at(observed, (t_idx, p_idx), 1)

    eval_z = np.clip(dirs[:, 2], -1, 1)
    eval_phi = np.arctan2(dirs[:, 1], dirs[:, 0]) % (2 * np.pi)
    et_idx = np.clip(((eval_z + 1.0) * 0.5 * tb).astype(int), 0, tb - 1)
    ep_idx = np.clip((eval_phi / (2 * np.pi) * pb).astype(int), 0, pb - 1)
    density_sum = np.zeros((tb, pb))
    density_count = np.zeros((tb, pb))
    np.add.at(density_sum, (et_idx, ep_idx), np.exp(logdensity.astype(np.float64)))
    np.add.at(density_count, (et_idx, ep_idx), 1)
    mean_density = density_sum / np.maximum(density_count, 1)

    bin_solid_angle = 4.0 * np.pi / (tb * pb)
    expected = mean_density * bin_solid_angle * len(samples)
    mask = expected > 50  # only compare bins with enough expected mass to be stable
    rel = np.abs(observed[mask] - expected[mask]) / expected[mask]
    assert np.median(rel) < 0.35, f"median relative bin deviation {np.median(rel)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__, *sys.argv[1:]]))
