#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["slangpy>=0.42,<0.44", "numpy", "scipy", "mpmath", "pytest"]
# ///
"""GPU accuracy/throughput microbenchmarks for TSNN/Mixtures/GaussianHelpers.slang.

For each scalar function (erf, erfcxPositive, expm1, log1p, log1mexp, erfinv,
erfinv_fast) this measures the library's current implementation and any
candidate alternative against a double-precision reference, on both the
function's "live" domain (what TruncatedGMM.slang actually feeds it) and a
wider "full" domain that also exercises edge cases and branch thresholds. It
also measures GPU throughput for both variants side by side.

This is a measurement tool, not a pass/fail gate: it prints a table and writes
tests/gaussianHelpers/output/<function>.json. The accept/reject call is made
by reading those numbers (see FINDING.md for the recorded decisions).

Run directly with:
    UV_CACHE_DIR=/tmp/tsnn-uv-cache uv run tests/gaussianHelpers/test_gaussian_helpers.py -s
"""

import os
import json
import zlib
from time import perf_counter
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
from scipy import special as sp

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent / "output"
THROUGHPUT_ITERS = 1024  # must match GaussianHelpersBench.slang's THROUGHPUT_ITERS
REPEATS = int(os.environ.get("TSNN_GH_BENCH_REPEATS", "200"))
THROUGHPUT_COUNT = int(os.environ.get("TSNN_GH_BENCH_COUNT", "65536"))


def buffer(device, values, rw=False):
    values = np.ascontiguousarray(values, dtype=np.float32)
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    return device.create_buffer(size=values.nbytes, usage=usage, data=values)


def ulp_distance(a_f32: np.ndarray, ref_f64: np.ndarray) -> np.ndarray:
    """Signed-magnitude-ordered integer bit distance between a and round(ref)."""
    a = np.asarray(a_f32, dtype=np.float32)
    b = np.asarray(ref_f64, dtype=np.float64).astype(np.float32)

    def order(x):
        i = x.view(np.int32).astype(np.int64)
        return np.where(i < 0, np.int64(0x80000000) - i, i + np.int64(0x80000000))

    finite = np.isfinite(a) & np.isfinite(b)
    dist = np.full(a.shape, np.inf)
    dist[finite] = np.abs(order(a[finite]) - order(b[finite])).astype(np.float64)
    # Both non-finite and equal (e.g. both -inf): treat as exact.
    both_nonfinite_equal = (~finite) & (
        (np.isnan(a) & np.isnan(b)) | (a == b)
    )
    dist[both_nonfinite_equal] = 0.0
    return dist


class Probe:
    def __init__(self):
        self.device = spy.create_device(include_paths=[ROOT], enable_hot_reload=False)
        self.kernels = {}
        names = []
        for fn in (
            "ErfCurrent", "ErfCandidate",
            "ErfcxCurrent",
            "Expm1Current", "Expm1Candidate", "Expm1Widertaylor",
            "Log1pCurrent", "Log1pCandidate", "Log1pWidertaylor",
            "Log1mexpCurrent", "Log1mexpLitlog2", "Log1mexpCandidatePrimitives", "Log1mexpWidertaylorPrimitives",
            "ErfinvCurrent", "ErfinvCandidate",
        ):
            names.append(f"accuracy{fn}Main")
            names.append(f"throughput{fn}Main")
        for name in names:
            program = self.device.load_program(
                module_name="tests/gaussianHelpers/GaussianHelpersBench.slang",
                entry_point_names=[name],
            )
            self.kernels[name] = self.device.create_compute_kernel(program)

    def accuracy(self, kernel_name, inputs: np.ndarray) -> np.ndarray:
        inp = buffer(self.device, inputs)
        out = buffer(self.device, np.zeros(len(inputs), np.float32), rw=True)
        self.kernels[kernel_name].dispatch(
            thread_count=[len(inputs), 1, 1],
            vars={"gInputs": inp, "gOutputs": out, "CB": {"gCount": len(inputs)}},
        )
        return out.to_numpy().view(np.float32).copy()

    def throughput(self, kernel_name, count: int, repeats: int = REPEATS):
        rng = np.random.default_rng(0)
        # Values don't matter for timing; kernel perturbs them internally anyway.
        inputs = buffer(self.device, rng.uniform(-0.4, -0.01, count))
        outputs = buffer(self.device, np.zeros(count, np.float32), rw=True)
        bindings = {"gInputs": inputs, "gOutputs": outputs, "CB": {"gCount": count}}
        kernel = self.kernels[kernel_name]
        for _ in range(3):
            kernel.dispatch(thread_count=[count, 1, 1], vars=bindings)
        self.device.wait()
        queries = self.device.create_query_pool(spy.QueryType.timestamp, repeats * 2)
        started = perf_counter()
        for i in range(repeats):
            kernel.dispatch(
                thread_count=[count, 1, 1],
                vars=bindings,
                query_pool=queries,
                query_index_before=2 * i,
                query_index_after=2 * i + 1,
            )
        self.device.wait()
        wall = perf_counter() - started
        stamps = np.array(queries.get_results(0, repeats * 2), dtype=np.uint64).reshape(-1, 2)
        durations = (stamps[:, 1] - stamps[:, 0]) / self.device.info.timestamp_frequency
        median = float(np.median(durations))
        evals = count * THROUGHPUT_ITERS
        return {
            "gpu_median_ms": median * 1000,
            "gpu_p95_ms": float(np.percentile(durations, 95)) * 1000,
            "gpu_evals_per_second": evals / median,
            "wall_evals_per_second": evals * repeats / wall,
        }


@pytest.fixture(scope="module")
def probe():
    return Probe()


# ─── Domains ────────────────────────────────────────────────────────────────
# "live" = what TruncatedGMM.slang actually feeds the function (see plan);
# "full" = wider sweep incl. edge cases, to catch regressions elsewhere.

def _mix(rng, *arrays):
    return np.concatenate(arrays).astype(np.float32)


def domain_erf(rng):
    live = rng.uniform(-6, 6, 20000)
    full = _mix(
        rng,
        rng.uniform(-1, 1, 20000),
        rng.uniform(-4, 4, 20000),
        np.array([0.0, -0.0, 1.0, -1.0, 1e-6, -1e-6, 3.9, -3.9]),
    )
    return live.astype(np.float32), full


def domain_erfcx(rng):
    live = np.abs(rng.uniform(0, 40, 20000))
    full = _mix(rng, rng.uniform(0, 5, 20000), rng.uniform(5, 60, 20000), np.array([0.0, 1e-6, 40.0]))
    return live.astype(np.float32), np.abs(full)


def domain_expm1(rng):
    log2 = np.log(2.0)
    live = -rng.uniform(0, log2, 20000)
    full = _mix(
        rng,
        rng.uniform(-log2, 0, 20000),
        rng.uniform(-5, 5, 20000),
        np.array([0.0, -1e-8, -log2, -log2 + 1e-6]),
    )
    return live.astype(np.float32), full


def domain_log1p(rng):
    live = -rng.uniform(0, 0.5, 20000)
    full = _mix(rng, rng.uniform(-0.5, 0, 20000), rng.uniform(-0.9, 9, 20000), np.array([0.0, -1e-8, -0.5]))
    return live.astype(np.float32), full


def domain_log1mexp(rng):
    # Clipped short of ~-88: exp(x) itself underflows to a float32 subnormal
    # there and to exactly 0 by ~-104, at which point log1p(-exp(x)) == 0 is
    # the correct float32 answer for *any* implementation -- comparing that
    # against a float64 reference that doesn't underflow is not a fair or
    # implementation-sensitive test, just the float32 pipeline's inherent
    # floor. See tests/gaussianHelpers/output/log1mexp_accuracy.json for the
    # value that flagged this (x ~= -112.68, both variants return 0.0f).
    live = -np.abs(rng.exponential(5, 15000)).clip(max=85.0)
    live = _mix(rng, live, -np.abs(rng.uniform(0, 1e-3, 5000)))
    full = _mix(
        rng,
        -np.abs(rng.exponential(10, 20000)).clip(max=85.0),
        np.array([0.0, -1e-8, -np.log(2.0), -np.log(2.0) + 1e-6, -50.0, -1e-30]),
    )
    return live.astype(np.float32), full


def domain_erfinv(rng):
    eps = 1e-6
    live = _mix(
        rng,
        rng.uniform(-0.99, 0.99, 15000),
        rng.uniform(-1 + eps, -0.99, 2500),
        rng.uniform(0.99, 1 - eps, 2500),
    )
    full = _mix(rng, live, np.array([0.0, 1 - 1e-7, -(1 - 1e-7)]))
    return live.astype(np.float32), full


DOMAINS = {
    "erf": (domain_erf, sp.erf),
    "erfcx": (domain_erfcx, sp.erfcx),
    "expm1": (domain_expm1, np.expm1),
    "log1p": (domain_log1p, np.log1p),
    # Mirror the implementation's own branch (not just log(-expm1(x)) uniformly):
    # for very negative x, 1-exp(x) rounds to exactly 1.0 even in float64, so
    # log(-expm1(x)) silently collapses to 0 there. log1p(-exp(x)) is the
    # numerically stable form for that tail at any precision.
    "log1mexp": (
        domain_log1mexp,
        lambda x: np.where(
            x.astype(np.float64) > -np.log(2.0),
            np.log(-np.expm1(x.astype(np.float64))),
            np.log1p(-np.exp(x.astype(np.float64))),
        ),
    ),
    "erfinv": (domain_erfinv, sp.erfinv),
}


def report(fn_label, variant_label, domain_label, gpu_out, ref, extra=None):
    d = ulp_distance(gpu_out, ref)
    finite = np.isfinite(d)
    row = {
        "function": fn_label,
        "variant": variant_label,
        "domain": domain_label,
        "max_ulp": float(np.max(d[finite])) if finite.any() else None,
        "rms_ulp": float(np.sqrt(np.mean(d[finite] ** 2))) if finite.any() else None,
        "nonfinite_mismatches": int((~finite).sum()),
        "n": len(gpu_out),
    }
    if extra:
        row.update(extra)
    print(json.dumps(row))
    return row


@pytest.mark.parametrize(
    "fn_label,current_kernel,candidate_kernels",
    [
        ("erf", "accuracyErfCurrentMain", {"candidate": "accuracyErfCandidateMain"}),
        ("erfcx", "accuracyErfcxCurrentMain", {}),
        (
            "expm1",
            "accuracyExpm1CurrentMain",
            {"candidate": "accuracyExpm1CandidateMain", "widertaylor": "accuracyExpm1WidertaylorMain"},
        ),
        (
            "log1p",
            "accuracyLog1pCurrentMain",
            {"candidate": "accuracyLog1pCandidateMain", "widertaylor": "accuracyLog1pWidertaylorMain"},
        ),
        (
            "log1mexp",
            "accuracyLog1mexpCurrentMain",
            {
                "litlog2": "accuracyLog1mexpLitlog2Main",
                "candidatePrimitives": "accuracyLog1mexpCandidatePrimitivesMain",
                "widertaylorPrimitives": "accuracyLog1mexpWidertaylorPrimitivesMain",
            },
        ),
        ("erfinv", "accuracyErfinvCurrentMain", {"candidate": "accuracyErfinvCandidateMain"}),
    ],
)
def test_accuracy(probe, fn_label, current_kernel, candidate_kernels):
    domain_fn, ref_fn = DOMAINS[fn_label]
    rng = np.random.default_rng(zlib.crc32(fn_label.encode()))
    live, full = domain_fn(rng)
    results = []
    for domain_label, values in (("live", live), ("full", full)):
        ref = ref_fn(values)
        for variant_label, kernel_name in [("current", current_kernel), *candidate_kernels.items()]:
            out = probe.accuracy(kernel_name, values)
            results.append(report(fn_label, variant_label, domain_label, out, ref))
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{fn_label}_accuracy.json").write_text(json.dumps(results, indent=2) + "\n")


# erfinv_fast (Winitzki) was unused dead code, benchmarked once as a data point
# (~1e9 ULP error -- see tests/gaussianHelpers/output/erfinv_fast_*.json and
# FINDING.md) and then deleted from GaussianHelpers.slang; no longer runnable.


@pytest.mark.parametrize(
    "fn_label,kernels",
    [
        ("erf", {"current": "throughputErfCurrentMain", "candidate": "throughputErfCandidateMain"}),
        ("erfcx", {"current": "throughputErfcxCurrentMain"}),
        (
            "expm1",
            {
                "current": "throughputExpm1CurrentMain",
                "candidate": "throughputExpm1CandidateMain",
                "widertaylor": "throughputExpm1WidertaylorMain",
            },
        ),
        (
            "log1p",
            {
                "current": "throughputLog1pCurrentMain",
                "candidate": "throughputLog1pCandidateMain",
                "widertaylor": "throughputLog1pWidertaylorMain",
            },
        ),
        (
            "log1mexp",
            {
                "current": "throughputLog1mexpCurrentMain",
                "litlog2": "throughputLog1mexpLitlog2Main",
                "candidatePrimitives": "throughputLog1mexpCandidatePrimitivesMain",
                "widertaylorPrimitives": "throughputLog1mexpWidertaylorPrimitivesMain",
            },
        ),
        ("erfinv", {"current": "throughputErfinvCurrentMain", "candidate": "throughputErfinvCandidateMain"}),
    ],
)
def test_throughput(probe, fn_label, kernels):
    print(f"GPU: {probe.device.info.adapter_name} ({probe.device.info.api_name})")
    results = []
    for variant_label, kernel_name in kernels.items():
        row = probe.throughput(kernel_name, THROUGHPUT_COUNT)
        row.update({"function": fn_label, "variant": variant_label, "batch": THROUGHPUT_COUNT})
        print(json.dumps(row))
        results.append(row)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{fn_label}_throughput.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    existing = [r for r in existing if r["variant"] not in kernels]
    (path).write_text(json.dumps(existing + results, indent=2) + "\n")


if __name__ == "__main__":
    import sys

    raise SystemExit(pytest.main(["-s", __file__, *sys.argv[1:]]))
