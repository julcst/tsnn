#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["slangpy>=0.42,<0.43", "numpy"]
# ///
"""GPU-timed benchmark for the HGGrid train/optimize step, mirroring the
test_warm_throughput methodology in tests/truncGMM/test_truncated_gmm.py:
GPU timestamp queries around each kernel, isolating device time from Python/
dispatch overhead and from readback.

Standalone (not routed through HGGridLearner/main.py) so it can be pointed at
either revision of the source with TSNN_HGGRID_SOURCE_ROOT and compared
directly, same idea as TSNN_GMM_SOURCE_ROOT in the truncGMM test.

Usage:
    UV_CACHE_DIR=/tmp/tsnn-uv-cache uv run benchmark.py [--steps N] [--label NAME]
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import slangpy as spy

HERE = Path(__file__).parent.absolute()
ROOT = HERE.parent.parent  # external/tsnn
OUT_DIR = HERE / "output"

GRID_DIM = 8
NUM_GAUSSIANS = 4
HIDDEN = 64
DEPTH = 3
NUM_CELLS = GRID_DIM * GRID_DIM
ENCODED_DIM = 2 * GRID_DIM
FINE_PARAM_COUNT = NUM_GAUSSIANS * 5
DEFAULT_BATCH_SIZE = 1 << 14
LOSS_SCALE = 128.0
GRAD_CLIP = 1.0


def _align4(x: int) -> int:
    return (x + 3) & ~3


def _mlp_byte_size(input_dim, hidden, depth, output_dim) -> int:
    byte_off = 0
    for l in range(depth + 1):
        in_size = input_dim if l == 0 else hidden
        out_size = output_dim if l == depth else hidden
        byte_off = _align4(byte_off + 2 * in_size * out_size)
        byte_off = _align4(byte_off + 2 * out_size)
    return byte_off


def make_buf(device, size_bytes, rw=False, data=None):
    usage = spy.BufferUsage.shader_resource
    if rw:
        usage |= spy.BufferUsage.unordered_access
    if data is not None:
        return device.create_buffer(size=size_bytes, usage=usage, data=data)
    return device.create_buffer(size=size_bytes, usage=usage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="Timed repeats per kernel")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE, help="trainMain thread count")
    ap.add_argument("--label", default=os.environ.get("TSNN_HGGRID_BENCH_LABEL", "working"))
    args = ap.parse_args()
    BATCH_SIZE = args.batch

    source_root = Path(os.environ.get("TSNN_HGGRID_SOURCE_ROOT", HERE))
    compiler_options = spy.SlangCompilerOptions(
        {
            "include_paths": [source_root, ROOT, ROOT / "TSNN", spy.SHADER_PATH],
            "defines": {
                "GRID_DIM": str(GRID_DIM),
                "NUM_GAUSSIANS": str(NUM_GAUSSIANS),
                "HG_HIDDEN": str(HIDDEN),
                "HG_DEPTH": str(DEPTH),
                "ACTIVATION": "1",
                "DEBUG_COUNTERS": "0",
            },
        }
    )
    device = spy.Device(compiler_options=compiler_options)
    print(f"GPU: {device.info.adapter_name} ({device.info.api_name})")

    def load(module, entry):
        return device.create_compute_kernel(
            device.load_program(module_name=module, entry_point_names=[entry])
        )

    t_compile = perf_counter()
    reset_k = load("Optimize.cs.slang", "resetMain")
    train_k = load("Train.cs.slang", "trainMain")
    optimize_k = load("Optimize.cs.slang", "optimizeMain")
    eval_nll_k = load("Infer.cs.slang", "evalNLLMain")
    compile_seconds = perf_counter() - t_compile
    print(f"Kernel load+compile: {compile_seconds:.2f} s")

    param_bytes = _mlp_byte_size(ENCODED_DIM, HIDDEN, DEPTH, FINE_PARAM_COUNT)
    param_elems = param_bytes // 2
    moment_bytes = param_elems * 4
    hist_bytes = NUM_CELLS * 4

    hist_logits = make_buf(device, hist_bytes, rw=True)
    hist_logit_grads = make_buf(device, hist_bytes, rw=True)
    hist_m1 = make_buf(device, hist_bytes, rw=True)
    hist_m2 = make_buf(device, hist_bytes, rw=True)
    params = make_buf(device, param_bytes, rw=True)
    params_master = make_buf(device, moment_bytes, rw=True)
    param_grads = make_buf(device, param_bytes, rw=True)
    m1 = make_buf(device, moment_bytes, rw=True)
    m2 = make_buf(device, moment_bytes, rw=True)

    n_reset = 256 * 8
    reset_k.dispatch(
        thread_count=[n_reset, 1, 1],
        vars={
            "gHistLogits": hist_logits,
            "gParams": params,
            "gParamsMaster": params_master,
            "CB": {
                "gLearningRate": 3e-3,
                "gCurrentStep": 1.0,
                "gDispatchThreadCount": n_reset,
                "gLossScale": LOSS_SCALE,
                "gGradClip": GRAD_CLIP,
                "gWeightDecay": 0.0,
            },
        },
    )
    for b in (hist_m1, hist_m2, m1, m2):
        enc = device.create_command_encoder()
        enc.clear_buffer(b)
        device.submit_command_buffer(enc.finish())
    device.wait()

    rng = np.random.default_rng(0)
    n_data = BATCH_SIZE * 4
    data = rng.random((n_data, 2), dtype=np.float32)
    data_buf = make_buf(device, int(data.nbytes), data=data.flatten())

    train_vars = {
        "gSamples": data_buf,
        "gHistLogits": hist_logits,
        "gHistLogitGrads": hist_logit_grads,
        "gParams": params,
        "gParamGrads": param_grads,
        "CB": {
            "gNumSamples": n_data,
            "gBatchSize": BATCH_SIZE,
            "gCurrentStep": 1,
            "gLossScale": LOSS_SCALE,
        },
    }
    optimize_vars = {
        "gHistLogits": hist_logits,
        "gHistLogitGrads": hist_logit_grads,
        "gHistMoments1": hist_m1,
        "gHistMoments2": hist_m2,
        "gParams": params,
        "gParamsMaster": params_master,
        "gParamGrads": param_grads,
        "gMoments1": m1,
        "gMoments2": m2,
        "CB": {
            "gLearningRate": 3e-3,
            "gCurrentStep": 1.0,
            "gDispatchThreadCount": n_reset,
            "gLossScale": LOSS_SCALE,
            "gGradClip": GRAD_CLIP,
            "gWeightDecay": 0.0,
        },
    }
    nll_out = make_buf(device, BATCH_SIZE * 4, rw=True)
    eval_nll_vars = {
        "gHistLogits": hist_logits,
        "gParams": params,
        "gTestSamples": data_buf,
        "gLogProbsOut": nll_out,
        "NLLCB": {"gNLLCount": BATCH_SIZE},
    }

    results = {}
    for name, kernel, vars_, thread_count in (
        ("evalNLLMain (forward only)", eval_nll_k, eval_nll_vars, [BATCH_SIZE, 1, 1]),
        ("trainMain", train_k, train_vars, [BATCH_SIZE, 1, 1]),
        ("optimizeMain", optimize_k, optimize_vars, [n_reset, 1, 1]),
    ):

        def zero_grads():
            enc = device.create_command_encoder()
            enc.clear_buffer(hist_logit_grads)
            enc.clear_buffer(param_grads)
            device.submit_command_buffer(enc.finish())

        for _ in range(args.warmup):
            if name == "trainMain":
                zero_grads()
            kernel.dispatch(thread_count=thread_count, vars=vars_)
        device.wait()

        queries = device.create_query_pool(spy.QueryType.timestamp, args.steps * 2)
        started = perf_counter()
        for i in range(args.steps):
            if name == "trainMain":
                zero_grads()
            kernel.dispatch(
                thread_count=thread_count,
                vars=vars_,
                query_pool=queries,
                query_index_before=2 * i,
                query_index_after=2 * i + 1,
            )
        device.wait()
        wall = perf_counter() - started
        stamps = np.array(queries.get_results(0, args.steps * 2), dtype=np.uint64).reshape(-1, 2)
        durations = (stamps[:, 1] - stamps[:, 0]) / device.info.timestamp_frequency
        median_ms = float(np.median(durations)) * 1000
        row = {
            "kernel": name,
            "gpu_median_ms": median_ms,
            "gpu_p95_ms": float(np.percentile(durations, 95)) * 1000,
            "wall_ms_per_call": wall / args.steps * 1000,
        }
        results[name] = row
        print(json.dumps(row))

    # End-to-end train+optimize step cost, and a full-run wall-clock projection.
    combined_median = results["trainMain"]["gpu_median_ms"] + results["optimizeMain"]["gpu_median_ms"]
    print(f"Combined GPU time per train+optimize step: {combined_median:.4f} ms")
    print(f"Projected wall time for 10,000 steps: {combined_median * 10 / 1000:.2f} s (GPU-bound estimate)")

    OUT_DIR.mkdir(exist_ok=True)
    report = {
        "adapter": device.info.adapter_name,
        "api": device.info.api_name,
        "compile_seconds": compile_seconds,
        "steps": args.steps,
        "batch_size": BATCH_SIZE,
        "combined_gpu_median_ms": combined_median,
        "projected_10k_steps_seconds": combined_median * 10 / 1000,
        "measurements": results,
    }
    (OUT_DIR / f"benchmark_{args.label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report written to {OUT_DIR / f'benchmark_{args.label}.json'}")


if __name__ == "__main__":
    main()
