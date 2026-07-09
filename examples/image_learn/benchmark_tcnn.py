#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "tqdm",
#     "pillow",
#     "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
# ]
#
# [tool.uv.extra-build-dependencies]
# tinycudann = ["torch", "setuptools<81"]
# ///
"""Benchmark harness for tiny-cuda-nn (PyTorch bindings), configured to match
TSNN's examples/image_learn/Network.slang exactly: same hash-grid encoding,
same MLP shape, same Adam hyperparameters, same relative-L2-luminance loss,
same batch size / step count. Compare against benchmark.py's output.

Note: tcnn's FullyFusedMLP kernel only supports ReLU hidden activations (its
fastest path); TSNN uses LeakyReLU. This is the standard "fastest available"
config on each side, not a bit-identical activation function -- see README.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import tinycudann as tcnn
from tqdm import trange

# Shared across examples/image_learn and examples/nde.
DEFAULT_IMAGE = Path(__file__).parent.parent / "einstein.png"


def load_image(path: str, res: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((res, res), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


HASH_LEVELS = 16
HASH_FEATURES = 2
HASH_LOG2_TABLE = 19
HASH_BASE_RES = 16
HASH_SCALE = 1.5

HIDDEN_SIZE = 64
HIDDEN_LAYERS = 4
OUTPUT_SIZE = 3

BATCH_SIZE = 1 << 14
STEPS = 5_000
LR = 1e-3
EVAL_EVERY = 100
RESOLUTION = 512


def relative_l2_luminance_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Mirrors TSNN::RelativeL2Luminance -- Rec.709 luminance of the *prediction*
    # (detached) forms the per-pixel denominator, broadcast over all 3 channels.
    lum = (pred.detach() * pred.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1, keepdim=True)
    denom = lum * lum + 1e-2
    return ((pred - target) ** 2 / denom).mean()


def build_model(device, jit_fusion: bool = False):
    encoding_config = {
        "otype": "HashGrid",
        "n_levels": HASH_LEVELS,
        "n_features_per_level": HASH_FEATURES,
        "log2_hashmap_size": HASH_LOG2_TABLE,
        "base_resolution": HASH_BASE_RES,
        "per_level_scale": HASH_SCALE,
    }
    network_config = {
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": HIDDEN_SIZE,
        "n_hidden_layers": HIDDEN_LAYERS,
    }
    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=2,
        n_output_dims=OUTPUT_SIZE,
        encoding_config=encoding_config,
        network_config=network_config,
    ).to(device)
    if jit_fusion:
        if not tcnn.supports_jit_fusion():
            raise RuntimeError("--jit requested but tcnn.supports_jit_fusion() is False on this build/GPU")
        model.jit_fusion = True
    return model


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def run_inference_benchmark(model, eval_uv: torch.Tensor, H: int, W: int, iters: int, warmup: int) -> dict:
    """Dedicated inference-only GPU throughput, timed with CUDA events (device
    time), run after training completes: this example trains once then infers
    from the fixed result, unlike an online setup (e.g. NRC) where training
    and inference are interleaved every frame -- so inference is benchmarked
    as its own back-to-back loop, not mixed into the training loop's timing."""
    with torch.no_grad():
        for _ in range(warmup):
            model(eval_uv).float()
        torch.cuda.synchronize()

        events = []
        for _ in range(iters):
            e_start = torch.cuda.Event(enable_timing=True)
            e_end = torch.cuda.Event(enable_timing=True)
            e_start.record()
            model(eval_uv).float()
            e_end.record()
            events.append((e_start, e_end))

        torch.cuda.synchronize()
        gpu_time = sum(e_start.elapsed_time(e_end) / 1000.0 for e_start, e_end in events)

    pixels = H * W * iters
    return {
        "iters": iters,
        "gpu_time_s": gpu_time,
        "mpixels_per_s": pixels / gpu_time / 1e6,
    }


def run(target: np.ndarray, device, steps: int, lr: float, eval_every: int, warmup: int,
        jit_fusion: bool = False) -> dict:
    H, W = target.shape[:2]
    target_t = torch.from_numpy(target).to(device)  # [H, W, 3]

    model = build_model(device, jit_fusion)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    param_count = sum(p.numel() for p in model.parameters())

    # Full-image UV grid for periodic PSNR eval.
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    eval_uv = torch.stack([(xx + 0.5) / W, (yy + 0.5) / H], dim=-1).reshape(-1, 2).float()

    def sample_batch():
        pixel = torch.randint(0, H * W, (BATCH_SIZE,), device=device)
        py, px = pixel // W, pixel % W
        uv = torch.stack([(px + 0.5) / W, (py + 0.5) / H], dim=-1).float()
        return uv, target_t[py, px]

    # tcnn's core CUDA kernels (the fused MLP itself) are precompiled
    # ahead-of-time -- jit_fusion (--jit) instead lets PyTorch JIT-fuse the
    # surrounding elementwise glue ops (padding/casts) into the fused kernel
    # call, which is the closer analogue to TSNN not needing separate glue
    # dispatches. Independently, the first CUDA call still pays for
    # context/allocator/cuBLAS-handle init -- run a few throwaway steps
    # before timing, then reinit model+optimizer so the timed run trains
    # `steps` iterations from scratch (same convention as benchmark.py /
    # scripts/nis_profile.py's WARMUP).
    for _ in range(warmup):
        uv, tgt = sample_batch()
        pred = model(uv).float()
        loss = relative_l2_luminance_loss(pred, tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model(eval_uv).float()
    torch.cuda.synchronize()

    model = build_model(device, jit_fusion)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # GPU device time for each stage separately, bracketed with its own CUDA
    # event pair per step, recorded without synchronizing so launches stay
    # async: "train" is sample_batch + forward + zero_grad + backward (grad
    # computation, comparable to TSNN's fused train_step), "optimize" is
    # optimizer.step() (comparable to TSNN's fused optimize_step). Both
    # exclude the periodic full-image eval; inference itself is benchmarked
    # separately after training, below.
    history = []
    torch.cuda.synchronize()
    train_gpu_time = 0.0
    optimize_gpu_time = 0.0

    t = trange(1, steps + 1, desc="tcnn")
    step = 1
    while step <= steps:
        block_end = min(step + eval_every - 1, steps)
        events = []

        for s in range(step, block_end + 1):
            e_train_start = torch.cuda.Event(enable_timing=True)
            e_train_end = torch.cuda.Event(enable_timing=True)
            e_opt_end = torch.cuda.Event(enable_timing=True)

            e_train_start.record()
            uv, tgt = sample_batch()
            pred = model(uv).float()
            loss = relative_l2_luminance_loss(pred, tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            e_train_end.record()

            optimizer.step()
            e_opt_end.record()

            events.append((e_train_start, e_train_end, e_opt_end))

        torch.cuda.synchronize()
        for e_train_start, e_train_end, e_opt_end in events:
            train_gpu_time += e_train_start.elapsed_time(e_train_end) / 1000.0
            optimize_gpu_time += e_train_end.elapsed_time(e_opt_end) / 1000.0
        t.update(block_end - step + 1)

        with torch.no_grad():
            pred_full = model(eval_uv).float().reshape(H, W, 3).clamp(0, 1)
        m = mse(pred_full.cpu().numpy(), target)  # .cpu() already blocks until the GPU is done
        psnr = -10 * math.log10(max(m, 1e-10))
        history.append({
            "step": block_end,
            "train_gpu_time_s": train_gpu_time,
            "optimize_gpu_time_s": optimize_gpu_time,
            "psnr": psnr,
        })
        t.set_postfix({"PSNR (dB)": f"{psnr:.2f}"})

        step = block_end + 1
    t.close()

    inference = run_inference_benchmark(model, eval_uv, H, W, iters=200, warmup=warmup)
    return {
        "backend": "tcnn",
        "gpu": torch.cuda.get_device_name(device),
        "param_count": param_count,
        "config": {
            "hash_levels": HASH_LEVELS,
            "hash_features": HASH_FEATURES,
            "hash_log2_table": HASH_LOG2_TABLE,
            "hash_base_res": HASH_BASE_RES,
            "hash_scale": HASH_SCALE,
            "hidden_size": HIDDEN_SIZE,
            "hidden_layers": HIDDEN_LAYERS,
            "activation": "ReLU",
            "network_otype": "FullyFusedMLP",
            "batch_size": BATCH_SIZE,
            "steps": steps,
            "lr": lr,
            "resolution": H,
            "jit_fusion": jit_fusion,
        },
        "history": history,
        "train_gpu_time_s": train_gpu_time,
        "optimize_gpu_time_s": optimize_gpu_time,
        "final_psnr": history[-1]["psnr"] if history else None,
        "inference": inference,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Input image path")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--res", type=int, default=RESOLUTION)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--warmup", type=int, default=10, help="Throwaway steps before timing (CUDA context/allocator init)")
    parser.add_argument("--jit", action="store_true", help="Enable tcnn's jit_fusion (requires tcnn.supports_jit_fusion())")
    parser.add_argument("--out", default="bench_tcnn.json")
    args = parser.parse_args()

    target = load_image(args.image, args.res)
    print(f"Target: {args.image} ({target.shape[1]}x{target.shape[0]})")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    if args.jit:
        print(f"JIT fusion: requested (supported={tcnn.supports_jit_fusion()})")

    result = run(target, device, args.steps, args.lr, args.eval_every, args.warmup, jit_fusion=args.jit)
    print(f"Parameters: {result['param_count']:,}")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[train]    gpu_time={result['train_gpu_time_s']:.3f}s")
    print(f"[optimize] gpu_time={result['optimize_gpu_time_s']:.3f}s")
    print(f"[psnr]     final={result['final_psnr']:.2f}dB")
    inf = result["inference"]
    print(f"[infer]    gpu_time={inf['gpu_time_s']:.3f}s ({inf['iters']} full-image passes, "
          f"{inf['mpixels_per_s']:.1f} Mpixels/s)")
    print(f"Result saved -> {args.out}")


if __name__ == "__main__":
    main()
