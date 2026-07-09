# TSNN
Deep Learning framework for [Slang](https://github.com/shader-slang/slang), aimed to simplify neural texture compression, neural radiance caching, neural importance sampling, etc., inspired by [tcnn](https://github.com/nvlabs/tiny-cuda-nn) and [RTXNS](https://github.com/NVIDIA-RTX/RTXNS).

This library only depends on Slang and has explicit support for [Falcor](https://github.com/nvidiagameworks/falcor) and [slangpy](https://github.com/shader-slang/slangpy)

## Advantages:
* **Cross-Platform Compatibility**: Unlike tcnn, TSNN is not tied to CUDA: Slang cross-compiles the same source to SPIR-V, DXIL, and CUDA, so the same network runs on any Vulkan or D3D12 GPU, not just NVIDIA hardware.
* **Full Kernel Fusion**: Training, optimization, and inference are each a single fused compute kernel, minimizing host/device overhead
* **Hardware Acceleration**: Natively uses Cooperative Vector Operations
* **Flexibility**: Slangs Auto-Diff system allows for arbitrary architectures and efficiently calculates gradients using Source Code Transformation

## Features

### Modules
* MLPs
* Neural Spline Flows
* Common loss functions (L1/L2, Relative L1/L2, Relative L2 Luminance)
* Common activation functions (ReLU, Swish, LeakyReLU, etc)

### Optimizers
* Adam/AdamW

### Encodings
* Hash Grid 2D/3D
* Spherical Harmonics
* One Blob

## Documentation
The framework is built around three manually-invoked fully-fused kernel invocations:
1. Training
2. Optimization
3. Inference

The implementation of these kernels is highly problem-specific, so this repo only provides utility functions/classes.

## Benchmarks
[`examples/image_learn`](examples/image_learn) fits a hash-grid + MLP to
[`examples/einstein.png`](examples/einstein.png) at 512x512 and compares
against an equivalent [tiny-cuda-nn](https://github.com/nvlabs/tiny-cuda-nn)
config (`benchmark_tcnn.py`): same encoding, network shape, batch size, and
Adam hyperparameters. Both scripts warm up (JIT-compile TSNN's Slang
pipelines / pay tcnn's one-time CUDA context and allocator init) before the
timed run.

Each of TSNN's three kernel invocations is timed separately with GPU
timestamp queries (CUDA events for tcnn), isolating device execution time
from host/Python overhead, and reported per iteration. Training and
optimizer are timed every step of the main training run; inference is timed
as a separate, dedicated back-to-back pass after training completes, since
this example trains once then infers from the fixed result (unlike an
online setup such as NRC, where training and inference interleave every
frame):

| Kernel | TSNN (Slang) | tiny-cuda-nn (JIT) | tiny-cuda-nn | vs JIT | vs plain |
|---|---|---|---|---|---|
| Training (fwd+bwd) | 472 us/step | 526 us/step | 644 us/step | 1.11x | 1.37x |
| Optimizer step | 508 us/step | 560 us/step | 558 us/step | 1.10x | 1.10x |
| Inference | 77 us/pass | 115 us/pass | 195 us/pass | 1.49x | 2.52x |

Averaged over 5,000 training steps and 200 full-image (512x512) inference
passes, batch 16,384, RTX 5070 Ti; "vs JIT"/"vs plain" are each tcnn column's
time divided by TSNN's (>1x = TSNN faster). The JIT column enables tcnn's
`jit_fusion` (`model.jit_fusion = True`, gated on
`tcnn.supports_jit_fusion()`), which lets PyTorch JIT-fuse the elementwise
glue around the fused kernel call instead of dispatching it separately —
closer to what TSNN's fully-fused kernel does natively, and it closes a
good chunk of the gap on every kernel. TSNN still wins across the board;
tcnn reaches a higher final PSNR at this step count (JIT: 52.9 dB, plain:
53.5 dB, vs. TSNN's 47.4 dB) — its hash grid adaptively shrinks unused table
entries at coarse levels, where TSNN currently always allocates the full
table per level.

Reproduce from `examples/image_learn` with `uv run --script benchmark.py`,
`uv run --script benchmark_tcnn.py [--jit]`, and
`uv run --script compare.py --markdown [--tcnn-jit bench_tcnn_jit.json]`.

## Examples
For examples using [slangpy](https://github.com/shader-slang/slangpy) see the [texture compression](examples/image_learn) and the [neural density estimation](examples/nde) example.

## Falcor Usage
To use this library in Falcor just add it as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) and list it in `external/CMakeLists.txt`:
```CMake
...
add_subdirectory(tsnn)
```
The shader library will be added automatically.
