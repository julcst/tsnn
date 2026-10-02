# TSNN
Neural density estimation helpers for [Slang](https://github.com/shader-slang/slang), aimed to simplify neural texture compression, neural radiance caching, neural importance sampling, etc., inspired by [tcnn](https://github.com/nvlabs/tiny-cuda-nn) and [RTXNS](https://github.com/NVIDIA-RTX/RTXNS).

This library only depends on Slang and has explicit support for [Falcor](https://github.com/nvidiagameworks/falcor) and [slangpy](https://github.com/shader-slang/slangpy)

## Advantages:
* **Cross-Platform Compatibility**: Unlike tcnn, TSNN is not tied to CUDA: Slang cross-compiles the same source to SPIR-V, DXIL, and CUDA, so the same network runs on any Vulkan or D3D12 GPU, not just NVIDIA hardware.
* **Full Kernel Fusion**: Training, optimization, and inference are each a single fused compute kernel, minimizing host/device overhead
* **Hardware Acceleration**: Natively uses Cooperative Vector Operations
* **Flexibility**: Slangs Auto-Diff system allows for arbitrary architectures and efficiently calculates gradients using Source Code Transformation

> [!WARNING]  
> Does not work for slangpy >= 0.43, because Slang v2026.12 makes CoopVec differentiable themselves, recent Slang also has [their own neural network module](https://github.com/shader-slang/slang/tree/master/source/standard-modules/neural).

## Features

### Modules
* MLPs
* Neural Spline Flows (Affine, Linear, Quadratic, RQS, Circular RQS)
* Neural Mixture Models (Histogram, truncated Gaussian, von Mises-Fisher)
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

### Image Compression vs tiny-cuda-nn
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
| Training (fwd+bwd) | 488.6 us/step | 520.0 us/step | 646.5 us/step | 1.06x | 1.32x |
| Optimizer step | 504.0 us/step | 551.1 us/step | 549.9 us/step | 1.09x | 1.09x |
| Layout conversion | 8.5 us/pass | — | — | — | — |
| Inference (converted weights) | 81.7 us/pass | 115.7 us/pass | 197.4 us/pass | 1.42x | 2.42x |

Averaged over 5,000 training steps and 200 full-image (512x512) inference
passes, batch 16,384, RTX 5070 Ti (2026-10-01). Ratios divide tcnn time by
TSNN time (>1x = TSNN faster). Final PSNR: TSNN 53.24 dB, tcnn JIT 52.84 dB,
tcnn plain 53.04 dB. Conversion includes matrix conversion and bias copies;
converted weights are reused for inference.

## Examples
For examples using [slangpy](https://github.com/shader-slang/slangpy) see the [texture compression](examples/image_learn), [neural density estimation and hierarchical Gaussian mixtures](examples/ndebench) examples.

## Falcor Usage
To use this library in Falcor just add it as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) and list it in `external/CMakeLists.txt`:
```CMake
...
add_subdirectory(tsnn)
```
The shader library will be added automatically.
