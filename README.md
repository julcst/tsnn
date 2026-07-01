# TSNN
Deep Learning framework for [Slang](https://github.com/shader-slang/slang), aimed to simplify neural texture compression, neural radiance caching, neural importance sampling, etc.
Inspired by [tcnn](https://github.com/nvlabs/tiny-cuda-nn) and [RTXNS](https://github.com/NVIDIA-RTX/RTXNS).
This library has no dependencies other than Slang and has explicit support for [Falcor](https://github.com/nvidiagameworks/falcor) and [slangpy](https://github.com/shader-slang/slangpy)

## Features
* MLPs
* Neural Spline Flows
* Common loss functions (L1/L2, Relative L1/L2, Relative L2 Luminance)
* Common activation functions (ReLU, Swish, LeakyReLU, etc)

### Optimizers
* Adam/AdamW

### Encodings
* Hash Grid
* Spherical Harmonics
* One Blob

## Documentation
The framework is built around three manually-invoked fully-fused kernel invocations:
1. Training
2. Optimization
3. Inference
The implementation of these kernels is highly problem-specific, so this repo only provides utility functions/classes.

## Examples
For examples using [slangpy](https://github.com/shader-slang/slangpy) see the [texture compression](examples/image_learn) and the [neural density estimation](examples/nde) example.

## Falcor Usage
To use this library in Falcor just add as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) and list it in `external/CMakeLists.txt`:
```CMake
...
add_subdirectory(tsnn)
```
The shader library will be added automatically.
