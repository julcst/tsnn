# Neural density estimation benchmark

Fit the luminance distribution of `../einstein.png` with nine density models,
using SlangPy 0.42 and cooperative-vector MLPs. Each method gets five seconds
of GPU training time; compilation, warmup, checkpoint evaluation and Python
submission overhead are excluded. The final checkpoint may exceed the budget
by one group of 64 updates. Training and inference use batches of 262,144.

```sh
cd examples/ndebench
uv sync
uv run benchmark.py
# CPU checks and a two-update GPU check of every configuration:
uv run benchmark.py --validate
uv run benchmark.py --smoke --output-directory output/smoke
# Replot a saved run without opening a GPU:
uv run plot.py output
```

`--architectures` selects comma-separated names from [Models.slangh](Models.slangh).
`--steps` overrides the time budget. `--seed` controls training and diagnostic samples; weight initialization is
deterministic. Results contain the image hash, package versions, adapter, checkpoints,
parameter counts and GPU timings, alongside raw full-resolution log-density arrays.

The configuration list also generates shader aliases and all five families of
kernel entry points. [models.py](models.py) reads it to derive grid MLP layouts;
the metadata kernel checks the host layout against the shader parameter count.
Add a grid configuration in one place rather than editing each kernel module.

[HGGrid.slang](architectures/HGGrid.slang) implements both HDF and HGGrid:
`G` is the width of each square histogram, `L` its cascade depth, `H` the hidden
MLP width, `D` the number of hidden layers, and `K` the number of leaf Gaussians.
The effective discrete resolution is `G^L` per axis. `K=0` gives uniform leaf
cells (HDF); `K>0` adds a truncated Gaussian mixture inside each cell (HGGrid).
Heads receive a one-hot encoding of the ancestor cells.

The retained grid configurations are:

| Configuration | Role |
|---|---|
| `HDF_G4L3_H32D2` | Narrow histogram heads, 64×64 leaves; cheaper updates in earlier sweeps. |
| `HDF_G8L2_H32D2` | Same leaf resolution with two wider heads; geometry comparison. |
| `HGGrid_G4L2_H16D3_K8` | 16×16 leaves with eight Gaussians per leaf; strong speed/quality tradeoff. |

DF-N/DF-L use 16 bins to compare directly with the 16-bin spline flows.
NSF-Q remains as a useful comparison even when dominated. Older deep/wide grid
variants, blob-conditioning sweeps, K=12/16 variants and the discretized-GMM cell
selector were removed after the saved sweeps showed little benefit to the
sampling-speed/variance tradeoff. Rankings describe a particular image, budget,
seed and device; configuration names do not encode a speed ranking.

`plot.py` writes convergence, sampling/variance Pareto and learned-density figures
as PNG and PDF, plus `summary.md`. Density panels share a linear color scale.
The sampling plot uses throughput and inverse Pearson χ², so top right is best.
Point colors and diagonal guides show expected variance after one millisecond:
`χ² / (throughput × 0.001 s)`. Throughput is in samples per second in this formula;
the color scale is in units of 10⁻⁹. This assumes independent samples from the fitted PDF.
KL and Pearson χ² are evaluated at all native-resolution texel centers. χ² is
`∫(p−q)²/q`, the importance-sampling variance for normalized densities; deterministic
quadrature avoids noise from estimating rare large ratios with a finite sample.
The raw report separately retains model-sample ratio statistics and the fraction
of non-finite ratios excluded from that estimate. Runs with non-finite ratios
are marked in the Pareto plot and excluded from its frontier. Inference throughput averages
100 warmed GPU dispatches after training.

`plot_real_layers.py` trains four representative models and plots their exact
first-stage densities, final densities and median-quantile sample paths.
`export.py` exports raw measurements to a report checkout via `--report-repo`.
README figures and the accompanying measured report live in [figures/](figures/);
large density arrays and scratch runs stay in the ignored `output/` directory.
