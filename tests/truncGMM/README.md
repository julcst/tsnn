# Truncated GMM validation and throughput

Run the full suite (including the 10,000-step Einstein fit):

```sh
MPLCONFIGDIR=/tmp/tsnn-mpl UV_CACHE_DIR=/tmp/tsnn-uv-cache \
  uv run tests/truncGMM/test_truncated_gmm.py
```

For just performance, append `-k warm_throughput`. Set
`TSNN_GMM_BENCH_REPEATS` to change the default 200 dispatches per measurement,
`TSNN_GMM_TEST_STEPS` for a shorter fit, and `TSNN_GMM_BENCH_LABEL` for a
separate JSON output filename. Results and the fit image go into `output/`.
`TSNN_GMM_SOURCE_ROOT` can point to an isolated source snapshot containing
`TSNN/` to compare implementations with the same test harness. Hot reload is
disabled so source changes cannot invalidate a running comparison.

Each kernel and batch size executes three warm-up dispatches and waits for
completion before timing. GPU durations use integer timestamp differences;
compilation, allocation, uploads, and readback are outside those durations.
The separately reported wall throughput includes Python dispatch, submission,
and the final GPU wait. Warm-up time is reported separately and can include
lazy compilation. The fit warms both gradient and Adam kernels, resets the
parameters and optimizer, then measures training including batch selection,
uploads, periodic readback, and the final wait. Training buffers are reused.

`evalMain` measures PDF evaluation. `nllGradMain` retains a differentiable
buffer-pointer control, while `nllArrayGradMain` accumulates derivatives in
local arrays and writes each parameter gradient once. Both use contiguous
writes across lanes (parameter-major storage). The fit uses the array path.
These timings describe this probe, not an end-to-end neural-network renderer.

The analytic component backward derivative is checked against double-precision
finite differences for all six inputs, including the PDF's exp(logPDF) chain
rule. Mixture checks cover all 80 parameters and invariance to a large common
logit offset. The full fit also checks normalization, half-input evaluation,
and sample distribution. Extreme-tail diagnostics deliberately report NaNs
without asserting that they are fixed: float erf differences can still cancel
to zero. No runtime NaN sanitization or tail branches were added.
