"""Run with examples/hggrid/.venv/bin/python tests/discretizedGMM/test_discretized_gmm.py."""
from pathlib import Path
from math import erf, sqrt
import numpy as np
import slangpy as spy

ROOT = Path(__file__).resolve().parents[2]


def reference(params):
    components = params.reshape(2, 5).astype(np.float64)
    weights = np.exp(components[:, 0] - components[:, 0].max())
    weights /= weights.sum()
    result = np.zeros((8, 8))
    for weight, component in zip(weights, components):
        axes = []
        for d in range(2):
            z = (np.arange(1, 8) / 8 - component[1 + d]) / np.exp(component[3 + d])
            cdf = np.array([0] + [0.5 * (1 + erf(v / sqrt(2))) for v in z] + [1])
            axes.append(np.diff(cdf))
        result += weight * np.outer(*axes)
    return result.ravel()


def main():
    device = spy.create_device(include_paths=[ROOT], enable_hot_reload=False)
    kernels = {}
    for name in ('evalMain', 'sampleMain'):
        program = device.load_program('tests/discretizedGMM/DiscretizedGMMTest.slang', [name])
        kernels[name] = device.create_compute_kernel(program)

    def run(name, params, count, size):
        inp = device.create_buffer(size=params.nbytes, usage=spy.BufferUsage.shader_resource, data=params)
        out = device.create_buffer(size=size * 4, usage=spy.BufferUsage.unordered_access)
        kernels[name].dispatch(thread_count=[count, 1, 1], vars={'gParams': inp, 'gOutputs': out})
        return out.to_numpy().view(np.float32).copy()

    cases = [
        [0.2, 0.3, 0.6, -1.5, -1.0, -0.4, 0.8, 0.2, -1.0, -1.8],
        [0, -0.2, 1.2, -1, -1, 0, 1.2, -0.2, -1, -1],
        [0, 0.4, 0.6, -4, -4, 0, 0.8, 0.2, -4, -4],
    ]
    for values in cases:
        params = np.array(values, dtype=np.float32)
        out = run('evalMain', params, 64, 832)
        expected = reference(params)
        np.testing.assert_allclose(out[:64], expected, atol=2e-6, rtol=2e-4)
        np.testing.assert_allclose(out[:64].sum(), 1, atol=2e-6)
        np.testing.assert_allclose(out[64:128], out[:64] * 64, atol=2e-5, rtol=2e-6)
        assert np.isfinite(out[128:768]).all()
        np.testing.assert_array_equal(out[768:], 0)
        if values is cases[0]:
            for j in range(10):
                plus, minus = params.astype(float), params.astype(float)
                plus[j] += 1e-4
                minus[j] -= 1e-4
                numerical = (-np.log(reference(plus)) + np.log(reference(minus))) / 2e-4
                np.testing.assert_allclose(out[128:768].reshape(64, 10)[:, j], numerical, atol=2e-3, rtol=2e-3)
        samples = run('sampleMain', params, 131072, 262144).reshape(-1, 2)
        assert ((samples >= 0) & (samples < 1)).all()
        cells = (samples * 8).astype(int)
        frequency = np.bincount(cells[:, 0] * 8 + cells[:, 1], minlength=64) / len(samples)
        assert np.all(np.abs(frequency - expected) < 6 * np.sqrt(expected * (1 - expected) / len(samples)) + 1e-4)
    print('Passed: reference masses, normalization, density, gradients, single-bin grid, sampling.')


if __name__ == '__main__':
    main()
