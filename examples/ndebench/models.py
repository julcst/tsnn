"""CPU-only configuration metadata, shared by the runner, plots and exports."""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODEL_SOURCE = (HERE / "Models.slangh").read_text()
GRID_CONFIGS = {
    name: tuple(map(int, values))
    for name, *values in re.findall(
        r"NDEBENCH_GRID\((\w+), (\d+), (\d+), (\d+), (\d+), (\d+)\)", MODEL_SOURCE
    )
}
ARCHITECTURES = tuple(re.findall(r"NDEBENCH_(?:MODEL|GRID)\((\w+)[,)]", MODEL_SOURCE))


def grid_layout(
    g: int, levels: int, hidden: int, depth: int, k: int
) -> list[tuple[int, int, int, int]]:
    """MLP blocks in HGGrid's layout order; K=0 omits the Gaussian head."""
    encoded = 2 * g * (levels if k else levels - 1)
    root = (1, hidden, depth, g * g)
    rest = (1 + encoded, hidden, depth, g * g)
    fine = (1 + encoded, hidden, depth, k * 5)
    return [root] + [rest] * (levels - 1) + ([fine] if k else [])


MLP_LAYOUTS = {
    "TMM": [(1, 32, 3, 80)],
    "DFN16": [(1, 16, 3, 16), (13, 16, 3, 16)],
    "DFL16": [(1, 16, 3, 16), (13, 16, 3, 16)],
    "NSFLinear": [(33, 32, 3, 16)] * 2,
    "NSFQuadratic": [(33, 32, 3, 33)] * 2,
    "NSFRQS": [(33, 32, 3, 47)] * 2,
    **{name: grid_layout(*config) for name, config in GRID_CONFIGS.items()},
}


LABELS = {
    "TMM": "TMM · K16",
    "DFN16": "DF-N · B16",
    "DFL16": "DF-L · B16",
    "NSFLinear": "NSF-L · B16",
    "NSFQuadratic": "NSF-Q · B16",
    "NSFRQS": "NSF-RQS · B16",
    **{
        name: f"{'HDF' if k == 0 else 'HGGrid'} · G{g} L{levels} H{h} D{d}"
        + (f" K{k}" if k else "")
        for name, (g, levels, h, d, k) in GRID_CONFIGS.items()
    },
}
