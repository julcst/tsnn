#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["slangpy", "pillow", "numpy", "matplotlib"]
# ///

import slangpy as spy
import numpy as np
from pathlib import Path


def main():

    device = spy.create_device(
        include_paths=[
            Path(__file__).parent.absolute(),
            Path(__file__).parent.parent.parent.absolute(),
            Path(__file__).parent.parent.parent.absolute() / "TSNN",
        ]
    )

    program = device.load_program(module_name="test.slang", entry_point_names=["main"])
    kernel = device.create_compute_kernel(program=program)

    buffer_a = device.create_buffer(
        element_count=1024,
        resource_type_layout=kernel.reflection.main.a,
        usage=spy.BufferUsage.shader_resource,
        data=np.linspace(0, 1, 1024, dtype=np.float32),
    )
    buffer_b = device.create_buffer(
        element_count=1024,
        resource_type_layout=kernel.reflection.main.b,
        usage=spy.BufferUsage.shader_resource,
        data=np.linspace(1, 0, 1024, dtype=np.float32),
    )
    buffer_c = device.create_buffer(
        element_count=1024,
        resource_type_layout=kernel.reflection.main.c,
        usage=spy.BufferUsage.unordered_access,
    )

    print(kernel)

    kernel.dispatch(
        thread_count=[1024, 1, 1], N=1024, a=buffer_a, b=buffer_b, c=buffer_c
    )

    data = buffer_c.to_numpy().view(np.float32)
    print(data)
    assert np.all(data == 1.0)


if __name__ == "__main__":
    main()
