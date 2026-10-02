"""Device-defined cooperative-vector layout for Network.slang's five MLP layers."""

import numpy as np
import slangpy as spy


def static_layout_source(name, offsets, count):
    """Define a device layout and export it for an extern link-time type."""
    pairs = ",\n            ".join(
        f"uint2({int(weight)}u, {int(bias)}u)" for weight, bias in offsets
    )
    return f"""import TSNN.Utils.MLP;
struct {name}Impl : IMLPStaticLayout
{{
    static uint2 offset(uint layer)
    {{
        static const uint2 offsets[{len(offsets)}] = {{
            {pairs}
        }};
        return offsets[layer];
    }}
    static uint paramCount() {{ return {int(count)}u; }}
}}
export struct {name} : IMLPStaticLayout = {name}Impl;
"""


def static_layouts_source(training_offsets, training_count,
                         inference_offsets, inference_count):
    return (static_layout_source("TrainingOffsets", training_offsets, training_count)
            + static_layout_source("InferenceOffsets", inference_offsets, inference_count))


def compute_layout(device, input_size=32, hidden_size=64, hidden_layers=4, output_size=3,
                   matrix_layout=spy.CoopVecMatrixLayout.training_optimal):
    """Return uint2 byte offsets and the padded fp16 storage element count.

    Weight sizes are opaque device data; only bias vectors have a known size.
    Keep this shape sequence synchronized with Network.slang.
    """
    offsets = []
    byte_offset = 0
    for layer in range(hidden_layers + 1):
        cols = input_size if layer == 0 else hidden_size
        rows = output_size if layer == hidden_layers else hidden_size
        weight_offset = byte_offset
        matrix_bytes = device.get_coop_vec_matrix_size(
            rows, cols, matrix_layout, spy.DataType.float16
        )
        # Eight-byte alignment also keeps the optimizer's half4 accesses in bounds.
        bias_offset = (byte_offset + matrix_bytes + 7) & ~7
        byte_offset = (bias_offset + rows * 2 + 7) & ~7
        offsets.append((weight_offset, bias_offset))
    return np.asarray(offsets, dtype=np.uint32), byte_offset // 2


class InferenceWeights:
    """GPU conversion of training weights, plus plain bias-vector copies."""

    def __init__(self, device, source, source_offsets, input_size, hidden_size,
                 hidden_layers, output_size):
        self.source = source
        self.offsets, self.param_count = compute_layout(
            device, input_size, hidden_size, hidden_layers, output_size,
            spy.CoopVecMatrixLayout.inferencing_optimal,
        )
        self.params = device.create_buffer(
            size=self.param_count * 2,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            | spy.BufferUsage.copy_destination,
        )
        self.src_descs = []
        self.dst_descs = []
        self.bias_copies = []
        for layer in range(hidden_layers + 1):
            rows = output_size if layer == hidden_layers else hidden_size
            cols = input_size if layer == 0 else hidden_size
            for descs, offsets, matrix_layout in (
                (self.src_descs, source_offsets, spy.CoopVecMatrixLayout.training_optimal),
                (self.dst_descs, self.offsets, spy.CoopVecMatrixLayout.inferencing_optimal),
            ):
                descs.append(spy.CoopVecMatrixDesc({
                    "rows": rows, "cols": cols, "element_type": spy.DataType.float16,
                    "layout": matrix_layout, "offset": int(offsets[layer, 0]),
                    "size": device.get_coop_vec_matrix_size(
                        rows, cols, matrix_layout, spy.DataType.float16
                    ),
                }))
            self.bias_copies.append((
                int(self.offsets[layer, 1]), int(source_offsets[layer, 1]),
                (rows * 2 + 3) & ~3,
            ))

    def convert(self, encoder):
        encoder.convert_coop_vec_matrices(
            self.params, self.dst_descs, self.source, self.src_descs
        )
        for dst_offset, src_offset, size in self.bias_copies:
            encoder.copy_buffer(self.params, dst_offset, self.source, src_offset, size)
