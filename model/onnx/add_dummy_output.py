"""Append a tiny dummy second output (Slice of the final output's input tensor).

QAIRT graph-prep misplan: single-output DLCs of this network misplan on the HTP;
one extra tiny output fixes compilation/execution (notes "RESOLUTION"). The
dummy slices [1,16,1,1] from the pre-Cast 1080p tensor (64 bytes f32 / 32 f16).

Usage: python add_dummy_output.py <in.onnx> <out.onnx>
"""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
m = onnx.load(str(src))
g = m.graph
assert len(g.output) == 1, f"expected single output, have {[o.name for o in g.output]}"
out0 = g.output[0]
prod = next(n for n in g.node if out0.name in n.output)
assert prod.op_type == "Cast", f"final node is {prod.op_type}, expected Cast"
x_in = prod.input[0]

inf = onnx.shape_inference.infer_shapes(m)
vi = next((v for v in inf.graph.value_info if v.name == x_in), None)
assert vi is not None, f"no shape info for {x_in}"
dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
assert len(dims) == 4 and dims[0] == 1, f"unexpected tail shape {dims}"
dt = vi.type.tensor_type.elem_type
c_end = min(16, dims[1])

g.initializer.extend([
 helper.make_tensor("dm_s", TP.INT64, [4], [0, 0, 0, 0]),
 helper.make_tensor("dm_e", TP.INT64, [4], [1, c_end, 1, 1]),
 helper.make_tensor("dm_a", TP.INT64, [4], [0, 1, 2, 3])])
g.node.append(helper.make_node(
 "Slice", [x_in, "dm_s", "dm_e", "dm_a"],
 ["dummy_out"], name="dm_slice"))
g.output.append(helper.make_tensor_value_info("dummy_out", dt, [1, c_end, 1, 1]))
onnx.checker.check_model(m)
onnx.save(m, str(dst))
nbytes = c_end * {TP.FLOAT: 4, TP.FLOAT16: 2}[dt]
print(f"wrote {dst}: outputs {[o.name for o in g.output]}, dummy {dt}={nbytes} bytes")
