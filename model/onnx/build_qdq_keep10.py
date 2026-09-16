"""QDQ keep-10 FSR4 builder (parameterized re-write of an earlier inline build).

Deletes the 34 rail-trim Clips (<=1-LSB trims, producer scale == clip scale),
keeps the 10 slice-fed Clips as real ops (they change requant scales by
0.30-0.65x), and wraps every remaining compute edge with Q/DQ pinned to the
champion A8 scales. Optionally re-authors the graph boundary as native NHWC
(--nhwc-io) so the converter drops its app-facing NCHW<->NHWC permute ops
(input_i8_0231 / cast_335_0231 = ~25% of attributed op cycles).

Usage:
  build_qdq_keep10.py <in_f16dum.onnx> <enc_info.txt> <out.onnx> [--nhwc-io]
"""
import re, sys
import numpy as np, onnx
from onnx import helper, numpy_helper

args = [a for a in sys.argv[1:] if not a.startswith('--')]
nhwc_io = '--nhwc-io' in sys.argv
src, encpath, dst = args
KEEP = {'clip_79','clip_86','clip_146','clip_153','clip_179','clip_186',
        'clip_220','clip_227','clip_253','clip_260'}

txt = open(encpath, encoding='utf-8', errors='replace').read()
encs = {}
for mt in re.finditer(r'([\w\.]+) encoding: bitwidth \d+, min [^,]+, max [^,]+, scale ([^,]+), offset', txt):
    encs[mt.group(1)] = float(mt.group(2))
print(f"champion encodings: {len(encs)}")

m = onnx.load(src)
g = m.graph

# 1. delete non-kept clips (rewire consumers to clip input)
prod = {o: n for n in g.node for o in n.output}
def consumers(t):
    return [u for u in g.node if t in list(u.input)]
ndeleted = nkept = 0
for n in [n for n in g.node if n.op_type == 'Clip']:
    if n.output[0] in KEEP:
        nkept += 1
        continue
    src_t = n.input[0]
    for u in consumers(n.output[0]):
        u.input[:] = [src_t if x == n.output[0] else x for x in list(u.input)]
    g.node.remove(n); ndeleted += 1
nrelu = sum(1 for n in g.node if n.op_type == 'Relu')
print(f"deleted {ndeleted} clips, kept {nkept} + {nrelu} relus; nodes now {len(g.node)}")

# drop orphaned clip-bound initializers
used = set()
for n in g.node:
    used.update(n.input)
for t in [t for t in g.initializer if t.name not in used]:
    g.initializer.remove(t)

# 2. optional native-NHWC boundary (metadata transposes the converter should fold)
if nhwc_io:
    i0 = g.input[0]
    dims = [d.dim_value for d in i0.type.tensor_type.shape.dim]  # [1,C,H,W]
    N, C, H, W = dims
    i0.type.tensor_type.shape.dim[1].dim_value = H
    i0.type.tensor_type.shape.dim[2].dim_value = W
    i0.type.tensor_type.shape.dim[3].dim_value = C
    first = next(n for n in g.node if n.input and n.input[0] == i0.name)
    first.input[0] = 'input_nchw'
    g.node.insert(0, helper.make_node('Transpose', [i0.name], ['input_nchw'],
                                      'in_nhwc_fold', perm=[0, 3, 1, 2]))
    # output: cast_335 stays a node; graph output becomes its NHWC transpose
    gout = next(o for o in g.output if o.name == 'cast_335')
    odims = [d.dim_value for d in gout.type.tensor_type.shape.dim]  # [1,8,H,W]
    _, OC, OH, OW = odims
    gout.name = 'cast_335_nhwc'
    gout.type.tensor_type.shape.dim[1].dim_value = OH
    gout.type.tensor_type.shape.dim[2].dim_value = OW
    gout.type.tensor_type.shape.dim[3].dim_value = OC
    g.node.append(helper.make_node('Transpose', ['cast_335'], ['cast_335_nhwc'],
                                   'out_nhwc_fold', perm=[0, 2, 3, 1]))
    print("nhwc-io boundary applied")

# 3. wrap every compute edge with pinned Q/DQ (node outputs, non-graph-out, has
#    champion scale, has consumers). cast_335 excluded: it is the pre-output
#    cast; wrapping it would add a requant pass the graph never had.
skip = set(i.name for i in g.initializer)
gouts = set(o.name for o in g.output)
edges = []
for n in g.node:
    for o in n.output:
        if o in skip or o in gouts or o == 'cast_335': continue
        if o not in encs:
            print(f"WARN no champion scale for edge {o}; skipping")
            continue
        if consumers(o):
            edges.append(o)
print(f"wrapping {len(edges)} edges")
for t in edges:
    s = encs[t]
    users = consumers(t)
    q_out, dq_out, sn, zn = t+'_q', t+'_dq', t+'_s', t+'_z'
    g.initializer.append(numpy_helper.from_array(np.array(s, dtype=np.float32), sn))
    g.initializer.append(numpy_helper.from_array(np.array(0, dtype=np.int8), zn))
    pi = next(i for i, n in enumerate(g.node) if t in list(n.output))
    c1, c2 = t+'_c1', t+'_c2'
    g.node.insert(pi+1, helper.make_node('Cast', [t], [c1], t+'_C1', to=1))
    g.node.insert(pi+2, helper.make_node('QuantizeLinear', [c1, sn, zn], [q_out], t+'_Q'))
    g.node.insert(pi+3, helper.make_node('DequantizeLinear', [q_out, sn, zn], [c2], t+'_DQ'))
    g.node.insert(pi+4, helper.make_node('Cast', [c2], [dq_out], t+'_C2', to=10))
    for u in users:
        u.input[:] = [dq_out if x == t else x for x in list(u.input)]

from collections import Counter
print(Counter(n.op_type for n in g.node))
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst, "nhwc_io" if nhwc_io else "")
