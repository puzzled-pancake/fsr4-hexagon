"""QDQ clip-free FSR4: delete Clips+Relus from the float graph, wrap every
remaining compute edge with Q/DQ pinned to champion A8 scales.

Per-edge semantics after deletion:
- Conv->Clip(lo,hi)->Relu->Conv becomes Conv->DQ(s_clipout)->Relu->Q/DQ(s_reluout)->Conv.
  (Relu KEPT: plain Relu is proven-correct on HTP; Q after it carries the relu scale.)
- Add->Clip->... becomes Add->DQ(s_clipout)->...
- Slice->Clip->... becomes Slice->DQ(s_clipout)->...
- Edges never clipped (conv->add, convtranspose->add bias chain, add->convtranspose,
  final add_334->cast) get Q/DQ with the champion scale of that tensor.
- Graph input: Q/DQ with input_i8 scale (keeps app-facing float32 input like fsr4_f16).
- Weights: left float (converter handles as before).

Usage: build_qdq_clipfree.py <in_f16dum.onnx> <enc_info.txt> <out.onnx>
"""
import re, sys
import numpy as np, onnx
from onnx import helper, numpy_helper

src, encpath, dst = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(encpath, encoding='utf-8', errors='replace').read()
encs = {}
for mt in re.finditer(r'([\w\.]+) encoding: bitwidth \d+, min [^,]+, max [^,]+, scale ([^,]+), offset', txt):
    encs[mt.group(1)] = float(mt.group(2))
print(f"champion encodings: {len(encs)}")

m = onnx.load(src)
g = m.graph

# 1. delete Clips (rewire consumers to clip input) and Relus (rewire to relu input)
prod = {o: n for n in g.node for o in n.output}
def consumers(t):
    return [u for u in g.node if t in u.input]
nclip = nrelu = 0
for n in [n for n in g.node if n.op_type == 'Clip']:
    src_t = n.input[0]
    for u in consumers(n.output[0]):
        u.input[:] = [src_t if x == n.output[0] else x for x in u.input]
    g.node.remove(n); nclip += 1
# Relus are KEPT (deleting them leaks negatives into the next conv; the QDQ grid
# only emulates the hi-side clamp, never the zero floor). Only clips are deleted.
nrelu = sum(1 for n in g.node if n.op_type == 'Relu')
print(f"deleted {nclip} clips, kept {nrelu} relus; nodes now {len(g.node)}")

# drop orphaned clip-bound initializers
used = set()
for n in g.node:
    used.update(n.input)
for t in [t for t in g.initializer if t.name not in used]:
    g.initializer.remove(t)

# 2. wrap every compute edge with pinned Q/DQ.
# Edge tensor T (node output, non-initializer, non-graph-output, has champion scale):
# insert T -> Q(s) -> T_q -> DQ(s) -> T_dq; consumers use T_dq.
# Choose scale: champion scale of the CONSUMED tensor name if present else producer tensor.
# After deletion, consumer input names are producer outputs (e.g. conv_10 feeds conv directly);
# champion scale for conv_10 output exists (== old clip input enc). Use producer-output scale.
skip = set(i.name for i in g.initializer)
gouts = set(o.name for o in g.output)
edges = []
for n in g.node:
    for o in n.output:
        if o in skip or o in gouts: continue
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
    pi = next(i for i, n in enumerate(g.node) if t in n.output)
    c1, c2 = t+'_c1', t+'_c2'
    g.node.insert(pi+1, helper.make_node('Cast', [t], [c1], t+'_C1', to=1))
    g.node.insert(pi+2, helper.make_node('QuantizeLinear', [c1, sn, zn], [q_out], t+'_Q'))
    g.node.insert(pi+3, helper.make_node('DequantizeLinear', [q_out, sn, zn], [c2], t+'_DQ'))
    g.node.insert(pi+4, helper.make_node('Cast', [c2], [dq_out], t+'_C2', to=10))
    for u in users:
        u.input[:] = [dq_out if x == t else x for x in u.input]

from collections import Counter
print(Counter(n.op_type for n in g.node))
onnx.checker.check_model(m)
onnx.save(m, dst)
print("wrote", dst)
