#!/usr/bin/env python3
"""Generate the daemon's gpu_data seeds from your fsr4_quality_weights.npz.

Usage: python gen_seeds.py <fsr4_quality_weights.npz> <out_dir> [W] [H]

Emits:
  pass0_wb.raw   pass-0 conv weights + bias packed fp16 in the deployed
                 order; md5-checked against the 1080p contract
                 0915f25de428aaeee396701e3abfd341 (three candidate pack
                 orders are tried; a loud warning prints if none match,
                 since pre0_final.comp pins a transposed weight contract).
  hist.raw       zero boot seed, fp16 RGBA at output res
  rec_prev.raw   zero boot seed (the recurrent state is rewritten by the
                 first frames and the cut detector resets history at
                 startup; zeros boot fine but will not md5-match the
                 deployed image-derived 1080p seeds).

Dimensions default to 1920x1080 (16,588,800 B each); use 1280 720 for the
720p arm.
"""
import sys, hashlib
import numpy as np

npz, out = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 1920
H = int(sys.argv[4]) if len(sys.argv) > 4 else 1080
z = np.load(npz)
w = np.asarray(z["pass0_weight_fkyxc"], dtype=np.float16)
b = np.asarray(z["pass0_bias"], dtype=np.float16)
cand = {
    "fkyxc+bias": w.tobytes() + b.tobytes(),
    "fkyxc-transposed+bias": w.transpose(0, 2, 1, 3).copy().tobytes() + b.tobytes(),
    "ckxyf+bias": w.transpose(3, 2, 1, 0).copy().tobytes() + b.tobytes(),
}
target = "0915f25de428aaeee396701e3abfd341"
for name, blob in cand.items():
    if hashlib.md5(blob).hexdigest() == target:
        open(out + "/pass0_wb.raw", "wb").write(blob)
        print("pass0_wb.raw: contract MATCH via", name)
        break
else:
    open(out + "/pass0_wb.raw", "wb").write(cand["fkyxc+bias"])
    print("WARNING: no pack order matched md5", target, "- wrote fkyxc+bias; check pre0_final.comp weight-order header (deployed order was transposed)")
open(out + "/hist.raw", "wb").write(np.zeros((H, W, 4), dtype=np.float16).tobytes())
open(out + "/rec_prev.raw", "wb").write(np.zeros((H, W, 4), dtype=np.float16).tobytes())
print("hist.raw / rec_prev.raw: zero boot seeds", W, "x", H)
