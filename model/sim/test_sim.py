"""Micro-tests for the FSR4 INT8 reference simulator (M2 gate)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import fsr4_sim as S

fails = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok: fails.append(name)

# 1. rounding = half away from zero
r = S.round_half_away(np.array([0.5, 1.5, 2.5, -0.5, -1.5, 0.4999999, -0.4999999], dtype=np.float32))
check("round half away from zero", list(r) == [1, 2, 3, -1, -2, 0, 0], str(list(r)))

# 2. conv2d vs manual (1x1 and 3x3 pad 1 zero border)
rng = np.random.default_rng(1)
x = rng.integers(-20, 20, (5, 6, 4), dtype=np.int8)
w = rng.integers(-10, 10, (3, 4, 3, 3), dtype=np.int8)
acc = S.conv2d_i8(x, w, stride=1, pad=1)
man = np.zeros((5, 6, 3), dtype=np.int64)
for oy in range(5):
    for ox in range(6):
        for f in range(3):
            s = 0
            for ky in range(3):
                for kx in range(3):
                    iy, ix = oy - 1 + ky, ox - 1 + kx
                    if 0 <= iy < 5 and 0 <= ix < 6:  # zero pad = tap skip
                        s += int(np.dot(x[iy, ix].astype(int), w[f,:, ky, kx].astype(int)))
            man[oy, ox, f] = s
check("conv2d 3x3 pad1 == manual (zero border)", np.array_equal(acc, man), f"maxdiff {np.abs(acc-man).max()}")

# 3. k2s2 stride 2 no pad, top-left corner
w2 = rng.integers(-5, 5, (2, 4, 2, 2), dtype=np.int8)
acc2 = S.conv2d_i8(x, w2, stride=2, pad=0)
s = sum(int(x[ky, kx, c]) * int(w2[0, c, ky, kx]) for ky in range(2) for kx in range(2) for c in range(4))
check("conv2d 2x2 s2 no-pad corner", acc2[0, 0, 0] == s)

# 4. ConvTranspose 2x2 pixel placement
xt = np.zeros((2, 2, 2), dtype=np.int8)  # [H=2,W=2,C=2]
xt[0, 0, 0] = 1   # pixel (0,0) ch0
xt[1, 1, 1] = 2   # pixel (1,1) ch1
xt[0, 1, 0] = 3   # pixel (0,1) ch0
wt = np.zeros((2, 1, 2, 2), dtype=np.int8)  # [cin,cout,ky,kx]
wt[0, 0, 0, 0] = 10   # ch0 -> tap (j=0,i=0)
wt[1, 0, 1, 1] = 7    # ch1 -> tap (j=1,i=1)
o = S.convtranspose2x2_i8(xt, wt)
ok = (o.shape == (4, 4, 1) and o[0, 0, 0] == 10 and o[3, 3, 0] == 14 and o[0, 2, 0] == 30
      and o[0, 1, 0] == 0 and o[1, 0, 0] == 0 and o[1, 1, 0] == 0)
check("ct2d placement out[2y+j,2x+i]", ok, str(o[:,:, 0]))

# 5. grouped tensor dequant halves
t = {"q": np.array([[[ -128, 127 ]]], dtype=np.int8), "s0": 0.01, "s1": 0.1}
d = S.dq(t)
check("grouped dequant halves", abs(d[0, 0, 0] + 1.28) < 1e-6 and abs(d[0, 0, 1] - 12.7) < 1e-5)

# 6. FNB passthrough half: internal concat keeps second-half bytes (uniform requant at same scale is identity)
q32 = rng.integers(-128, 128, (8, 8, 32), dtype=np.int8)
L = S.load_layers(); L = S.load_with_pass0_scale(L)
res, xq, C = S._fnb_core({"q": q32, "s0": 0.015390855260193348, "s1": 0.018884973600506783},
                          L, "encoder3_ResidualBlock_0", 0.015390855260193348, 0.018884973600506783,
                          0.014567121863365173, "32")
check("fnb core runs, shapes", res.shape == (8, 8, 32) and C == 32)

# 7. pass0: ch7 must not influence output
feats = rng.normal(0, 0.5, (16, 16, 8)).astype(np.float16)
f2 = feats.copy(); f2[..., 7] = 123.0
q_a = S.pass0(feats, L); q_b = S.pass0(f2, L)
check("pass0 ignores ch7 (weights zero)", np.array_equal(q_a["q"], q_b["q"]))
# pass0 manual check, single output cell + filter
w0 = L["pass0_weight_fkyxc"].astype(np.float16); b0 = L["pass0_bias"]
accm = np.float32(0)
for ky in range(2):
    for kx in range(2):
        for c in range(7):
            accm = np.float32(accm + np.float32(np.float32(w0[3, ky, kx, c]) * feats[ky, kx, c]))
qm = int(np.clip(np.floor(abs(accm + b0[3]) / L["_pass0_scale"] + 0.5) * np.sign(accm + b0[3]), -128, 127))
check("pass0 manual cell", int(q_a["q"][0, 0, 3]) == qm, f"{int(q_a['q'][0,0,3])} vs {qm}")

# 8. full graph determinism + output stats at two sizes
for (H, W) in [(32, 64), (540, 960)]:
    xg = {"q": rng.integers(-60, 60, (H, W, 16), dtype=np.int8), "s": S.S_IN}
    o1 = S.run_graph(xg, L); o2 = S.run_graph(xg, L)
    check(f"graph {W}x{H} -> ({2*H},{2*W},8) fp16, deterministic",
          o1.shape == (2*H, 2*W, 8) and o1.dtype == np.float16 and np.array_equal(o1, o2),
          f"range [{float(o1.min()):.2f},{float(o1.max()):.2f}]")

# 9. saturation sanity: extreme input doesn't crash and stays bounded
xg = {"q": np.full((16, 16, 16), 127, dtype=np.int8), "s": S.S_IN}
o = S.run_graph(xg, L)
check("graph bounded on saturated input", np.isfinite(o.astype(np.float32)).all())

print("\n" + ("ALL SIM MICRO-TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
sys.exit(1 if fails else 0)
