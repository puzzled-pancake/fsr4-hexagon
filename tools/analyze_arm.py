#!/usr/bin/env python3
# energy-matrix analysis: integrates the fuel-gauge trace over exactly
# the benchmark's elevated-load segment. Window start = t0 (pad-A trigger =
# bench start, load-in included — identical across arms); window end = first
# sustained drop below the power threshold AFTER the load ramps up (the
# bench->results transition). Same objective rule for every arm.
# Usage: py -3 analyze_arm.py samples_<label>.csv [fps=30] [thr_W=7.5] [drop_s=4]
import csv, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'samples_armA.csv'
fps = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 7.5
drop_s = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

rows = []
with open(path) as f:
    r = csv.reader(f); next(r)
    for t, v, i in r:
        rows.append((float(t), float(v) * 1e-6 * float(i) * 1e-6))  # W (<0 = discharge)

t0 = rows[0][0]
P = [(t - t0, -p) for t, p in rows]  # positive watts
E_all = sum((P[k][1] + P[k-1][1]) / 2 * (P[k][0] - P[k-1][0]) for k in range(1, len(P)))
T_all = P[-1][0]

# end of bench: first sustained-below-threshold span after power has exceeded it
hi = next((k for k, (_, p) in enumerate(P) if p >= thr), None)
end_t = None
if hi is not None:
    k = hi
    while k < len(P):
        if P[k][1] < thr:
            j = k
            while j < len(P) and P[j][0] - P[k][0] < drop_s:
                if P[j][1] >= thr:
                    k = j; break
                j += 1
            else:
                end_t = P[k][0]; break
            k = j
        k += 1
if end_t is None:
    end_t = T_all
    note = '(no drop found - using full window)'
else:
    note = f'(power fell below {thr}W at t0+{end_t:.0f}s and stayed)'

seg = [(t, p) for t, p in P if t <= end_t]
E = sum((seg[k][1] + seg[k-1][1]) / 2 * (seg[k][0] - seg[k-1][0]) for k in range(1, len(seg)))
T = seg[-1][0]
frames = fps * T

print(f'== {path} ==')
print(f'window: {len(rows)} samples, {T_all:.1f}s, {E_all:.1f} J, avg {E_all/T_all:.2f} W')
print(f'BENCH segment: 0..{end_t:.1f}s {note}')
print(f'  {E:.1f} J  |  avg {E/T:.2f} W  |  at {fps:.0f}fps = {frames:.0f} frames')
print(f'  >> {E/frames*1000:.1f} mJ/frame  ({E/frames:.4f} J/frame)')
print()
row = ''
for k in range(int(T) // 5 + 1):
    lo, hi2 = k * 5, k * 5 + 5
    b = [p for t, p in seg if lo <= t < hi2]
    if b:
        row += f'{lo:3d}s:{sum(b)/len(b):5.1f} '
        if (k + 1) % 8 == 0:
            print(row); row = ''
if row: print(row)
