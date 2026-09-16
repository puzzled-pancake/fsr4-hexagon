#!/usr/bin/env python3
"""Carve a bench window out of a fuel-gauge capture + GameNative metrics log,
and compute power / frames / energy-per-frame the way the whitepaper did.

Inputs
 samples_<label>.csv t_sec,voltage_uV,current_uA (tools/arm_trigger.sh, ~8.5 Hz, UNPLUGGED)
 metrics_<label>.jsonl {"timestampMs","fps",...} (GameNative powercontrol log, 2 Hz)

Method (docs/BENCH_METHOD.md)
 - window start: first sample where power leaves the load baseline and ramps
 - window end: first sample < 7.5 W after the ramp (loading/menu tail),
 or --end override; fps trace is the tiebreak arbiter
 - energy = trapezoid integral of V*I over the window
 - frames = integral of the fps field over the SAME window (totalFrameCount
 is a sliding window, not an odometer — never integrate it)
 - sanity gates printed: mean fps + frame count, to be checked against the
 band established by sibling runs before you trust the cut
 - temps: cpuTempC/gpuTempC mean/max over the window (free, from telemetry)

Usage
 python carve_energy.py samples_A.csv metrics_A.jsonl [--start S] [--end E] [--quiet]
"""
import argparse, csv, json, sys

def load_samples(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                # this fuel gauge reports DISCHARGE as negative current; we only
                # ever want magnitude (the run must be unplugged anyway)
                rows.append((float(r["t_sec"]),
                             abs(float(r["voltage_uV"])) / 1e6,
                             abs(float(r["current_uA"])) / 1e6))
            except (ValueError, KeyError, TypeError):
                pass
    return rows

def load_metrics(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
                rows.append((d["timestampMs"] / 1000.0, float(d.get("fps", 0.0)),
                             d.get("cpuTempC"), d.get("gpuTempC")))
            except (json.JSONDecodeError, KeyError):
                pass
    return rows

def trapz(ts, vs):
    return sum((ts[i+1] - ts[i]) * (vs[i] + vs[i+1]) / 2 for i in range(len(ts) - 1))

def find_window(samples, args):
    powers = [v * i for _, v, i in samples]
    n = len(samples)
    # load baseline = quietest decile at the head of the capture
    head = sorted(powers[: max(3, n // 10)])
    base = head[len(head) // 2]
    thr = base + 1.5 # W above the loading/menu floor
    start = args.start
    if start is None:
        for k, p in enumerate(powers):
            if p > thr:
                start = samples[k][0]
                break
    end = args.end
    if end is None:
        ramp_over = False
        for k in range(n - 1):
            t = samples[k][0]
            if t <= (start or 0):
                continue
            if powers[k] > thr + 0.5:
                ramp_over = True
            if ramp_over and k > 3 and powers[k] < 7.5 and powers[k] < powers[k-1] <= powers[k-2] + 0.3:
                end = t # first sustained sub-7.5W sample after the bench tail
                break
    if end is None:
        end = samples[-1][0]
    return start or samples[0][0], end

def coarse_trace(samples, step=10.0):
    """Power averaged per `step` seconds, for eyeballing cut points."""
    t0 = samples[0][0]
    buckets = {}
    for t, v, i in samples:
        buckets.setdefault(int((t - t0) // step), []).append(v * i)
    return [(b * step, sum(p) / len(p)) for b, p in sorted(buckets.items())]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("metrics")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    s = load_samples(a.samples)
    m = load_metrics(a.metrics)
    if not s or not m:
        sys.exit("empty input")
    # --start/--end may be seconds-from-capture-start (small numbers, as the
    # fallback trace prints them) or absolute epochs (large numbers)
    if a.start is not None and a.start < 10_000_000:
        a.start += s[0][0]
    if a.end is not None and a.end < 10_000_000:
        a.end += s[0][0]

    # align epochs: metrics timestamps are wall clock, samples are host clock.
    # offset so that metric times inside the capture window line up with the
    # sample span (both cover the same armed capture).
    m_t0, m_t1 = m[0][0], m[-1][0]
    s_t0, s_t1 = s[0][0], s[-1][0]
    off = (m_t0 + (m_t1 - m_t0) / 2) - (s_t0 + (s_t1 - s_t0) / 2)
    mt = [(t - off, fps, ct, gt) for t, fps, ct, gt in m]

    w0, w1 = find_window(s, a)
    win_s = [(t, v * i) for t, v, i in s if w0 <= t <= w1]
    win_m = [(t, fps, ct, gt) for t, fps, ct, gt in mt if w0 <= t <= w1]
    if len(win_s) < 10 or len(win_m) < 5:
        print("auto cut failed (thin window). Power per 10 s bucket — pick cuts and")
        print("rerun with --start/--end (seconds from capture start):")
        for b, p in coarse_trace(s):
            bar = "#" * int(p * 4)
            print(f" {b:5.0f}s {p:5.2f} W {bar}")
        sys.exit(2)
    ts = [x[0] for x in win_s]; ps = [x[1] for x in win_s]
    energy_j = trapz(ts, ps)
    dur = w1 - w0
    frames = trapz([x[0] for x in win_m], [x[1] for x in win_m])
    mean_fps = sum(x[1] for x in win_m) / len(win_m)
    temps = [(x[2], x[3]) for x in win_m if x[2] is not None]
    w = energy_j / dur
    mj = energy_j / frames * 1000 if frames else float("nan")

    print(f"window {w0:.1f} -> {w1:.1f} s ({dur:.1f} s, {len(win_s)} power samples)")
    print(f"mean power {w:6.2f} W")
    print(f"frames {frames:8.0f} (fps-integral)")
    print(f"mean fps {mean_fps:6.2f}")
    print(f"energy/frame {mj:6.1f} mJ ({energy_j:.0f} J total, {1/w:.2f} s/J)")
    if not a.quiet:
        if temps:
            c = [t[0] for t in temps]; g = [t[1] for t in temps if t[1] is not None]
            print(f"temps (C) cpu {min(c):.0f}-{max(c):.0f} gpu {min(g):.0f}-{max(g):.0f}")
        print("sanity: check mean fps + frames against sibling runs before trusting this cut;")
        print(" when power and fps disagree about the end, the fps side wins (edit --end).")

if __name__ == "__main__":
    main()
