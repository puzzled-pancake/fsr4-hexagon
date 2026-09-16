#!/usr/bin/env python3
# Whitepaper figures from the measured data. Casual captions live
# in the paper; these just render the numbers.
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# QP points at the author's private capture dir (not included);
# point it at your own samples_*.csv / metrics_*.jsonl captures
# (carve_energy.py formats them).
QP = os.path.join(os.path.dirname(HERE), "phase1", "qnn")

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 11.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "xtick.color": "#444444", "ytick.color": "#444444",
})

NAVY, AMBER = "#2b5f8a", "#e8873a"
GREEN, RED, GREY = "#3f9d63", "#c0504d", "#8a8a8a"

# ---- measured rows (bench-only cuts, fps-integral frames) ----------------
NATIVE = {  # preset: (W, fps)
    "Lowest": (9.39, 29.47), "Medium": (8.56, 18.29), "High": (8.61, 15.47),
}
DLSS = {    # preset: (W, fps)  (0x50, A710 pins)
    "Lowest": (11.48, 36.09), "Medium": (10.68, 20.96), "High": (10.25, 17.66),
}
FPS_DELTA = {"Lowest": "+22% fps", "Medium": "+15% fps", "High": "+14% fps"}
LADDER = [  # label, W, fps, fr/J, sub
    ("native Lowest", 9.39, 29.47, 3.14, "9.4 W, 29.5 fps"),
    ("0x50 + A710 pins", 11.48, 36.09, 3.14, "11.5 W, 36.1 fps"),
    ("0x30 + A710 pins", 12.22, 28.42, 2.33, "12.2 W, 28.4 fps"),
    ("0x30 + A510 pins", 11.27, 26.22, 2.33, "11.3 W, 26.2 fps"),
]
PRESETS = ["Lowest", "Medium", "High"]
ERR = 0.06  # measured run-to-run band on mJ/frame
BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82)

# ---- fig 1: fps vs power with iso-fr/J lines (the money chart) -----------
fig, ax = plt.subplots(figsize=(9.2, 6.0))
for frj, lx in ((1.0, 12.8), (2.0, 12.8), (3.0, 12.8), (4.0, 12.2)):
    w = [6.5, lx]; f = [x * frj for x in w]
    ax.plot(w, f, ":", color="#aaaaaa", lw=1.1, zorder=1)
    ax.annotate(f"{frj:.0f} fr/J", xy=(lx, lx * frj), fontsize=9.5,
                color="#888888", ha="right", va="bottom", rotation=17)
# parity highlight: both Lowest points sit on the 3 fr/J line
nw, nf = NATIVE["Lowest"]; dw, df = DLSS["Lowest"]
ax.plot([nw, dw], [nf, df], ls="--", lw=1.4, color=GREEN, alpha=0.85, zorder=2)
ax.annotate("parity within noise, 22% more frames", xy=((nw + dw) / 2, (nf + df) / 2 + 0.5),
            xytext=(7.75, 37.6), fontsize=10, color="#2a7a4a", ha="left",
            bbox=BOX, zorder=5,
            arrowprops=dict(arrowstyle="->", color="#2a7a4a", lw=1.0))
for p in PRESETS:
    nw, nf = NATIVE[p]; dw, df = DLSS[p]
    ax.annotate("", xy=(dw, df), xytext=(nw, nf),
                arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.3,
                                shrinkA=11, shrinkB=11, alpha=0.8))
    ax.scatter([nw], [nf], s=150, color=NAVY, edgecolor="white", linewidth=1.4, zorder=4)
    ax.scatter([dw], [df], s=150, color=AMBER, edgecolor="white", linewidth=1.4, zorder=4)
    ax.annotate(p, xy=((nw + dw) / 2, (nf + df) / 2), textcoords="offset points",
                xytext=(-6, -16), ha="center", fontsize=11, color="#333333",
                fontweight="bold")
ax.annotate("native Lowest", (9.39, 29.47), textcoords="offset points",
            xytext=(-12, 10), ha="right", fontsize=10, color=NAVY, bbox=BOX, zorder=5)
ax.annotate("FSR4+NPU Lowest", (11.48, 36.09), textcoords="offset points",
            xytext=(-4, -22), ha="center", fontsize=10, color="#b56a20", bbox=BOX, zorder=5)
ax.scatter([], [], s=140, color=NAVY, edgecolor="white", label="native 1080p")
ax.scatter([], [], s=140, color=AMBER, edgecolor="white", label="FSR4 on NPU (540p internal)")
ax.legend(loc="upper left", frameon=False, fontsize=10.5)
ax.set_xlabel("system power (W, battery fuel gauge, unplugged)")
ax.set_ylabel("delivered fps (RotTR benchmark, uncapped)")
ax.set_xlim(7.5, 13); ax.set_ylim(10, 42)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_fps_vs_power.png"))
plt.close(fig)

# ---- fig 2: energy per frame by preset ------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 5.6))
x = range(len(PRESETS)); w = 0.36
nat_mj = [NATIVE[p][0] / NATIVE[p][1] * 1000 for p in PRESETS]
dls_mj = [DLSS[p][0] / DLSS[p][1] * 1000 for p in PRESETS]
b1 = ax.bar([i - w / 2 for i in x], nat_mj, w, color=NAVY, label="native",
            yerr=[v * ERR for v in nat_mj], ecolor="#333333", capsize=4, error_kw={"lw": 1.1})
b2 = ax.bar([i + w / 2 for i in x], dls_mj, w, color=AMBER, label="FSR4 on NPU",
            yerr=[v * ERR for v in dls_mj], ecolor="#333333", capsize=4, error_kw={"lw": 1.1})
for bars, vals in ((b1, nat_mj), (b2, dls_mj)):
    for r, v in zip(bars, vals):
        ax.annotate(f"{v:.0f}", (r.get_x() + r.get_width() / 2, v + v * ERR),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=11, fontweight="bold")
ax.annotate("the parity row:\nnative 318.5, FSR4 318.2",
            xy=(0 + w / 2, dls_mj[0] + dls_mj[0] * ERR + 10),
            xytext=(0.02, 455), fontsize=10.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1,
                            connectionstyle="arc3,rad=-0.25"))
ax.set_xticks(list(x))
ax.set_xticklabels([f"{p}\n{FPS_DELTA[p]}" for p in PRESETS], fontsize=11.5)
ax.set_ylabel("energy per frame (mJ)")
ax.set_ylim(0, 640); ax.legend(frameon=False, fontsize=10.5, loc="upper left")
ax.margins(y=0)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig2_mj_per_frame.png"))
plt.close(fig)

# ---- fig 3: the corner/pin ladder ------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 5.4))
labels = [l for l, _, _, _, _ in LADDER]
vals = [v for _, _, _, v, _ in LADDER]
subs = [s for *_, s, in LADDER]
cols = [NAVY, GREEN, RED, RED]
bars = ax.bar(labels, vals, color=cols, width=0.6, edgecolor="white", linewidth=1)
for r, v in zip(bars, vals):
    ax.annotate(f"{v:.2f} fr/J", (r.get_x() + r.get_width() / 2, v),
                textcoords="offset points", xytext=(0, 6), ha="center",
                fontsize=12, fontweight="bold")
ax.axhline(3.14, color=NAVY, ls="--", lw=1.1, alpha=0.55)
ax.annotate("native Lowest, the line to beat", xy=(3.42, 3.20),
            fontsize=10, color=NAVY, ha="right", bbox=BOX, zorder=5)
ax.set_ylabel("frames per joule (Lowest preset, uncapped)")
ax.set_ylim(0, 3.55)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f"{l}\n{s}" for l, s in zip(labels, subs)], fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_corner_ladder.png"))
plt.close(fig)

# ---- fig 4: a raw power trace with the bench window carved out ------------
rows = []
with open(os.path.join(QP, "samples_unc_dlss_low_r59.csv")) as f:
    r = csv.reader(f); next(r)
    for t, v, i in r:
        rows.append((float(t), float(v), float(i)))
t0 = 1789600884.5151
T = [t - t0 for t, v, i in rows]
P = [-(v * i) * 1e-12 for t, v, i in rows]
# 1-second rolling mean to tame the 8.5Hz spikiness
K = 9
roll = [sum(P[max(0, k - K):k + 1]) / len(P[max(0, k - K):k + 1]) for k in range(len(P))]
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.plot(T, P, lw=0.8, color=NAVY, alpha=0.30)
ax.plot(T, roll, lw=1.8, color=NAVY)
ax.axvspan(14, 133, color=GREEN, alpha=0.07)
ax.axvline(14, color=GREEN, ls="--", lw=1.3)
ax.axvline(133, color=RED, ls="--", lw=1.3)
ax.annotate("bench starts (power ramp)", xy=(15, 12.3), xytext=(34, 14.6),
            fontsize=10.5, color="#2a7a4a",
            arrowprops=dict(arrowstyle="->", color="#2a7a4a", lw=1.1))
ax.annotate("bench ends (collapse to 3 W)", xy=(132.2, 4.6), xytext=(84, 2.9),
            fontsize=10.5, color=RED, bbox=BOX, zorder=5,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
ax.annotate("menu and loading either side:\nreal watts, but they do not count",
            xy=(146, 8.6), xytext=(138, 5.1), fontsize=10, color="#666666")
ax.set_xlabel("seconds since pad-A trigger")
ax.set_ylabel("system power (W)")
ax.set_xlim(0, 180); ax.set_ylim(2, 16)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig4_bench_window.png"))
plt.close(fig)

print("figures written:", [f for f in sorted(os.listdir(HERE)) if f.endswith(".png")])
