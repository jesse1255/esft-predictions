"""
Free ringing of the relaxed Q = 2 ring (lattice_dyn2.py ring): rms radius against time with a
damped-cosine fit, compared with the in-block bound mode of §4.4 (ω² = 0.465, computed with the
axisymmetric code and the full kinetic metric).

Usage: python plot_ringing.py out.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
C1, C2 = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def model(t, A, om, ph, c, g):
    return c + A * np.exp(-g * t) * np.cos(om * t + ph)


def main():
    out = sys.argv[1]
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    d = json.load(open(os.path.join(DATA, "flagpair_dyn2_ring_fused02_eps0.05_N41.json")))
    t = np.array([x["t"] for x in d["trace"]])
    R = np.array([x["R"][0] for x in d["trace"]])
    p, cov = curve_fit(model, t, R, p0=[np.ptp(R) / 2, 0.68, 0.0, R.mean(), 0.0], maxfev=20000)
    om, dom = abs(p[1]), np.sqrt(cov[1, 1])
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    tt = np.linspace(t[0], t[-1], 800)
    ax.plot(tt, model(tt, *p), color=C2, lw=2, label=f"擬合：ω = {om:.4f} ± {dom:.4f}")
    ax.plot(t[::3], R[::3], "o", color=C1, ms=5, mec=SURF, mew=1.5, label="時間演化：環的均方根半徑")
    ax.set_xlabel("t", color=INK2)
    ax.set_ylabel("均方根半徑", color=INK2)
    ax.set_title(f"Q = 2 環的自由振動（完整動能，h = 0.3）：ω = {om:.3f}；§4.4 另一套程式算的束縛模 ω = √0.465 = 0.682",
                 fontsize=10, color=INK, loc="left")
    lo, hi = R.min(), R.max()
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.45 * (hi - lo))
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURF)
    print("wrote", out, "omega", om, "+-", dom)


if __name__ == "__main__":
    main()
