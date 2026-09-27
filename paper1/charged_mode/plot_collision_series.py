"""
Time series of the ring collisions with the full kinetic term (lattice_dyn2.py, §5.9):
energy left in the box (the sponge removes what is radiated), the lattice charge, and the
content of each link (01, 12, 02) at the saved snapshots.

Usage: python plot_collision_series.py out.png
Needs data/flagpair_dyn2_collide_touch_V{0.5,0.8}_N41_sponge.json and the snapshots
$FLAGPAIR_SCRATCH/dyn2_collide_touch_V*_N41_sponge_t*.npz.
"""

import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lattice3d import SCRATCH, DATA

# categorical slots 1-3 of the reference palette (validated all-pairs, light surface)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
E_Q2 = 509.35          # static energy of the Q = 2 ring on this lattice (h = 0.3)


def link_content(U, h=0.3):
    return {f"{a}{b}": float(h ** 3 * np.sum(0.5 * (np.abs(U[..., a, b]) ** 2 + np.abs(U[..., b, a]) ** 2)))
            for a, b in ((0, 1), (1, 2), (0, 2))}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def main():
    out = sys.argv[1]
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
    runs = [("0.5", C1), ("0.8", C2)]
    data = {V: json.load(open(os.path.join(DATA, f"flagpair_dyn2_collide_touch_V{V}_N41_sponge.json"))) for V, _ in runs}
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), facecolor=SURF)
    # (a) energy left in the box
    ax = axs[0, 0]
    style(ax)
    for V, c in runs:
        r = data[V]["rows"]
        t = [x["t"] for x in r]
        E = [x["E"] for x in r]
        ax.plot(t, E, color=c, lw=2, solid_capstyle="round", label=f"V = {V}（γ = {data[V]['gamma']:.2f}）")
    ax.axhline(E_Q2, color=INK2, lw=1)
    ax.annotate("Q = 2 環的靜態能量 509.35", (0.5, E_Q2), xytext=(0, 4), textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_title("(a) 盒內剩下的能量：被吸收層帶走的就是輻射", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("t", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlim(0, 33)
    # (b) lattice charge
    ax = axs[0, 1]
    style(ax)
    for V, c in runs:
        r = data[V]["rows"]
        ax.plot([x["t"] for x in r], [-x["Q"] for x in r], color=c, lw=2, label=f"V = {V}")
    for q, txt in ((1, "互扣（荷 1）"), (2, "分開或合成（荷 2）"), (3, "互扣（荷 3）")):
        ax.axhline(q, color=INK2 if q == 2 else GRID, lw=1)
        ax.annotate(txt, (33, q), xytext=(-4, 3), textcoords="offset points", ha="right", color=INK2, fontsize=8.5)
    ax.set_ylim(0.6, 3.4)
    ax.set_xlim(0, 33)
    ax.set_title("(b) 格點荷 −Q：接觸時暫時偏離（格點上核心重疊），之後回到 2", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("t", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    # (c, d) link content at the snapshots
    for k, (V, _) in enumerate(runs):
        ax = axs[1, k]
        style(ax)
        fs = sorted(glob.glob(os.path.join(SCRATCH, f"dyn2_collide_touch_V{V}_N41_sponge_t*.npz")),
                    key=lambda f: float(re.search(r"_t([0-9.]+)\.npz", f).group(1)))
        ts = [float(re.search(r"_t([0-9.]+)\.npz", f).group(1)) for f in fs]
        L = [link_content(np.load(f)["U"]) for f in fs]
        series = (("01", C1, "連結 01"), ("12", C2, "連結 12"), ("02", C3, "連結 02（新產生）"))
        ends = []
        for key, c, name in series:
            y = [x[key] for x in L]
            ax.plot(ts, y, color=c, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=name)
            ends.append([y[-1], name.split("（")[0]])
        # end labels, nudged apart so that they never overlap (at least 2.2 units in y)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 2.2)
        for yl, txt in ends:
            ax.annotate(txt, (ts[-1], yl), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.axhline(21.02, color=GRID, lw=1)
        ax.annotate("Q = 2 環的單一連結含量 21.0", (0.3, 21.02), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=8.5)
        ax.set_xlim(0, 34)
        ax.set_ylim(0, 34)
        ax.set_title(f"({'cd'[k]}) V = {V}：各連結的含量（每 5 個時間單位的快照）", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("t", color=INK2)
        ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.suptitle("完整動能的兩環對撞（h = 0.3 格點，吸收邊界）：01 環 + 12 環 → 接觸時產生 02 → 合成連結 12 的 Q = 2 環 + 輻射",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
