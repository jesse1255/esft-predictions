"""
Snapshots of a lattice descent (lattice_path.py release): where the vortex cores are.

For each snapshot the sites with 1 − |U_cc|² > thr are drawn in 3D: line 0 (the 01 vortex)
red, line 2 (the 12 vortex) blue, line 1 (shared) small green points.  Titles give the
chunk (100 L-BFGS iterations each), the interaction energy, the relative tilt of the two
rings and the overlap O = ∫ q₀ q₂.

Usage: python plot_release.py out.png N h tag chunk1 chunk2 ...   (states from $FLAGPAIR_SCRATCH)
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lattice3d import Lattice, SCRATCH, DATA


def main():
    out, N, h, tag = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    chunks = [float(c) for c in sys.argv[5:]]
    lat = Lattice(N, h)
    rows = {}
    fn = os.path.join(DATA, f"flagpair_lattice_path_N{N}_h{h:g}.json")
    for r in json.load(open(fn))["rows"]:
        if r.get("kind") == f"release_{tag}":
            rows = {row["chunk"]: row for row in r["rows"]}
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
    fig = plt.figure(figsize=(4.2 * len(chunks), 4.6))
    for i, ch in enumerate(chunks):
        U = np.load(os.path.join(SCRATCH, f"lattice_release_{tag}_N{N}_h{h:g}_s{ch:.3f}.npz"))["U"]
        ax = fig.add_subplot(1, len(chunks), i + 1, projection="3d")
        for c, col, thr, sz in ((0, "crimson", 0.8, 4), (2, "navy", 0.8, 4), (1, "green", 0.9, 1)):
            q = 1.0 - np.abs(U[..., c, c]) ** 2
            m = q > thr
            ax.scatter(lat.X[m], lat.Y[m], lat.Z[m], s=sz, c=col, alpha=0.35 if c != 1 else 0.15, linewidths=0)
        r = rows.get(int(ch) + 1, rows.get(int(ch), {}))
        ttl = f"第 {int(ch) + 1} 段"
        if r:
            ttl += f"：E_int {r['E_int']:+.2f}\n相對傾角 {r['rel_tilt']:.0f}°，重疊 {r['overlap02']:.2f}"
        ax.set_title(ttl, fontsize=9)
        lim = 3.6
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("放開約束後的下坡：紅＝01 漩渦的核心（線 0），藍＝12 漩渦的核心（線 2），綠＝共享的線 1", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
