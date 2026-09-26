"""
Snapshots of a lattice descent (lattice_path.py release): where the vortex cores are.

For each snapshot the sites with 1 − |U_cc|² > thr are drawn in 3D: line 0 (the 01 vortex)
red, line 2 (the 12 vortex) blue, line 1 (shared) small green points.  Titles give the
chunk (100 L-BFGS iterations each), the interaction energy, the relative tilt of the two
rings and the overlap O = ∫ q₀ q₂.

Usage: python plot_release.py out.png N h E_ref file1.npz[:label] file2.npz[:label] ...
  E_ref: energy of the two isolated vortices on the same lattice (interaction energies are
  quoted relative to it).
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
    out, N, h, E_ref = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    items = []
    for arg in sys.argv[5:]:
        fn, _, label = arg.partition(":")
        items.append((fn, label))
    lat = Lattice(N, h)
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
    ncol = 3 if len(items) > 3 else len(items)
    nrow = (len(items) + ncol - 1) // ncol
    fig = plt.figure(figsize=(4.6 * ncol, 4.4 * nrow + 0.6))
    for i, (fn, label) in enumerate(items):
        U = np.load(fn)["U"]
        g = lat.geometry(U)
        q = [1.0 - np.abs(U[..., c, c]) ** 2 for c in range(3)]
        O = float(lat.h ** 3 * np.sum(q[0] * q[2]))
        E_int = lat.energy(U) - E_ref
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        for c, col, thr, sz, al in ((1, "limegreen", 0.9, 2, 0.15), (0, "red", 0.7, 12, 0.7), (2, "blue", 0.7, 12, 0.7)):
            m = q[c] > thr
            ax.scatter(lat.X[m], lat.Y[m], lat.Z[m], s=sz, c=col, alpha=al, linewidths=0)
        tilt = g.get("pair", {}).get("relative_tilt_deg")
        ttl = (label + "\n" if label else "") + f"E_int {E_int:+.1f}，重疊 {O:.2f}"
        if tilt is not None and g[0]["weight"] > 1 and g[2]["weight"] > 1:
            ttl += f"，相對傾角 {tilt:.0f}°"
        ax.set_title(ttl, fontsize=9)
        lim = 3.0
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=14, azim=-50)
    fig.suptitle("放開約束後的下坡（h = 0.3 格點，每段 100 步 L-BFGS）：紅＝01 漩渦的核心（線 0），"
                 "藍＝12 漩渦的核心（線 2），淡綠＝共享的線 1\nE_int 相對於兩個單獨、與格線對齊的漩渦；格點釘扎約 0.6，小於 1 的差別不可解讀",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
