"""
Figure for prime_decay.py: how the unstable axial rings decay on the 3D lattice (energy against L-BFGS
iterations) and what their core curves look like before and after, with the writhe of each core curve.
The undeformed relaxations (eps = 0, the lattice's own relaxation of the embedded ring) are drawn as
reference marks, not as decays.

Usage: python plot_prime_decay.py [figures/prime_decay_R6.png]
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def tag_of(r):
    return f"{r['key']}_k{r['k']}" + ("" if r["eps"] == 0.2 else f"_eps{r['eps']:g}") + ("_pc" if any(r.get("shift", [0])) else "")


def main(fn="figures/prime_decay_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lattice3d import Lattice
    from prime_decay import lattice_ring
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"4x1": "#2a78d6", "5x1": "#eb6834", "6x1": "#4a3aa7", "7x1": "#e34948", "8x1": "#eda100",
           "4x2": "#9a6b1f", "2x2": "#1baf7a", "3x2": "#0e8a5f", "3x3": "#6fcf97"}
    order = {"4x1": 0, "5x1": 1, "6x1": 2, "7x1": 3, "8x1": 4, "4x2": 5, "2x2": 6, "3x2": 7, "3x3": 8}
    allruns = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "data", "prime_decay_*_k*.json")))]
    runs = sorted([r for r in allruns if r["eps"] > 0], key=lambda r: (order.get(r["key"], 9), r["k"]))
    refs = [r for r in allruns if r["eps"] == 0]
    wfn = os.path.join(HERE, "data", "prime_decay_writhe.json")
    wr = json.load(open(wfn))["states"] if os.path.exists(wfn) else {}
    lat = Lattice(41, 0.3)
    n = len(runs)
    rows = (n + 2) // 3
    fig = plt.figure(figsize=(14, 4.4 + 3.7 * rows), facecolor=SURF)
    gs = fig.add_gridspec(1 + rows, 3, height_ratios=[1.15] + [1.1] * rows)
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(SURF)
    for r in runs:
        it = [0] + [h[0] for h in r["history"]]
        E = [r["E_seed"]] + [h[1] for h in r["history"]]
        rel = [100 * (e / r["E_axial_lattice"] - 1) for e in E]
        lab = f"{r['m']}×{r['n']}（Q = {r['Q']}，推 k = {r['k']}）：{r['E_axial_lattice']:.0f} → {r['E_final']:.0f}"
        ax.plot(it, rel, color=COL.get(r["key"], INK2), lw=2, marker="o", ms=3.5, ls="--" if r["key"] in ("2x2", "3x2") else "-", label=lab)
    # undeformed relaxations, relative to the same ring embedded at a lattice site (as the decays are)
    site = {r["key"]: r["E_axial_lattice"] for r in allruns if not any(r.get("shift", [0]))}
    xmax = max(h[0] for r in runs for h in r["history"])
    for j, r in enumerate(sorted(refs, key=lambda r: r["E_final"] / site.get(r["key"], r["E_axial_lattice"]))):
        y = 100 * (r["E_final"] / site.get(r["key"], r["E_axial_lattice"]) - 1)
        ax.plot([0.62 * xmax, xmax], [y, y], color=INK2, lw=0.9, ls=":")
        ax.annotate(f"{r['m']}×{r['n']} 不推，只讓格點鬆弛" + ("，環心在格子中心" if any(r.get("shift", [0])) else "，環心在格點上")
                    + f"：{y:+.2f}%", (0.62 * xmax, y), xytext=(0, 3 if j % 2 else -11), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("L-BFGS 步數", color=INK2)
    ax.set_ylabel("相對軸對稱環的能量（%）", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title("在完整的三條線模型裡沿最不穩定的方向輕推再鬆弛：會扭的環（實線）一路往下掉，2×2、3×2（虛線）只動到格點本身的尺度",
                 fontsize=10.5, color=INK, loc="left")
    for j, r in enumerate(runs):
        ax = fig.add_subplot(gs[1 + j // 3, j % 3], projection="3d")
        _, U0, _ = lattice_ring(r["key"], r["k"], 0.0, lat)
        U = np.load(os.path.join(SCRATCH, f"prime_decay_{tag_of(r)}.npz"))["U"]
        for UU, col, s, a in ((U0, "#c8c6c0", 2, 0.25), (U, None, 4, 0.9)):
            c = np.abs(UU[..., 0, 0]) ** 2 < 0.1
            x, y, z = lat.X[c], lat.Y[c], lat.Z[c]
            if col is None:
                ax.scatter(x, y, z, c=z, cmap="coolwarm", s=s, alpha=a, depthshade=False, vmin=-2, vmax=2)
            else:
                ax.scatter(x, y, z, color=col, s=s, alpha=a, depthshade=False)
        lim = 4.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim / 2, lim / 2)
        ax.set_box_aspect((2, 2, 1))
        ax.view_init(elev=35, azim=-60)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        w = wr.get(tag_of(r), {}).get("0.1", {})
        wtxt = (f"，扭曲數 Wr = {w['Wr']:+.2f}" if abs(w["Wr"]) >= 0.005 else "，扭曲數 Wr = 0.00") if w else ""
        ax.set_title(f"{r['m']}×{r['n']}（Q = {r['Q']}），推 k = {r['k']}：放出 {r['released']:+.1f}\n"
                     f"半徑 {r['shape_axial'].get('r_mean', 0):.2f} → {r['shape_final'].get('r_mean', 0):.2f}{wtxt}，荷 {r['charge']:+.2f}",
                     fontsize=9, color=INK)
    fig.suptitle("不穩定代表什麼：軸對稱環的核心曲線（灰）與衰變後的形狀（彩色，顏色是高度）", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(os.path.join(HERE, fn), dpi=110, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main(*sys.argv[1:])
