"""
Figure for prime_charges.py: energy per Q^(3/4) of every axial ring (m, n) with Q = m·n ≤ 8, the
cheapest one per charge, and what the competing factorizations look like for Q = 6 and Q = 7.

Usage: python plot_prime_charges.py [figures/prime_charges_R6.png]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def main(fn="figures/prime_charges_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_hopf_pair import freeze_A
    from hopfion_axisym import Grid, U3
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2, GRID, C1, C2, C3, C4 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
    d = json.load(open(os.path.join(HERE, "data", "prime_charges_ne32x48_a5.json")))
    per = d["per_factorization"]
    chk_fn = os.path.join(HERE, "data", "prime_charges_check_ne40x64_a7.json")
    chk = {f"{r['m']}x{r['n']}": r for r in json.load(open(chk_fn))["results"]} if os.path.exists(chk_fn) else {}
    fam = {"azimuthal": ([(q, 1) for q in range(1, 9)], C1, "o", "Q×1：全部繞環方向"),
           "two": ([(2, 2), (3, 2), (4, 2)], C2, "s", "m×2：繞管 2 圈"),
           "m2": ([(2, 3), (2, 4)], C3, "^", "2×n：繞環 2 圈、繞管 n 圈"),
           "merid": ([(1, q) for q in range(2, 9)], C4, "x", "1×n：全部繞管方向（分裂成幾個環）")}
    fig = plt.figure(figsize=(14, 8.4), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    ax = fig.add_subplot(gs[0, :3])
    style(ax)
    best = {}
    for key, (pairs, col, mk, lab) in fam.items():
        xs, ys = [], []
        for (m, n) in pairs:
            r = per.get(f"{m}x{n}")
            if r is None:
                continue
            Q = m * n
            y = r["E"] / Q ** 0.75
            xs.append(Q)
            ys.append(y)
            if Q not in best or r["E"] < best[Q][1]:
                best[Q] = ((m, n), r["E"], y)
        ax.plot(xs, ys, color=col, lw=1.4 if key != "merid" else 0, marker=mk, ms=8, mew=1.6,
                mfc=col if mk not in ("x",) else None, label=lab, zorder=3)
    bx = sorted(best)
    ax.plot(bx, [best[q][2] for q in bx], color=INK, lw=0.8, ls=":", zorder=2)
    ax.scatter(bx, [best[q][2] for q in bx], s=210, facecolors="none", edgecolors=INK, lw=1.2, zorder=4,
               label="每個荷最便宜的那個")
    for q in bx:
        (m, n), E, y = best[q]
        prime = q > 1 and all(q % k for k in range(2, q))
        ax.annotate(f"{m}×{n}", (q, y - 9), ha="center", va="top", fontsize=9, color=INK)
    comp = [best[q][2] for q in (1, 4, 6, 8)]
    trend = float(np.mean(comp))
    ax.axhline(trend, color=INK2, lw=0.8, ls="--")
    ax.annotate(f"合數 1、4、6、8 的平均 {trend:.0f}", (4.55, trend - 1.5), ha="left", va="top", color=INK2, fontsize=9)
    ax.set_xticks(range(1, 9))
    ax.set_xticklabels([f"{q}\n{'質數' if q > 1 and all(q % k for k in range(2, q)) else ''}" for q in range(1, 9)])
    ax.set_xlim(0.5, 8.6)
    ax.set_ylim(280, 440)
    ax.set_xlabel("荷 Q = m × n", color=INK2)
    ax.set_ylabel("E / Q^0.75", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("軸對稱的環：合數荷可以把圈數分給兩個方向（2×2、3×2、4×2），質數荷只能 Q×1 → Q ≥ 5 的質數比較貴",
                 fontsize=10.5, color=INK, loc="left")
    ax = fig.add_subplot(gs[0, 3])
    style(ax)
    pen = [100 * (best[q][2] / trend - 1) for q in bx]
    cols = [C2 if (q > 1 and all(q % k for k in range(2, q))) else C1 for q in bx]
    ax.bar(bx, pen, color=cols, width=0.62)
    ax.axhline(0, color=INK2, lw=0.8)
    for q, p in zip(bx, pen):
        ax.annotate(f"{p:+.1f}%", (q, p + (0.6 if p >= 0 else -0.6)), ha="center", va="bottom" if p >= 0 else "top",
                    fontsize=8.5, color=INK)
    ax.set_xticks(bx)
    ax.set_xlabel("Q（橙：質數，藍：1 與合數）", color=INK2)
    ax.set_ylabel("相對合數平均的能量（%）", color=INK2)
    ax.set_ylim(-8, 16)
    ax.set_title("質數的「代價」", fontsize=10.5, color=INK, loc="left")
    # shapes: u3 in the meridional plane for the Q = 6 factorizations and Q = 7
    G = freeze_A(Grid(d["ne_r"], d["ne_z"], p=2, a=d["a"], half=True))
    rho = np.linspace(0.0, 5.0, 251)
    zz = np.linspace(0.0, 3.5, 176)
    RR, ZZ = np.meshgrid(rho, zz, indexing="ij")
    P = G.interp_points(RR.ravel(), ZZ.ravel())
    for j, key in enumerate(("6x1", "3x2", "2x3", "7x1")):
        r = per[key]
        U = np.load(os.path.join(SCRATCH, f"prime_mn_state_m{r['m']}_n{r['n']}_{r['seed']}_ne{d['ne_r']}x{d['ne_z']}_a{d['a']:g}.npz"))["U"]
        u3 = (P @ U[:, U3]).reshape(RR.shape)
        full = np.concatenate([u3[:, ::-1], u3[:, 1:]], axis=1)
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(full.T, origin="lower", extent=[0, 5, -3.5, 3.5], cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.contour(np.linspace(0, 5, 251), np.linspace(-3.5, 3.5, 351), full.T, levels=[0.0], colors=[INK], linewidths=0.8)
        ax.set_xlabel("ρ（離軸距離）", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8)
        q = r["m"] * r["n"]
        ax.set_title(f"Q = {q}：{r['m']}×{r['n']}，E = {r['E']:.0f}", fontsize=9.5, color=INK)
    fig.suptitle("質數荷會「受挫」嗎？我們模型單一連結扇區（CP¹ Faddeev–Skyrme，有質量）的軸對稱環；下排：子午面上的 u₃（紅 = 核心，藍 = 真空）",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, fn), dpi=110, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main(*sys.argv[1:])
