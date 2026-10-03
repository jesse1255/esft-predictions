"""
Figure for nlines.py offblock: can two four-line rings (links 01 and 23) pass through each other?
Lowest eigenvalues of the off-block Hessian per configuration, and where the softest mode lives.

Usage: python plot_offblock.py [figures/four_lines_passage_R6.png]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def main(fn="figures/four_lines_passage_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    res = json.load(open(os.path.join(HERE, "data", "nlines_offblock.json")))
    vecs = np.load(os.path.join(SCRATCH, "nlines_offblock_vecs.npy"), allow_pickle=True).item()
    UA = np.load(os.path.join(SCRATCH, "lattice_ref_A01_N41_h0.3.npz"))["U"]
    N, h = UA.shape[0], 0.3
    core = 1 - np.abs(UA[..., 0, 0]) ** 2
    rot = lambda F: np.rot90(F, 1, axes=(1, 2))
    cores = {"single": [core], "coaxial_on_top": [core, core], "crossed": [core, rot(core)],
             "linked": [np.roll(core, -2, 0), np.roll(rot(core), 2, 0)]}
    NAME = {"single": "單一個 01 環", "coaxial_on_top": "01 + 23 同軸疊在一起", "crossed": "01 + 23 垂直交叉",
            "linked": "01 + 23 互扣"}
    names = [k for k in NAME if k in res]
    beta = res[names[0]]["continuum_bottom"]
    fig = plt.figure(figsize=(14, 4.6), facecolor=SURF)
    gs = fig.add_gridspec(1, 1 + len(names), width_ratios=[1.6] + [1] * len(names))
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(SURF)
    for i, k in enumerate(names):
        lam = res[k]["lam"]
        ax.scatter([i] * len(lam), lam, color=C1, s=36, zorder=3)
    ax.axhline(beta, color=C2, lw=1.2, ls="--")
    ax.annotate("真空連續譜的底 β", (len(names) - 0.6, beta), color=C2, fontsize=9, va="bottom", ha="right")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.annotate("< 0：穿過時會開始混合（不穩定）", (-0.4, 0), color=INK2, fontsize=8.5, va="top")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NAME[k] for k in names], fontsize=8.5, rotation=12)
    ax.set_ylabel("離塊 Hessian 的最低本徵值", color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    lo = min(min(res[k]["lam"]) for k in names)
    ax.set_ylim(min(-0.01, lo - 0.01), beta * 1.25)
    ax.set_title("把 01、23 兩環混合的方向（02、03、12、13）", fontsize=10, color=INK, loc="left")
    m = N // 2
    ext = [-(N - 1) / 2 * h, (N - 1) / 2 * h] * 2
    for j, k in enumerate(names):
        ax = fig.add_subplot(gs[0, 1 + j])
        v = vecs[k][:, 0].reshape(N, N, N, 8)
        dens = (v ** 2).sum(-1)
        sl = dens[:, m, :].T                      # the xz plane (y = 0): both ring planes cut it
        ax.imshow(sl / sl.max(), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1)
        for c in cores[k]:
            ax.contour(np.linspace(ext[0], ext[1], N), np.linspace(ext[0], ext[1], N), c[:, m, :].T, levels=[0.5],
                       colors=[C2], linewidths=1.0)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{NAME[k]}\n最軟模 λ = {res[k]['lam'][0]:.4f}", fontsize=9, color=INK)
    fig.suptitle("四條線：兩個不共用線的環能不能直接穿過彼此？藍：最軟的混合模態（y = 0 截面），橙：環的核心",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(HERE, fn), dpi=115, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main(*sys.argv[1:])
