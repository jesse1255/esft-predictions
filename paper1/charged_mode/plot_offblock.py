"""
Figure for nlines.py offblock / refine / nlines_along_mode.py: can two four-line rings (links 01 and 23)
pass through each other?  Lowest eigenvalues of the off-block Hessian per configuration, the energy
along the negative mode of the coaxial overlap, and where the softest modes live.

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
    SURF, INK, INK2, GRID, C1, C2, C3 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
    res = json.load(open(os.path.join(HERE, "data", "nlines_offblock.json")))
    line = json.load(open(os.path.join(HERE, "data", "nlines_offblock_line.json")))
    names = ["single", "coaxial_on_top", "crossed", "linked"]
    NAME = {"single": "單一個 01 環", "coaxial_on_top": "01 + 23 同軸疊合", "crossed": "01 + 23 垂直交叉",
            "linked": "01 + 23 互扣"}

    def vec(k):
        r = os.path.join(SCRATCH, f"nlines_offblock_vec_{k}_refined.npy")
        if os.path.exists(r):
            return np.load(r)
        return np.load(os.path.join(SCRATCH, f"nlines_offblock_vec_{k}.npy"))[:, 0]
    UA = np.load(os.path.join(SCRATCH, "lattice_ref_A01_N41_h0.3.npz"))["U"]
    N, h = UA.shape[0], 0.3
    core = 1 - np.abs(UA[..., 0, 0]) ** 2
    rot = lambda F: np.rot90(F, 1, axes=(1, 2))
    cores = {"single": [core], "coaxial_on_top": [core], "crossed": [core, rot(core)],
             "linked": [np.roll(core, -2, 0), np.roll(rot(core), 2, 0)]}
    beta = res["single"]["continuum_bottom"]
    fig = plt.figure(figsize=(14, 7.8), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    ax = fig.add_subplot(gs[0, :2])
    style(ax)
    for i, k in enumerate(names):
        ax.scatter([i - 0.08] * len(res[k]["lam"]), res[k]["lam"], color=C1, s=34, zorder=3,
                   label="短預算（30 次）" if i == 0 else None)
        if "refined" in res[k]:
            ax.scatter([i + 0.08], [res[k]["refined"]["lam"]], color=C2, marker="D", s=40, zorder=4,
                       label="精煉（60 次）" if k == "coaxial_on_top" else None)
    if "single_long" in res:
        ax.scatter([0 + 0.08] * len(res["single_long"]["lam"]), res["single_long"]["lam"], color=C3, marker="s", s=30,
                   zorder=3, label="單環長時間（63 次）")
    ax.axhline(beta, color=C2, lw=1.0, ls="--")
    ax.annotate("真空連續譜的底 β = 0.108", (-0.45, beta + 0.003), color=C2, fontsize=9, ha="left", va="bottom")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.annotate("< 0：不穩定（兩族開始混合）", (3.45, -0.002), color=INK2, fontsize=9, ha="right", va="top")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([NAME[k] for k in names], fontsize=9)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.04, 0.135)
    ax.set_ylabel("離塊 Hessian 的最低本徵值", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("把兩族混合的方向（02、03、12、13）：只有完全疊合時出現負模", fontsize=10, color=INK, loc="left")
    ax = fig.add_subplot(gs[0, 2:])
    style(ax)
    rows = np.array(line["rows"])
    ax.plot([0] + list(rows[:, 0]), [0] + list(rows[:, 1] - line["E0"]), color=C1, lw=2, marker="o", ms=4, label="真正的能量")
    ss = np.linspace(0, 0.8, 100)
    k2 = (rows[0, 2] - line["E0"]) / rows[0, 0] ** 2
    ax.plot(ss, k2 * ss ** 2, color=C2, lw=1.2, ls="--", label="二次預測（負曲率）")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-0.6, 1.2)
    ax.set_xlabel("最大混合角（弧度）", color=INK2)
    ax.set_ylabel("E − E(疊合)", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    lo = rows[np.argmin(rows[:, 1])]
    ax.set_title(f"沿負模走直線：最低只降 {lo[1] - line['E0']:.2f}（在 {lo[0]:.1f} rad），之後四次項把它推回去",
                 fontsize=10, color=INK, loc="left")
    m = N // 2
    ext = [-(N - 1) / 2 * h, (N - 1) / 2 * h] * 2
    grid = np.linspace(ext[0], ext[1], N)
    for j, k in enumerate(names):
        ax = fig.add_subplot(gs[1, j])
        dens = (vec(k).reshape(N, N, N, 8) ** 2).sum(-1)
        sl = dens[:, m, :].T
        ax.imshow(sl / sl.max(), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1)
        for c in cores[k]:
            ax.contour(grid, grid, c[:, m, :].T, levels=[0.5], colors=[C2], linewidths=1.0)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-3.2, 3.2)
        ax.set_xticks([])
        ax.set_yticks([])
        lam = res[k]["refined"]["lam"] if "refined" in res[k] else res[k]["lam"][0]
        ax.set_title(f"{NAME[k]}：最軟模 λ = {lam:+.3f}", fontsize=9, color=INK)
    fig.suptitle("四條線：兩個不共用線的環能不能直接穿過彼此？下排：最軟的混合模態（y = 0 截面，藍）與環的核心（橙）",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, fn), dpi=110, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main(*sys.argv[1:])
