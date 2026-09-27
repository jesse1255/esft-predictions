"""
Figure for the resonance windows of kink–antikink collisions (phi4_windows.py): outcome against the
incoming speed (wide scan and a zoom at the edge of the widest two-collision window), the
uncertainty fraction f(ε) that measures the dimension of the set of window edges, and the
resonance law of the two-collision windows.

Usage: python plot_phi4_windows.py out.png
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
C1, C2 = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROWS = ["捕獲", "碰 1 次彈開", "碰 2 次", "碰 3 次", "碰 4 次以上"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_axisbelow(True)


def strip(ax, tag, title):
    d = json.load(open(os.path.join(DATA, f"phi4_windows_{tag}.json")))
    v, lab = np.array(d["v"]), np.minimum(np.array(d["label"]), 4)
    for r in range(5):
        sel = lab == r
        ax.vlines(v[sel], r - 0.36, r + 0.36, color=INK if r else INK2, lw=0.6 if r else 0.4)
    ax.set_yticks(range(5))
    ax.set_yticklabels(ROWS, fontsize=9, color=INK)
    ax.set_ylim(-0.6, 4.6)
    ax.set_xlim(v[0], v[-1])
    ax.grid(True, axis="x", color=GRID, lw=1)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("入射速度 v（光速 = 1）", color=INK2)
    return v, lab


def main():
    out = sys.argv[1]
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    an = json.load(open(os.path.join(DATA, "phi4_windows_analysis.json")))
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor=SURF)
    style(axs[0, 0])
    strip(axs[0, 0], "wide", "(a) 碰撞結果 vs 速度：v_c ≈ 0.259 以上碰一次就彈開，以下多半被捕獲，\n中間有一串「窗」（灰底是 (b) 放大的範圍）")
    ax = axs[0, 0]
    ax.axvspan(0.2035, 0.2068, color=GRID, alpha=0.6, lw=0)
    style(axs[0, 1])
    strip(axs[0, 1], "zoom1", "(b) 放大最寬的「碰 2 次」窗的右邊緣：\n同樣的結構又出現一次（窗裡有窗）")
    axs[0, 1].ticklabel_format(axis="x", style="plain", useOffset=False)
    # (c) uncertainty fraction
    ax = axs[1, 0]
    style(ax)
    ax.grid(True, color=GRID, lw=1)
    for tag, c, name in (("wide", C1, "大範圍掃描"), ("zoom1", C2, "放大掃描")):
        e, f = np.array(an[tag]["eps"]), np.array(an[tag]["f"])
        p = np.polyfit(np.log(e), np.log(f), 1)
        ax.loglog(e, f, "o", color=c, ms=6, mec=SURF, mew=2, label=f"{name}：斜率 {p[0]:.2f} → D = {1 - p[0]:.2f}")
        ax.loglog(e, np.exp(np.polyval(p, np.log(e))), color=c, lw=2)
    e0 = np.array([1e-6, 1e-2])
    ax.loglog(e0, 0.05 * (e0 / 1e-6), color=INK2, lw=1)
    ax.annotate("斜率 1：邊界只是孤立的點（D = 0）", (1.3e-5, 0.8), color=INK2, fontsize=8.5)
    ax.set_ylim(0.03, 1.0)
    # tick labels in math mode (DejaVu glyphs): the CJK font has no minus sign
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt = FuncFormatter(lambda val, _: f"$10^{{{int(round(np.log10(val)))}}}$")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("速度差 ε", color=INK2)
    ax.set_ylabel("f(ε)：v 與 v + ε 結果不同的比例", color=INK2)
    ax.set_title("(c) 分數維度：f ∝ ε^(1 − D)，橫跨 4 個數量級，D ≈ 0.6", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    # (d) resonance law of the two-collision windows
    ax = axs[1, 1]
    style(ax)
    ax.grid(True, color=GRID, lw=1)
    w = [x for x in an["wide"]["windows"] if x[0] == 2]
    cent = np.array([(x[1] + x[2]) / 2 for x in w])[:12]

    def res(vc):
        y = 1 / np.sqrt(vc ** 2 - cent ** 2)
        p = np.polyfit(np.arange(len(cent)), y, 1)
        return np.sqrt(np.mean((y - np.polyval(p, np.arange(len(cent)))) ** 2)) / p[0]

    vc = minimize_scalar(res, bounds=(cent[-1] + 1e-7, 0.262), method="bounded").x
    y = 1 / np.sqrt(vc ** 2 - cent ** 2)
    n = np.arange(len(cent))
    p = np.polyfit(n, y, 1)
    ax.plot(n, np.polyval(p, n), color=C1, lw=2)
    ax.plot(n, y, "o", color=C1, ms=7, mec=SURF, mew=2)
    ax.set_xlabel("「碰 2 次」窗的序號 n", color=INK2)
    ax.set_ylabel("1 / √(v_c² − v_n²)", color=INK2)
    ax.set_title(f"(d) 共振律：兩次碰撞之間內部振動多轉一圈，就多一個窗（v_c = {vc:.4f}）",
                 fontsize=10, color=INK, loc="left")
    fig.suptitle("φ⁴ 扭結–反扭結碰撞：共振產生分形（1 + 1 維，標準例子，不是我們的模型）", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    main()
