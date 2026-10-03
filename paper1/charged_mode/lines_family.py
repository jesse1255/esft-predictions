"""
All flag models come from one base (the user's remark, 2026-10-03: three is three modes of the same
segment; turning it by 180 degrees; 0 … 6 all have their modes; five should also have SU(6)).

1. One base, n modes.  The n lines of the SU(n) flag model are the n states m = −j … j of one
   rotating segment of spin j = (n − 1)/2 (the principal SU(2) inside SU(n), Kostant): spin ½ → 2
   modes, spin 1 → 3, spin 2 → 5, spin 5/2 → 6.  Turning by 180° about a perpendicular axis sends
   m → −m: the modes pair up (m, −m), with one unpaired m = 0 mode exactly when n is odd.
2. Links by class.  In the Coxeter plane line m sits at the n-th root of unity e^{2πi m/n}; the link
   (m, m') is the chord of length 2 sin(πk/n), k = |m − m'| folded to ≤ n/2.  Counted per class;
   ratios between classes; the golden ratio appears when 5 divides n.
3. Who talks to whom.  Positive roots e_a − e_b; the Killing products α·β ∈ {−1, 0, +1} are the
   linking coefficients of §5.5 (0 for links that share no line: new for n ≥ 4).
4. Pascal's triangle: n lines have C(n, k) groups of k lines — 1-line terms (Faddeev, mass),
   2-line terms (σ, links), 3-line terms (three-cycle) …

Usage: python lines_family.py [table|figure|all]
"""

import itertools
import json
import os
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHI = (1 + 5 ** 0.5) / 2


def family(n):
    j = (n - 1) / 2
    ms = [j - i for i in range(n)]
    classes = {}
    for a, b in itertools.combinations(range(n), 2):
        k = min(b - a, n - (b - a))
        classes.setdefault(k, []).append((a, b))
    lengths = {k: 2 * np.sin(np.pi * k / n) for k in classes}
    ks = sorted(classes)
    ratios = [lengths[ks[i + 1]] / lengths[ks[i]] for i in range(len(ks) - 1)]
    golden = any(abs(r - PHI) < 1e-9 for r in ratios) or any(abs(lengths[a] / lengths[b] - PHI) < 1e-9
                                                               for a in ks for b in ks if a != b)
    roots = [(a, b) for a, b in itertools.combinations(range(n), 2)]

    def dot(r, s):
        va, vb = np.zeros(n), np.zeros(n)
        va[r[0]], va[r[1]] = 1, -1
        vb[s[0]], vb[s[1]] = 1, -1
        return int(round(va @ vb))
    prods = [dot(r, s) for r, s in itertools.combinations(roots, 2)]
    return dict(n=n, spin=j, modes_m=ms, zero_mode=(n % 2 == 1), links=comb(n, 2), triangles=comb(n, 3),
                cartan=n - 1, dim=n * n - 1,
                classes={int(k): dict(count=len(v), length=float(lengths[k])) for k, v in classes.items()},
                ratios=[float(r) for r in ratios], golden=bool(golden),
                killing=dict(minus1=prods.count(-1), zero=prods.count(0), plus1=prods.count(1)),
                pascal=[comb(n, k) for k in range(n + 1)])


def cmd_table():
    out = []
    print(f"{'n':>2} {'spin':>5} {'0-mode':>6} {'links':>5} {'tri':>4} {'Cartan':>6} {'dim':>4}   classes (count x length)        ratios        golden  pairs(-1/0/+1)")
    for n in (2, 3, 4, 5, 6, 7, 10, 15):
        f = family(n)
        cl = ", ".join(f"{c['count']}x{c['length']:.3f}" for k, c in sorted(f["classes"].items()))
        print(f"{n:>2} {f['spin']:>5} {str(f['zero_mode']):>6} {f['links']:>5} {f['triangles']:>4} {f['cartan']:>6} {f['dim']:>4}   "
              f"{cl:32s} {str([round(r, 3) for r in f['ratios']]):14s} {str(f['golden']):6s} "
              f"{f['killing']['minus1']}/{f['killing']['zero']}/{f['killing']['plus1']}")
        out.append(f)
    json.dump(out, open(os.path.join(DATA, "lines_family.json"), "w"), indent=1)
    return out


def cmd_figure(out="figures/lines_family_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    COL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
    NAME = {1: "相鄰", 2: "隔一個", 3: "正對面（180°）"}
    ns = (2, 3, 4, 5, 6)
    fig, axs = plt.subplots(1, len(ns), figsize=(15, 3.9), facecolor=SURF)
    for ax, n in zip(axs, ns):
        f = family(n)
        ang = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
        P = np.array([[np.cos(a), np.sin(a)] for a in ang])
        drawn = set()
        for a, b in itertools.combinations(range(n), 2):
            k = min(b - a, n - (b - a))
            ax.plot(P[[a, b], 0], P[[a, b], 1], color=COL.get(k, INK2), lw=2.2 if k == 1 else 1.6)
            drawn.add(k)
        ax.scatter(P[:, 0], P[:, 1], s=60, color=INK, zorder=3)
        for i in range(n):
            m = f["modes_m"][i]
            lab = f"{m:+g}" if m != 0 else "0"
            ax.annotate(lab, P[i] * 1.22, ha="center", va="center", fontsize=9, color=INK)
        ax.set_aspect("equal")
        ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.6)
        ax.axis("off")
        lens = "、".join(f"{f['classes'][k]['length']:.3f}" for k in sorted(f["classes"]))
        extra = "（比值 φ）" if f["golden"] else ""
        ax.set_title(f"{n} 條線＝自旋 {f['spin']:g} 的 {n} 個狀態\n{f['links']} 個連結，長度 {lens}{extra}", fontsize=9.5, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COL[k], lw=2.2) for k in (1, 2, 3)]
    fig.legend(handles, [NAME[k] for k in (1, 2, 3)], loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("同一個基底的 n 種模式：線 m 放在 n 次單位根上（Coxeter 平面），連結是弦，長度 2 sin(πk/n)；數字是 m（轉 180° 讓 m → −m）",
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(os.path.join(HERE, out), dpi=115, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("table", "all"):
        cmd_table()
    if what in ("figure", "all"):
        cmd_figure()
