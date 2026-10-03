"""
Irrational numbers, continued fractions and the golden ratio (the user's suggestion, 2026-10-03,
from the video 「闰的哲学：连分数与黄金分割的真正密码」).  Small calculations only (seconds).

1. packing   Points n = 1, 2, … at radius √n and angle 2π n α (α = fraction of a turn per step).
             Rational α = p/q puts the points on q straight spokes; an irrational α never repeats.
             Packing quality d(α) = smallest distance between any two of the first N points
             ("how many balls fit"): the maximum is at the golden fraction α = 1/φ = 0.618034
             (equivalently 1/φ² = 0.381966).  Also the step α = 1/(2π) (one radian per step,
             the integer polar plot), whose spiral arms come in 6, 44, 710 (continued fraction).
2. cf        Continued fractions [a0; a1, a2, …]: a large partial quotient means a very good rational
             approximation (near-resonance); all ones (φ) is the hardest number to approximate.
             Applied also to the frequencies of our model (§4.4).
3. staircase A vibration of frequency ω below the mass gap (= 1) radiates only through the first
             harmonic n ω > 1: n = ⌈1/ω⌉, radiated power ∝ A^(2n).  The order jumps at ω = 1/n:
             a staircase whose steps pile up at ω → 0.
4. coxeter   The roots e_i − e_j of SU(n) (n lines) projected to the Coxeter plane: lengths
             2 sin(πk/n).  Their ratio is the golden ratio exactly for five lines (n = 5).

Usage: python irrational_patterns.py [packing|cf|staircase|coxeter|figure|all]
"""

import json
import os
import sys
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHI = (1 + 5 ** 0.5) / 2


def points(alpha, N):
    n = np.arange(1, N + 1)
    r = np.sqrt(n)
    th = 2 * np.pi * alpha * n
    return np.stack([r * np.cos(th), r * np.sin(th)], -1)


def dmin(alpha, N):
    from scipy.spatial import cKDTree
    p = points(alpha, N)
    d, _ = cKDTree(p).query(p, k=2)
    return float(d[:, 1].min())


def cmd_packing(N=1000, M=20001):
    al = np.linspace(0.0, 0.5, M)
    d = np.array([dmin(a, N) for a in al])
    i = int(np.argmax(d))
    out = dict(N=N, alpha=al.tolist(), dmin=d.tolist(), best_alpha=float(al[i]), best_dmin=float(d[i]),
               golden=dict(alpha=1 / PHI ** 2, dmin=dmin(1 / PHI ** 2, N)))
    print(f"best alpha in [0, 0.5]: {al[i]:.6f} (1/phi^2 = {1 / PHI ** 2:.6f}), d_min = {d[i]:.4f}; at 1/phi^2 exactly:"
          f" {out['golden']['dmin']:.4f}")
    for name, a in (("3/8", 3 / 8), ("2/5", 0.4), ("0.61853 (video)", 0.61853), ("1/(2 pi)", 1 / (2 * np.pi)),
                    ("sqrt2 - 1", 2 ** 0.5 - 1), ("1/e", 1 / np.e), ("1/pi", 1 / np.pi)):
        print(f"   alpha = {name:16s} {a:.6f}: d_min = {dmin(a, N):.4f}")
    # the five best local maxima
    loc = [j for j in range(1, M - 1) if d[j] > d[j - 1] and d[j] >= d[j + 1] and d[j] > 0.5 * d[i]]
    loc = sorted(loc, key=lambda j: -d[j])[:8]
    print("   highest local maxima:", [(round(al[j], 5), round(d[j], 3)) for j in loc])
    out["local_maxima"] = [(float(al[j]), float(d[j])) for j in loc]
    json.dump(out, open(os.path.join(DATA, "irrational_packing.json"), "w"))
    return out


def cf(x, n=10):
    a = []
    for _ in range(n):
        q = int(np.floor(x))
        a.append(q)
        f = x - q
        if f < 1e-12:
            break
        x = 1 / f
    return a


def convergents(a):
    out, h0, h1, k0, k1 = [], 1, a[0], 0, 1
    out.append(Fraction(h1, k1))
    for q in a[1:]:
        h0, h1 = h1, q * h1 + h0
        k0, k1 = k1, q * k1 + k0
        out.append(Fraction(h1, k1))
    return out


def cmd_cf():
    nums = {"golden 1/phi": 1 / PHI, "pi": np.pi, "1/(2 pi) (one radian per step)": 1 / (2 * np.pi), "sqrt 2": 2 ** 0.5,
            "e": np.e,
            "single vortex omega = sqrt(0.781)": 0.781 ** 0.5, "Q=2 ring omega1 = sqrt(0.465)": 0.465 ** 0.5,
            "Q=2 ring omega2 = sqrt(0.893)": 0.893 ** 0.5, "A12 softest omega = sqrt(0.0285)": 0.0285 ** 0.5,
            "Q=2 omega2/omega1": (0.893 / 0.465) ** 0.5, "single/Q=2 omega": (0.781 / 0.465) ** 0.5,
            "E(Q=2)/E(Q=1) (h = 0.3)": 509.35 / 320.71}
    out = {}
    for k, x in nums.items():
        a = cf(x, 8)
        c = convergents(a)
        out[k] = dict(x=x, cf=a, convergents=[f"{f.numerator}/{f.denominator}" for f in c])
        print(f"{k:38s} {x:.6f}  cf {a}  convergents {[f'{f.numerator}/{f.denominator}' for f in c[:6]]}")
    json.dump(out, open(os.path.join(DATA, "irrational_cf.json"), "w"), indent=1)
    return out


def cmd_staircase():
    modes = {"A12": 0.0285 ** 0.5, "Q=2 ring 1": 0.465 ** 0.5, "single vortex": 0.781 ** 0.5, "Q=2 ring 2": 0.893 ** 0.5}
    for k, w in modes.items():
        n = int(np.ceil(1 / w - 1e-12))
        print(f"{k:14s} omega = {w:.4f}: n = {n} (n omega = {n * w:.4f}; gap to the threshold 1/n: {w - 1 / n:+.4f}) -> P ∝ A^{2 * n}")
    return modes


def cmd_coxeter():
    out = {}
    for n in range(3, 11):
        L = sorted({round(2 * np.sin(np.pi * k / n), 10) for k in range(1, n // 2 + 1)})
        ratios = [L[i + 1] / L[i] for i in range(len(L) - 1)]
        out[n] = dict(lengths=L, ratios=ratios)
        print(f"{n} lines (SU({n})): projected root lengths {np.round(L, 4).tolist()}  ratios {np.round(ratios, 4).tolist()}"
              + ("   <- golden ratio" if any(abs(r - PHI) < 1e-9 for r in ratios) else ""))
    json.dump(out, open(os.path.join(DATA, "irrational_coxeter.json"), "w"), indent=1)
    return out


def cmd_figure(out="figures/irrational_patterns_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["path.simplify"] = False          # keep the narrow spikes of the packing curve
    SURF, INK, INK2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    pk = json.load(open(os.path.join(DATA, "irrational_packing.json")))
    fig = plt.figure(figsize=(13, 7.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.8, 1])
    for j, (title, a, N) in enumerate((("黃金比例 α = 0.618034（每步轉 0.618 圈）", 1 / PHI, 600),
                                        ("影片的 α = 0.61853（只差 0.0005）", 0.61853, 600),
                                        ("有理數 α = 5/8 = 0.625：8 條直線", 0.625, 600),
                                        ("每步 1 弧度（α = 1/2π）：6、44、710 條旋臂", 1 / (2 * np.pi), 1200))):
        ax = fig.add_subplot(gs[0, j])
        p = points(a, N)
        ax.scatter(p[:, 0], p[:, 1], s=4 if N > 700 else 7, color=INK, lw=0)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9, color=INK)
    ax = fig.add_subplot(gs[1, :3])
    ax.set_facecolor(SURF)
    al, d = np.array(pk["alpha"]), np.array(pk["dmin"])
    ax.plot(al, d, color=C1, lw=0.6)
    i = int(np.argmax(d))
    ax.plot([al[i]], [d[i]], "o", color=C2, ms=7, mec=SURF, mew=1.5)
    ax.annotate("最大：α = 0.38197 ≈ 1/φ²（≡ 0.618）", (al[i], d[i]), xytext=(8, 2), textcoords="offset points",
                color=INK, fontsize=9, va="bottom")
    for frac in ("1/4", "2/7", "1/3", "3/8", "2/5", "3/7"):
        f = Fraction(frac)
        ax.annotate(frac, (float(f), 0.0), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    color=INK2, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0, d.max() * 1.12)
    ax.set_xlabel("每步轉的圈數 α（0.5 以上對稱）", color=INK2)
    ax.set_ylabel("前 1000 點的最小間距（能放多大的球）", color=INK2)
    ax.set_title("能放最多球的比例：有理數 p/q 處掉到谷底（點排成 q 條直線），最大值在黃金比例", fontsize=10, color=INK, loc="left")
    ax = fig.add_subplot(gs[1, 3])
    ax.set_facecolor(SURF)
    w = np.linspace(0.05, 1.0, 2000)
    nn = np.ceil(1 / w - 1e-12)
    ax.plot(w, 2 * nn, color=C1, lw=1.5)
    for name, wm, off in (("A₁,₂（6ω = 1.013）", 0.0285 ** 0.5, (6, 2)), ("Q=2 環", 0.465 ** 0.5, (-14, -14)),
                          ("單一漩渦", 0.781 ** 0.5, (-14, 8)), ("Q=2 環 (2)", 0.893 ** 0.5, (-6, -14))):
        ax.plot([wm], [2 * np.ceil(1 / wm - 1e-12)], "o", color=C2, ms=6, mec=SURF, mew=1.5)
        ax.annotate(name, (wm, 2 * np.ceil(1 / wm - 1e-12)), xytext=off, textcoords="offset points", fontsize=8, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 6, 8, 12, 20, 40])
    ax.set_yticklabels(["2", "4", "6", "8", "12", "20", "40"])
    ax.minorticks_off()
    ax.set_xlabel("內部振動的頻率 ω（門檻 = 1）", color=INK2)
    ax.set_ylabel("輻射功率 ∝ A 的幾次方（2n）", color=INK2)
    ax.set_title("層級：要 n = ⌈1/ω⌉ 倍頻才越過門檻", fontsize=10, color=INK, loc="left")
    fig.suptitle("無理數怎麼堆出圖形（重現影片的兩張圖）以及它在我們模型裡對應什麼", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, out), dpi=115, facecolor=SURF)
    print("wrote", out)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("packing", "all"):
        cmd_packing()
    if what in ("cf", "all"):
        cmd_cf()
    if what in ("staircase", "all"):
        cmd_staircase()
    if what in ("coxeter", "all"):
        cmd_coxeter()
    if what in ("figure", "all"):
        cmd_figure()
