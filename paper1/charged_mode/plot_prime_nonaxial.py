"""
Figure for prime_nonaxial.py and prime_nonaxial_check.py: the lowest Hessian eigenvalue of each axial ring
in the azimuthal sectors k = 1 … 4 (red: a deformation that lowers the energy, blue: stable), with the
axis-localized artifacts of 4 × 2 marked.

Usage: python plot_prime_nonaxial.py [figures/prime_nonaxial_R6.png]
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(fn="figures/prime_nonaxial_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    res = {o["key"]: o for o in json.load(open(os.path.join(HERE, "data", "prime_nonaxial.json")))["results"]}
    loc = {(r["key"], r["k"]): r for r in json.load(open(os.path.join(HERE, "data", "prime_nonaxial_localize.json")))}
    rows = ["3x1", "4x1", "2x2", "5x1", "6x1", "3x2", "7x1", "8x1", "4x2"]
    isprime = lambda q: q > 1 and all(q % j for j in range(2, q))
    K = 4
    A = np.full((len(rows), K), np.nan)
    for i, key in enumerate(rows):
        for k in range(1, K + 1):
            v = np.asarray(res[key]["sectors"][str(k)]["lowest"], float)
            A[i, k - 1] = np.nanmin(v) if np.isfinite(v).any() else 1.0
    fig, ax = plt.subplots(figsize=(9.6, 7.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    norm = TwoSlopeNorm(vmin=-0.45, vcenter=0.0, vmax=1.0)
    ax.imshow(np.clip(A, -0.45, 1.0), cmap=cmap, norm=norm, aspect="auto")
    for i, key in enumerate(rows):
        for k in range(1, K + 1):
            v = A[i, k - 1]
            artifact = key == "4x2" and k in (2, 3)
            if k == 1 and abs(v) < 0.1:
                txt = f"{v:+.3f}\n零模"
            elif v >= 0.995:
                txt = "≥ 1\n（連續譜）"
            else:
                txt = f"{v:+.3f}"
            if artifact:
                lr = loc.get((key, k))
                txt = f"{v:+.2f}\n軸上假象" + (f"\n（管上 {lr['tube']:.0%}）" if lr else "")
                ax.add_patch(plt.Rectangle((k - 1.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor=INK2, lw=0))
            elif v < -0.02 and (key, k) in loc:
                txt += f"\n管上 {loc[(key, k)]['tube']:.0%}"
            ax.text(k - 1, i, txt, ha="center", va="center", fontsize=8.6, color=INK)
    labels = []
    for key in rows:
        m, n = (int(x) for x in key.split("x"))
        q = m * n
        kind = "質數" if isprime(q) else ("單方向" if n == 1 and q > 3 else ("平衡" if n > 1 else ""))
        labels.append(f"{m}×{n}（Q = {q}）{kind}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"k = {k}\n{d}" for k, d in zip(range(1, K + 1), ("平移、傾斜", "橢圓", "三瓣", "四瓣"))], fontsize=9.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("軸對稱環的非軸對稱穩定性：每格是該變形方向的最低 Hessian 本徵值\n紅 = 這樣變形會降低能量（不穩定），藍 = 穩定；"
                 "「管上 x%」是負模落在環的核心管子上的權重", fontsize=10.5, color=INK, loc="left")
    fig.text(0.01, 0.01, "單方向的環（Q×1，Q ≥ 4）都會彎曲或扭轉；平衡的 2×2、3×2 穩定。4×2 找到的負模集中在對稱軸上（離散化病態），它的穩定性尚未判定。",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(HERE, fn), dpi=115, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main(*sys.argv[1:])
