"""
Primes as the "irrational" members of the integers, and a transform that turns them into waves
(the user's suggestion, 2026-10-03: a prime has no divisor before it, nothing pairs with it, like an
irrational ratio that never closes; can the primes be transformed into such patterns?).

1. Irrational frequencies.  Give each prime the frequency log p.  For different primes the ratio
   log p / log q is irrational: log p / log q = a/b would mean p^b = q^a, impossible by unique
   factorisation.  So the primes are a set of mutually incommensurate frequencies, and every integer
   is an integer combination of them: log n = Σ k_p log p (n = Π p^{k_p}) — the integers are the
   combination tones of the primes.
2. Spirals.  Primes at angle = p radians (one radian per step): the arms come from 2π ≈ 44/7 ≈ 710/113;
   the primes occupy only the residue classes coprime to the modulus (2 of 6 arms, 20 of 44, 280 of
   710: Euler's φ), and spread evenly over them (Dirichlet).  The golden angle does not hide this:
   on the sunflower (n at angle 2πn/φ², radius √n) the visible arms are Fibonacci in number (89 for
   n ≈ 500–1500, 144 for n ≈ 2000–6000), and the primes can sit only on the arms coprime to it — 48
   of the 144 — so the middle ring shows streaks, the 89- and 233-rings (prime arm numbers) do not.
3. The transform.  F(t) = Σ_n Λ(n) n^{−1/2} e^{−i t log n} e^{−n/X} (Λ(p^k) = log p, zero otherwise): all
   primes sounding together, each at its frequency log p, softly cut off at X.  By the explicit formula
   (a theorem) its peaks sit at the imaginary parts of the zeros of the Riemann zeta function,
   14.13, 21.02, 25.01, …, with height ≈ log X and width ≈ 2π/log X: the primes, transformed, are a
   spectrum of resonance frequencies.  (A sharp cut-off at X leaves the main term √X/t, which swamps
   the peaks for large X; the soft cut-off makes it ∝ e^{−πt/2}.)

Usage: python primes_spectrum.py [X]
"""

import json
import os
import sys
from math import gcd

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def sieve(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0]


def lambda_terms(P, M):
    """The prime powers n = p^k <= M with the von Mangoldt weight log p."""
    n, lam = [], []
    Pf = P.astype(float)
    for k in range(1, 64):
        sel = Pf ** k <= M
        if not sel.any():
            break
        n.append(Pf[sel] ** k)
        lam.append(np.log(Pf[sel]))
    n, lam = np.concatenate(n), np.concatenate(lam)
    o = np.argsort(n)
    return n[o], lam[o]


def spectrum(n, lam, X, t, chunk=2000):
    """F(t) = sum_n Lambda(n) n^(-1/2 - i t) e^(-n/X).  The smooth cut-off e^(-n/X) removes the
    main term: by the explicit formula F = -zeta'/zeta(s) + Gamma(1-s) X^(1-s) - sum_rho Gamma(rho-s) X^(rho-s)
    + O(1/X) at s = 1/2 + it, and |Gamma(1/2 - it)| ~ e^(-pi t/2) kills the second term for t > 5;
    at a zero rho = 1/2 + i gamma the pole of -zeta'/zeta cancels against Gamma(rho - s), leaving
    |F(gamma)| ~ log X - 0.58, a peak of width ~ 2 pi / log X."""
    sel = n <= 30 * X
    n, lam = n[sel], lam[sel]
    w = lam / np.sqrt(n) * np.exp(-n / X)
    ln = np.log(n)
    F = np.zeros(len(t), complex)
    for s in range(0, len(n), chunk):
        F += np.exp(-1j * np.outer(t, ln[s:s + chunk])) @ w[s:s + chunk]
    return F


def peaks_of(t, A, frac=0.45):
    idx = [i for i in range(1, len(t) - 1) if A[i] > A[i - 1] and A[i] >= A[i + 1] and A[i] > frac * A.max()]
    out = []
    for i in idx:                                          # parabolic refinement
        y0, y1, y2 = A[i - 1], A[i], A[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        out.append((float(t[i] + d * (t[1] - t[0])), float(y1 - 0.25 * (y0 - y2) * d)))
    return out


def main(X=100_000):
    M = max(1_000_000, 12 * X)
    P = sieve(M)
    out = dict(X=X, n_primes=int(len(P)), sieve_to=M)
    print(f"{len(P)} primes up to {M}")
    # 2. residue classes
    res = {}
    for q in (6, 44, 710):
        cnt = np.bincount(P[P > q] % q, minlength=q)
        occ = [int(r) for r in range(q) if cnt[r] > 0]
        cop = [r for r in range(q) if gcd(r, q) == 1]
        spread = float(cnt[cop].std() / cnt[cop].mean())
        res[q] = dict(occupied=len(occ), coprime=len(cop), relative_spread=spread)
        print(f"   mod {q:3d}: primes occupy {len(occ):3d} classes; phi({q}) = {len(cop):3d}; relative spread over them {spread:.3f}")
    out["residues"] = res
    # 3. the transform, for three cut-offs: the peaks grow as more primes join
    n, lam = lambda_terms(P, M)
    t = np.arange(5.0, 60.0, 0.02)
    curves = {}
    for Xc in (1_000, 10_000, X):
        curves[Xc] = np.abs(spectrum(n, lam, Xc, t))
    A = curves[X]
    pk = peaks_of(t, A)
    try:
        import mpmath
        zeros = [float(mpmath.zetazero(k).imag) for k in range(1, 14)]
    except ImportError:
        zeros = [14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719, 43.327073, 48.005151,
                 49.773832, 52.970321, 56.446248, 59.347044]
    match = []
    for z in zeros:
        near = min(pk, key=lambda p: abs(p[0] - z))
        match.append((z, near[0], near[0] - z, near[1]))
    spurious = [p for p in pk if min(abs(p[0] - z) for z in zeros) > 0.3]
    print(f"   cut-off X = {X}: {len(pk)} peaks above 45 % of the highest; log X - 0.577 = {np.log(X) - 0.5772:.2f}")
    for z, p, d, h in match:
        print(f"      zero {z:9.4f}   peak {p:8.3f}   difference {d:+.3f}   height {h:6.2f}")
    print("      peaks not at a zero:", [(round(a, 2), round(b, 2)) for a, b in spurious])
    for Xc in (1_000, 10_000):
        pc = peaks_of(t, curves[Xc])
        hit = sum(1 for z in zeros if min(abs(p[0] - z) for p in pc) < 0.3)
        print(f"   cut-off X = {Xc}: {len(pc)} peaks, {hit} of {len(zeros)} zeros found within 0.3")
    out.update(t=t.tolist(), absF={str(k): v.tolist() for k, v in curves.items()}, peaks=pk, zeros=zeros,
               match=match, spurious=spurious)
    json.dump(out, open(os.path.join(DATA, "primes_spectrum.json"), "w"))
    figure(P, out)


def figure(P, out, fn="figures/primes_spectrum_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig = plt.figure(figsize=(13, 7.4), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95])
    phi = (1 + 5 ** 0.5) / 2
    isp = np.zeros(int(P[-1]) + 1, bool)
    isp[P] = True
    GREY = "#d9d7d1"
    # (a) angle = p radians, p <= 3000: the arms of 44/7 ≈ 2π, primes on 20 of the 44
    ax = fig.add_subplot(gs[0, 0])
    n = np.arange(1, 3001).astype(float)
    q = P[P <= 3000].astype(float)
    ax.scatter(n * np.cos(n), n * np.sin(n), s=0.6, color=GREY, lw=0)
    ax.scatter(q * np.cos(q), q * np.sin(q), s=2.2, color=INK, lw=0)
    ax.set_title("角度 = p 弧度（p ≤ 3000）：44 條臂（44/7 ≈ 2π）\n只有 20 條有質數（φ(44) = 20）", fontsize=9, color=INK)
    # (b) zoom on 5·10⁵ … 10⁶ near angle 0: the 710 rays of 710/113 ≈ 2π, primes on 280 of them
    ax = fig.add_subplot(gs[0, 1])
    n = np.arange(400_000, 1_000_001).astype(float)
    x, y = n * np.cos(n), n * np.sin(n)
    sel = (x > 5e5) & (np.abs(y) < 1.6e5)
    ax.scatter(x[sel], y[sel], s=0.25, color=GREY, lw=0)
    sel &= isp[n.astype(int)]
    ax.scatter(x[sel], y[sel], s=0.9, color=INK, lw=0)
    ax.set_title("放大 50 萬到 100 萬的一角：近乎直線的 710 條射線\n（710/113 ≈ 2π），只有 280 條有質數（φ(710) = 280）", fontsize=9, color=INK)
    # (c) golden angle (sunflower), integers <= 10000 grey, primes black
    ax = fig.add_subplot(gs[0, 2])
    n = np.arange(1, 10_001).astype(float)
    th = 2 * np.pi * n / phi ** 2
    ax.scatter(np.sqrt(n) * np.cos(th), np.sqrt(n) * np.sin(th), s=0.7, color=GREY, lw=0)
    q = n[isp[n.astype(int)]]
    th = 2 * np.pi * q / phi ** 2
    ax.scatter(np.sqrt(q) * np.cos(th), np.sqrt(q) * np.sin(th), s=1.6, color=INK, lw=0)
    ax.set_title("黃金角（向日葵）：中段看到 144 條臂，質數只能落在\n其中 48 條（2、3 的倍數那 96 條全空）→ 條紋", fontsize=9, color=INK)
    for ax in fig.axes:
        ax.set_aspect("equal")
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(SURF)
    t = np.array(out["t"])
    A = np.array(out["absF"][str(out["X"])])
    for z in out["zeros"]:
        ax.axvline(z, color=C2, lw=1, alpha=0.8)
    for Xc, c, lw in (("1000", "#b9c9e0", 0.9), ("10000", "#7fa3d4", 1.0)):
        ax.plot(t, out["absF"][Xc], color=c, lw=lw, label=f"X = {int(Xc):,}")
    ax.plot(t, A, color=C1, lw=1.3, label=f"X = {out['X']:,}")
    ax.legend(loc="center left", bbox_to_anchor=(0.005, 0.55), frameon=False, fontsize=9)
    ax.annotate("橙線：黎曼 ζ 函數零點的虛部（14.13, 21.02, 25.01, …）", (5.5, A.max() * 1.02), color=INK, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlim(5, 60)
    ax.set_xlabel("頻率 t", color=INK2)
    ax.set_ylabel("|F(t)|", color=INK2)
    ax.set_title(f"把質數變成波：每個質數 p 以頻率 log p 一起振動（權重 e^(−p/X)；含 p², p³…），越多質數加入，尖峰越尖，全落在 ζ 零點上",
                 fontsize=10, color=INK, loc="left")
    fig.suptitle("質數是整數裡「無理」的成員：頻率 log p 彼此沒有有理比例；排成螺旋、再變成共振頻譜", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, fn), dpi=115, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
