"""
Which waves correspond to the zeros of the Riemann zeta function (the user's question, 2026-10-03:
the peaks of the prime spectrum sit exactly on the zeta zeros; what actual wave is that — not a fit).
Small calculations only (the 1000 zeros take about a minute, then everything is seconds).

1. The grating.  F(t) of primes_spectrum.py is literally the far-field (Fraunhofer) amplitude of a
   grating with slits at the positions log n (n = p^k), each passing an amplitude Λ(n)/√n: its bright
   lines are the zeros.  Swap the roles: a grating with slits at the zeros γ_k diffracts into bright
   lines at log 2, log 3, log 4, log 5, log 7, … — the zeros are a one-dimensional quasicrystal whose
   diffraction pattern is the primes (Dyson).
2. The identity (Guinand–Weil explicit formula, a theorem; no parameter):
       Σ_γ h(γ) = h(i/2) + h(−i/2) − g(0) log π + (1/2π) ∫ h(r) Re ψ(1/4 + ir/2) dr − 2 Σ_n Λ(n) n^(−1/2) g(log n),
   g(u) = (1/2π) ∫ h(r) e^(−iru) dr, sum over all zeros ±γ.  With h(r) = e^(−(r/T)²) cos(r u):
       Φ(u) ≡ Σ_{γ>0} e^(−(γ/T)²) cos(γ u)
            = e^(1/4T²) cosh(u/2) + ½ A(u) − ½ log π G(u) − ½ Σ_n Λ(n) n^(−1/2) [G(u − log n) + G(u + log n)],
       G(v) = T/(2√π) e^(−T²v²/4),  A(u) = (1/2π) ∫ e^(−(r/T)²) cos(ru) Re ψ(1/4 + ir/2) dr.
   Left side from the zeros only, right side from the primes only: they must agree point by point.
3. The waves of the prime staircase (Riemann, von Mangoldt):
       ψ(x) = Σ_{n ≤ x} Λ(n) = x − Σ_ρ x^ρ/ρ − log 2π − ½ log(1 − x^(−2)),
   each zero pair is one wave 2√x cos(γ log x − arg ρ)/|ρ| in log x; adding them rebuilds the steps.
4. What kind of physical spectrum: nearest-neighbour spacings of the zeros (unfolded with the
   Riemann–von Mangoldt count) against Poisson, GOE and GUE (Wigner surmises): GUE is the statistics
   of energy levels of chaotic quantum systems without time-reversal symmetry (Montgomery, Odlyzko).
5. The other wave: primes at their plain positions p (not log p).  The structure factor
   S(k) = |Σ_p e^(ikp)|² / π(X) has Bragg peaks at k = 2π m/q with S = π(X) (μ(q)/φ(q))² (Ramanujan sum +
   Dirichlet): q = 2 → 1, q = 3, 6 → 1/4, q = 5, 10 → 1/16, and NO peak at q = 4, 8, 9 (μ = 0).  This is the
   spirals of primes_spectrum.py as a diffraction pattern (Torquato, Zhang, de Courcy-Ireland 2018).

Usage: python zeta_waves.py zeros [K]   → data/zeta_zeros.json (cached)
       python zeta_waves.py all
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ZFILE = os.path.join(DATA, "zeta_zeros.json")


def _zero(k):
    import mpmath
    mpmath.mp.dps = 20
    return float(mpmath.zetazero(k).imag)


def cmd_zeros(K=1000, procs=3):
    from multiprocessing import Pool
    with Pool(procs) as pool:
        z = pool.map(_zero, range(1, K + 1), chunksize=20)
    json.dump(dict(K=K, gamma=z), open(ZFILE, "w"))
    print(f"{K} zeros, last {z[-1]:.6f}")
    return np.array(z)


def zeros():
    return np.array(json.load(open(ZFILE))["gamma"])


def mangoldt(M):
    from primes_spectrum import sieve, lambda_terms
    P = sieve(M)
    return P, lambda_terms(P, M)


def identity(g, u, T):
    """Left side (zeros) and right side (primes) of the windowed explicit formula at the points u."""
    left = (np.exp(-(g[None, :] / T) ** 2) * np.cos(np.outer(u, g))).sum(axis=1)
    _, (n, lam) = mangoldt(int(np.exp(u.max() + 12.0 / T)) + 2)
    G = lambda v: T / (2 * np.sqrt(np.pi)) * np.exp(-T ** 2 * v ** 2 / 4)
    ln = np.log(n)
    right = np.exp(1 / (4 * T ** 2)) * np.cosh(u / 2) - 0.5 * np.log(np.pi) * G(u)
    right -= 0.5 * ((lam / np.sqrt(n))[None, :] * (G(u[:, None] - ln[None, :]) + G(u[:, None] + ln[None, :]))).sum(axis=1)
    # A(u): (1/2π) ∫ e^(−(r/T)²) cos(ru) Re ψ(1/4 + ir/2) dr, by quadrature on a fine grid
    from scipy.special import psi as digamma
    r = np.linspace(0.0, 6.0 * T, 60001)
    w = np.exp(-(r / T) ** 2) * np.real(digamma(0.25 + 0.5j * r))
    A = np.array([np.trapezoid(w * np.cos(r * x), r) for x in u]) / np.pi      # ∫_{−∞}^{∞} = 2 ∫_0^∞, times 1/2π
    right += 0.5 * A
    return left, right


def staircase(g, x):
    """ψ(x) from the zeros: x − Σ_ρ x^ρ/ρ − log 2π − ½ log(1 − x^(−2)), pairs ρ, ρ̄."""
    rho = 0.5 + 1j * g
    lx = np.log(x)
    waves = 2 * np.real(np.exp(np.outer(lx, rho)) / rho[None, :])
    return x - waves.sum(axis=1) - np.log(2 * np.pi) - 0.5 * np.log(1 - x ** -2.0)


def psi_exact(x):
    _, (n, lam) = mangoldt(int(x.max()) + 1)
    c = np.concatenate([[0.0], np.cumsum(lam)])
    return c[np.searchsorted(n, x, side="right")]


def spacings(g):
    """Unfolded nearest-neighbour spacings: N(T) = (T/2π) log(T/2πe) + 7/8."""
    Nt = g / (2 * np.pi) * np.log(g / (2 * np.pi * np.e)) + 7 / 8
    return np.diff(Nt)


def wigner(s, beta):
    if beta == 1:
        return np.pi / 2 * s * np.exp(-np.pi * s ** 2 / 4)
    return 32 / np.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / np.pi)


def bragg(X=100_000):
    from primes_spectrum import sieve
    from math import gcd
    P = sieve(X).astype(float)
    out = []
    for q in range(2, 13):
        for m in range(1, q):
            if gcd(m, q) != 1:
                continue
            S = abs(np.exp(2j * np.pi * m * P / q).sum()) ** 2 / len(P)
            out.append((m, q, S / len(P)))
    return len(P), out


def main():
    g = zeros()
    K = len(g)
    out = dict(K=K)
    # 2. the identity, with the window as wide as the zeros allow (weight at γ_K: e^-9)
    T = g[-1] / 3.0
    u = np.linspace(0.3, 4.2, 3901)
    left, right = identity(g, u, T)
    err = np.abs(left - right).max()
    print(f"identity with K = {K} zeros, T = {T:.1f}: max |zeros − primes| = {err:.2e} (spike depth at log 2: {left[np.argmin(np.abs(u - np.log(2)))]:.2f})")
    # where are the dips of the zero side, and are they all prime powers?
    smooth = np.exp(1 / (4 * T ** 2)) * np.cosh(u / 2)          # the smooth term of the formula (no parameter)
    dl = left - smooth
    dips = [i for i in range(1, len(u) - 1) if dl[i] < dl[i - 1] and dl[i] <= dl[i + 1] and dl[i] < -3.0]
    xs = np.exp(u[dips])
    _, (n, lam) = mangoldt(80)
    pp = [int(v) for v in n if v <= np.exp(4.2)]
    found = sorted({int(round(x)) for x in xs})
    print(f"   dips of the zero sum, minus cosh(u/2), below −3 (x = e^u): {found}")
    print(f"   prime powers up to e^4.2 = {np.exp(4.2):.1f}: {pp}")
    print(f"   every dip a prime power: {set(found) <= set(pp)};  every prime power found: {set(pp) <= set(found)}")
    out.update(T=T, max_err=float(err), dips=found, prime_powers=pp)
    # the build-up: 1, 10, 100, K zeros (window scaled to each)
    build = {}
    for k in (1, 10, 100, K):
        Tk = g[k - 1] / 3.0 if k > 1 else g[0]
        build[k] = (np.exp(-(g[None, :k] / Tk) ** 2) * np.cos(np.outer(u, g[:k]))).sum(axis=1)
    out["u"] = np.round(u, 4).tolist()
    out["build"] = {str(k): np.round(v, 4).tolist() for k, v in build.items()}
    out["right"] = np.round(right, 4).tolist()
    # 3. the staircase
    x = np.linspace(1.5, 60.0, 5000)
    ex = psi_exact(x)
    stair = {}
    for k in (10, 100, K):
        sk = staircase(g[:k], x)
        stair[k] = sk
        print(f"   staircase psi(x) on [1.5, 60] from {k:4d} zeros: rms error {np.sqrt(np.mean((sk - ex) ** 2)):.3f}")
    out["x"] = np.round(x, 4).tolist()
    out["psi"] = np.round(ex, 4).tolist()
    out["stair"] = {str(k): np.round(v, 4).tolist() for k, v in stair.items()}
    # 4. spacings
    s = spacings(g)
    s = s / s.mean()
    small = float(np.mean(s < 0.25))
    from scipy import stats
    cdf_gue = lambda v: np.array([np.trapezoid(wigner(np.linspace(0, a, 400), 2), np.linspace(0, a, 400)) for a in np.atleast_1d(v)])
    cdf_goe = lambda v: 1 - np.exp(-np.pi * np.atleast_1d(v) ** 2 / 4)
    ks = {name: float(stats.kstest(s, f).statistic) for name, f in
          (("Poisson", lambda v: 1 - np.exp(-np.atleast_1d(v))), ("GOE", cdf_goe), ("GUE", cdf_gue))}
    print(f"   spacings (unfolded, mean 1): fraction below 0.25 = {small:.3f} (Poisson 0.221, GOE 0.048, GUE 0.017);"
          f" Kolmogorov–Smirnov distance {ks}")
    out.update(spacings=np.round(s, 5).tolist(), small=small, ks=ks)
    # 5. Bragg peaks of the primes at plain positions
    nP, br = bragg()
    def factor(q):
        f, p = {}, 2
        while p * p <= q:
            while q % p == 0:
                f[p] = f.get(p, 0) + 1
                q //= p
            p += 1
        if q > 1:
            f[q] = f.get(q, 0) + 1
        return f

    def mobius(q):
        f = factor(q)
        return 0 if any(e > 1 for e in f.values()) else (-1) ** len(f)

    def totient(q):
        t = q
        for p in factor(q):
            t = t // p * (p - 1)
        return t
    print(f"   Bragg peaks of the {nP} primes up to 1e5 at k = 2π m/q (S/π(X), theory (μ(q)/φ(q))²):")
    rows = []
    for m, q, S in br:
        th = (int(mobius(q)) / int(totient(q))) ** 2
        rows.append((m, q, S, th))
    for q in range(2, 13):
        r = [x for x in rows if x[1] == q]
        print(f"      q = {q:2d}: measured {np.mean([x[2] for x in r]):.4f} (spread {np.std([x[2] for x in r]):.1e}),"
              f" theory {r[0][3]:.4f}")
    out["bragg"] = rows
    json.dump(out, open(os.path.join(DATA, "zeta_waves.json"), "w"))
    figure(out, g)


def figure(out, g, fn="figures/zeta_waves_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["path.simplify"] = False
    SURF, INK, INK2, GRID, C1, C2, C3 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a"
    fig = plt.figure(figsize=(14, 10.2), facecolor=SURF)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)

    u = np.array(out["u"])
    x = np.exp(u)
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    off = 0.0
    for k, c in (("1", "#b9c9e0"), ("10", "#8fb0dc"), ("100", "#5a8fd6"), (str(out["K"]), C1)):
        y = np.array(out["build"][k])
        y = y / np.abs(y).max()
        ax.plot(x, y - off, color=c, lw=1.0)
        ax.annotate(f"{k} 個零點", (70, -off), color=INK2, fontsize=9, va="center")
        off += 2.2
    for n in out["prime_powers"]:
        ax.axvline(n, color=C2, lw=0.6, alpha=0.45, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1.35, 100)
    ax.set_yticks([])
    ax.set_xticks([2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 19, 23, 25, 27, 29, 31, 32, 37, 41, 43, 47, 49, 53, 59, 61, 64])
    ax.set_xticklabels([str(v) for v in ax.get_xticks()], fontsize=7.5)
    ax.minorticks_off()
    ax.set_xlabel("x（對數刻度）；橙線：質數與質數的次方", color=INK2)
    ax.set_title("反過來：每個 ζ 零點 γ 是一個波 cos(γ log x)；把它們加起來，尖峰只出現在質數和質數的次方（沒有任何參數）",
                 fontsize=10.5, color=INK, loc="left")
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    sel = (u > 0.55) & (u < 1.75)
    ax.plot(x[sel], np.array(out["build"][str(out["K"])])[sel], color=C1, lw=2.4, label=f"只用 {out['K']} 個零點")
    ax.plot(x[sel], np.array(out["right"])[sel], color=C2, lw=1.0, ls="--", label="只用質數（顯式公式）")
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 4, 5])
    ax.set_xticklabels(["2", "3", "4", "5"])
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"同一個量的兩種算法逐點重合（最大差 {out['max_err']:.1e}）", fontsize=10, color=INK, loc="left")
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    xx = np.array(out["x"])
    ax.plot(xx, out["psi"], color=INK, lw=1.2, label="真的質數階梯 ψ(x)")
    for k, c in (("10", "#9fc0e8"), (str(out["K"]), C1)):
        ax.plot(xx, out["stair"][k], color=c, lw=1.0, label=f"由 {k} 個零點的波重建")
    ax.set_xlim(1.5, 60)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_xlabel("x", color=INK2)
    ax.set_title("質數階梯 ψ(x) = x − Σ 2√x cos(γ log x − arg ρ)/|ρ| − …", fontsize=10, color=INK, loc="left")
    ax = fig.add_subplot(gs[2, 0])
    style(ax)
    s = np.array(out["spacings"])
    ax.hist(s, bins=np.linspace(0, 3, 31), density=True, color="#c9d8ee", edgecolor=SURF)
    sv = np.linspace(0, 3, 300)
    ax.plot(sv, np.exp(-sv), color=INK2, lw=1.2, ls=":", label="Poisson（不相關）")
    ax.plot(sv, wigner(sv, 1), color=C3, lw=1.4, ls="--", label="GOE（混沌、有時間反演）")
    ax.plot(sv, wigner(sv, 2), color=C2, lw=1.8, label="GUE（混沌、沒有時間反演）")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("相鄰零點的間距（平均 = 1）", color=INK2)
    ax.set_title(f"零點的間距：像量子混沌系統的能階（{out['K']} 個零點）", fontsize=10, color=INK, loc="left")
    ax = fig.add_subplot(gs[2, 1])
    style(ax)
    for m, q, S, th in out["bragg"]:
        ax.plot([m / q, m / q], [0, S], color=C1, lw=2.2)
        ax.plot([m / q], [th], marker="_", color=C2, ms=12, mew=2)
    from math import gcd
    for q in (4, 8, 9):
        for m in range(1, q):
            if gcd(m, q) == 1:
                ax.annotate("×", (m / q, 0.0105), color=C2, ha="center", va="center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 1.5)
    ax.set_yticks([0.01, 0.1, 1.0])
    ax.set_yticklabels(["0.01", "0.1", "1"])
    ax.minorticks_off()
    ax.set_ylabel("峰高 S/π(X)", color=INK2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("波數 k/2π", color=INK2)
    ax.set_title("另一種波：質數放在原來的位置 p 去繞射：峰在分數 m/q，高度 (μ(q)/φ(q))²（橙），q = 4、8、9 沒有峰（×）",
                 fontsize=9.3, color=INK, loc="left")
    fig.suptitle("ζ 零點對應的波：零點和質數是一對「晶體與繞射圖樣」；零點的排列像量子混沌的能階", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(HERE, fn), dpi=110, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "zeros":
        cmd_zeros(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    else:
        main()
