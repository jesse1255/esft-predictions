"""
Primes as the digits of an irrational number (the user's question, 2026-10-03: like 3.14159…, put the
primes in as the digits; what kind of number is that; can a new one be invented?).  Small calculations
only (seconds).

Known ways (literature checked 2026-10-03):
1. decimal digits    C = 0.2357111317192329… (Copeland & Erdős 1946): proven normal in base 10 (every
                     block of digits equally frequent), hence irrational.  For π this is still open.
2. binary places     ρ = Σ_p 2^(−p) = 0.4146825098…: binary digit 1 at the prime places.
3. mixed radix       λ = Σ_n (p_n − 1)/(p_1 ⋯ p_{n−1}) = 2.920050977316… (Fridman, Garbulsky, Glecer, Grime,
                     Tron Florentin, Amer. Math. Monthly 126 (2019) 70): f_1 = λ, f_{n+1} = ⌊f_n⌋(f_n − ⌊f_n⌋ + 1)
                     gives ⌊f_n⌋ = 2, 3, 5, 7, 11, … (Bertrand: p_{n+1} < 2 p_n).  Irrational (proved there).
4. partial quotients α_P = [0; 2, 3, 5, 7, 11, …] = 0.43233208718…  (studied, e.g. arXiv:1003.4015);
                     irrational because the continued fraction never ends.
Proposed here (novelty not checked):
5. gaps as partial quotients  α_g = [0; 1, 2, 2, 4, 2, 4, 2, 4, 6, 2, 6, …], g_n = p_{n+1} − p_n
                     (twin primes are the 2's).
6. the dictionary the other way: x = [0; a_1, a_2, a_3, …] ↦ N(x) = 2^a_1 3^a_2 5^a_3 ⋯ : rationals ↦ ordinary
                     integers, irrationals ↦ infinite products of primes, the golden ratio ↦ 2·3·5·7·11⋯ (every
                     prime once), π − 3 = [0; 7, 15, 1, 292, …] ↦ 2^7 3^15 5 7^292 ⋯ (292: the 355/113).
Test: the sunflower packing of §5.12 (points at radius √n, angle 2π n x): how "irrational" each number is,
level by level.  A partial quotient a_{k+1} makes spokes with q_k arms and drops the smallest distance to
about √(2π / a_{k+1}) once n passes q_k q_{k+1} / (4π).

Usage: python primes_digits.py
"""

import json
import os

import numpy as np

from primes_spectrum import sieve
from irrational_patterns import dmin

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PHI = (1 + 5 ** 0.5) / 2


def constants(dps=320):
    import mpmath as mp
    mp.mp.dps = dps
    P = [int(p) for p in sieve(20000)]
    s = ""
    for p in P:
        s += str(p)
        if len(s) > dps + 20:
            break
    C = mp.mpf("0." + s)
    rho = mp.fsum(mp.mpf(2) ** (-p) for p in P if p < 3.4 * dps + 50)
    lam, prim = mp.mpf(0), mp.mpf(1)
    for p in P:
        lam += (p - 1) / prim
        prim *= p
        if prim > mp.mpf(10) ** (dps + 20):
            break

    def cf_value(a):
        x = mp.mpf(0)
        for q in reversed(a):
            x = 1 / (q + x)
        return x
    alpha_P = cf_value(P[:400])
    gaps = [P[i + 1] - P[i] for i in range(400)]
    alpha_g = cf_value(gaps)
    return mp, P, gaps, dict(copeland_erdos=C, prime_constant=rho, buenos_aires=lam, cf_primes=alpha_P, cf_gaps=alpha_g)


def cf_terms(mp, x, n=24):
    a = []
    x = x - mp.floor(x)
    for _ in range(n):
        if x == 0:
            break
        x = 1 / x
        q = int(mp.floor(x))
        a.append(q)
        x -= q
    return a


def denominators(a):
    q0, q1, out = 0, 1, []
    for t in a:
        q0, q1 = q1, t * q1 + q0
        out.append(q1)
    return out


def main():
    mp, P, gaps, K = constants()
    out = dict(values={k: mp.nstr(v, 30) for k, v in K.items()})
    print("values:")
    for k, v in K.items():
        print(f"   {k:15s} {mp.nstr(v, 30)}")
    # the Buenos Aires constant gives the primes back, as many as its digits hold
    f, got = K["buenos_aires"], []
    for i in range(400):
        fl = int(mp.floor(f))
        if fl != P[i]:
            break
        got.append(fl)
        f = fl * (f - fl + 1)
    print(f"   Buenos Aires constant with {mp.mp.dps} digits gives back the first {len(got)} primes (up to {got[-1]});"
          f" the digits of those primes total {sum(len(str(p)) for p in got)}")
    out["buenos_aires_primes_recovered"] = len(got)
    # continued fractions: the "digits" that do not depend on the base
    terms = {k: cf_terms(mp, v) for k, v in K.items()}
    terms["golden"] = [1] * 24
    out["cf"] = terms
    print("continued fractions (partial quotients of the fractional part):")
    for k, a in terms.items():
        print(f"   {k:15s} {a[:16]}   spokes (denominators) {denominators(a)[:7]}")
    # the sunflower test
    turns = {"golden": 1 / PHI ** 2, "cf_primes": float(K["cf_primes"]), "cf_gaps": float(K["cf_gaps"]),
             "copeland_erdos": float(K["copeland_erdos"]), "prime_constant": float(K["prime_constant"]),
             "buenos_aires": float(K["buenos_aires"] - 2)}
    Ns = np.unique(np.round(np.logspace(1, np.log10(5000), 45)).astype(int))
    curves = {k: [dmin(a, int(N)) for N in Ns] for k, a in turns.items()}
    print("smallest distance among the first N points (sunflower), golden = reference:")
    for k in turns:
        print(f"   {k:15s} N = 100: {dmin(turns[k], 100):.3f}   N = 1000: {dmin(turns[k], 1000):.3f}   N = 5000: {dmin(turns[k], 5000):.3f}")
    # paper estimate for alpha_P: d drops to ~ sqrt(2 pi / p_{k+1}) once n > q_k q_{k+1} / (4 pi)
    q = denominators(terms["cf_primes"])
    est = [(q[k] * q[k + 1] / (4 * np.pi), (2 * np.pi / terms["cf_primes"][k + 1]) ** 0.5) for k in range(5)]
    print("   cf_primes paper estimate (n where the level sets in, distance):", [(round(a), round(b, 2)) for a, b in est])
    out.update(turns=turns, N=Ns.tolist(), dmin=curves, cf_primes_estimate=est)
    # the dictionary x = [0; a1, a2, ...] -> 2^a1 3^a2 5^a3 ...
    dic = {}
    for name, x in (("golden 1/phi", 1 / mp.phi), ("pi - 3", mp.pi - 3), ("e - 2", mp.e - 2), ("sqrt2 - 1", mp.sqrt(2) - 1),
                    ("355/113 - 3", mp.mpf(355) / 113 - 3)):
        a = cf_terms(mp, x, 8)
        a = [t for t in a if t < 10 ** 6]
        dic[name] = " · ".join(f"{p}^{t}" if t > 1 else f"{p}" for p, t in zip(P, a)) + (" ⋯" if len(a) >= 6 else "")
        print(f"   N({name}) = {dic[name]}")
    out["dictionary"] = dic
    json.dump(out, open(os.path.join(DATA, "primes_digits.json"), "w"), indent=1)
    figure(out)


def figure(out, fn="figures/primes_digits_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from irrational_patterns import points
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"golden": "#0b0b0b", "cf_primes": "#eb6834", "cf_gaps": "#1baf7a", "copeland_erdos": "#2a78d6",
           "prime_constant": "#9b59b6", "buenos_aires": "#c9a227"}
    NAME = {"golden": "黃金比例 [0; 1, 1, 1, …]", "cf_primes": "質數當連分數 [0; 2, 3, 5, 7, 11, …]",
            "cf_gaps": "質數間隔當連分數 [0; 1, 2, 2, 4, 2, …]（新）", "copeland_erdos": "質數串成小數 0.2357111317…",
            "prime_constant": "二進位質數常數 0.41468…", "buenos_aires": "布宜諾斯艾利斯常數 2.92005…"}
    fig = plt.figure(figsize=(14, 7.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
    for j, k in enumerate(("golden", "cf_primes", "cf_gaps", "copeland_erdos")):
        ax = fig.add_subplot(gs[0, j])
        p = points(out["turns"][k], 2000)
        ax.scatter(p[:, 0], p[:, 1], s=2.0, color=COL[k], lw=0)
        ax.set_aspect("equal")
        ax.axis("off")
        a = out["cf"][k][:7]
        ax.set_title(NAME[k] + "\n連分數 " + ", ".join(str(t) for t in a) + " …", fontsize=8.8, color=INK)
    ax = fig.add_subplot(gs[1, :])
    ax.set_facecolor(SURF)
    for k in NAME:
        ax.plot(out["N"], out["dmin"][k], color=COL[k], lw=2.2 if k in ("golden", "cf_primes", "cf_gaps") else 1.2,
                label=NAME[k])
    for n, d in out["cf_primes_estimate"][1:4]:
        ax.plot([n], [d], marker="v", color=COL["cf_primes"], ms=7)
    ax.set_xscale("log")
    ax.set_xlim(10, 5000)
    ax.set_xlabel("點數 N（越往右，看的層級越深）", color=INK2)
    ax.set_ylabel("最小間距（越大，能放的球越大）", color=INK2)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title("向日葵測試：每一層的部分商 a 越大，那一層越像有理數（排成直線），間距掉到約 √(2π/a)；▼ 是質數連分數的紙上估計",
                 fontsize=10, color=INK, loc="left")
    fig.suptitle("把質數排成一個無理數的「位數」：小數、二進位、混合進位、連分數；用 §5.12 的堆積測試看它們有多「無理」",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(HERE, fn), dpi=115, facecolor=SURF)
    print("wrote", fn)


if __name__ == "__main__":
    main()
