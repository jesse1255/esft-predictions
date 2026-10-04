"""
The zeta zeros as one sound (the user's remark, 2026-10-04: not one frequency matching each zero, but
the whole spectrum fused together — does it gather into a beam?  it looks like sound or music; a spectrum
gathered, not dispersed).  Small calculations only (seconds).

A(u) = (1/K) Σ_k e^{i(f_k u + φ_k)}: K = 1000 frequencies sounding together with equal amplitude.
For the zeta zeros (f_k = γ_k, all in phase at u = 0) the envelope |A(u)| gathers into sharp pulses
exactly at u = log n, n = p^k (the explicit formula: height ≈ (Γ/2πK) Λ(n)/√n, Γ = γ_K, no parameter).
Controls with K frequencies each:
  random phases   the same frequencies with scrambled phases (a dispersed spectrum)
  GUE             random-matrix levels unfolded to the zeros' density (same local statistics, no arithmetic)
  Poisson         independent levels at the same density
  laser comb      equally spaced over the same band: a mode-locked laser, pulses periodic in 2π/Δ
Sound: the sums played with u = 0.25 + 3t (frequencies γ·3/2π = 7–680 Hz; the start skips the u = 0 pulse),
so the clicks come at t = (log n − 0.25)/3 seconds.

Usage: python zeta_sound.py           → pulses, controls, sound files
       python zeta_sound.py cancel    → where each spectrum puts its weight (resonance vs cancellation)
"""

import json
import os

import numpy as np

from zeta_waves import zeros

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def counting(T):
    return T / (2 * np.pi) * np.log(T / (2 * np.pi * np.e)) + 7 / 8


def to_heights(x, g):
    """Unit-mean-spacing levels x (x[0] = 0) mapped to the zeros' mean density: T = N⁻¹(N(γ₁) + x)."""
    T = np.linspace(10.0, 1.2 * g[-1], 400001)
    return np.interp(counting(g[0]) + x, counting(T), T)


def spectra(g, seed=7):
    rng = np.random.default_rng(seed)
    K = len(g)
    out = {"zeros": (g, np.zeros(K)), "random_phase": (g, rng.uniform(0, 2 * np.pi, K))}
    n = 2000
    A = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    ev = np.linalg.eigvalsh((A + A.conj().T) / 2)
    c = ev[n // 2 - 600: n // 2 + 600]                      # the flat middle of the semicircle
    x = np.polyval(np.polyfit(c, np.arange(len(c)), 9), c)  # unfolded: mean spacing 1
    out["gue"] = (to_heights(x[:K] - x[0], g), np.zeros(K))
    s = rng.exponential(1.0, K)
    out["poisson"] = (to_heights(np.concatenate([[0.0], np.cumsum(s[:-1])]), g), np.zeros(K))
    out["laser_comb"] = (np.linspace(g[0], g[-1], K), np.zeros(K))
    return out


def signal(f, phi, u, chunk=4000):
    A = np.empty(len(u), complex)
    for s in range(0, len(u), chunk):
        A[s:s + chunk] = np.exp(1j * (np.outer(u[s:s + chunk], f) + phi[None, :])).sum(axis=1)
    return A / len(f)


def prime_powers(M):
    from primes_spectrum import sieve, lambda_terms
    n, lam = lambda_terms(sieve(M), M)
    return n.astype(int), lam


def analyse(u, A, g, lo=0.25, hi=4.2, tol=0.004):
    n, lam = prime_powers(int(np.exp(hi)) + 1)
    ln = np.log(n)
    env = np.abs(A)
    W = (u >= lo) & (u <= hi)
    near = np.zeros(len(u), bool)
    for v in ln:
        near |= np.abs(u - v) < tol
    bg = float(np.sqrt(np.mean(env[W & ~near] ** 2)))
    peaks = [i for i in np.nonzero(W)[0][1:-1] if env[i] > env[i - 1] and env[i] >= env[i + 1]]
    peaks = sorted(peaks, key=lambda i: -env[i])
    top = peaks[:20]
    at_pp = sum(1 for i in top if np.min(np.abs(ln - u[i])) < tol)
    found = [int(m) for m, v in zip(n, ln) if any(abs(u[i] - v) < tol and env[i] > 3 * bg for i in peaks)]
    strongest_other = max([env[i] for i in peaks if np.min(np.abs(ln - u[i])) >= tol], default=0.0)
    pred = (g[-1] / (2 * np.pi)) * lam / np.sqrt(n) / len(g)
    meas = [float(env[np.abs(u - v) < tol].max()) for v in ln]
    return dict(background=bg, top20_at_prime_powers=at_pp, found_above_3bg=found, n_prime_powers=int(len(n)),
                strongest_other=float(strongest_other), pp=n.tolist(), pred=pred.tolist(), meas=meas)


def sound(f, phi, fn, s=3.0, u0=0.25, u1=9.25, sr=22050):
    from scipy.io import wavfile
    t = np.arange(0.0, (u1 - u0) / s, 1.0 / sr)
    a = signal(f, phi, u0 + s * t).real
    a = a / np.sqrt(np.mean(a ** 2)) * 0.12                 # same loudness (rms) for every file
    ramp = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)
    a = np.clip(a * ramp, -0.99, 0.99)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    wavfile.write(fn, sr, (a * 32767).astype(np.int16))
    return fn


def main():
    g = zeros()
    sp = spectra(g)
    u = np.arange(0.0, 9.0, 2e-4)
    out = dict(K=len(g), u_step=2e-4, results={})
    sig = {}
    for name, (f, phi) in sp.items():
        A = signal(f, phi, u)
        sig[name] = A
        r = analyse(u, A, g)
        out["results"][name] = {k: v for k, v in r.items() if k not in ("pp", "pred", "meas")}
        print(f"{name:13s} background {r['background']:.4f}; of the 20 tallest peaks in u = 0.25…4.2, {r['top20_at_prime_powers']:2d} sit at"
              f" log(prime power); prime powers with a peak above 3× background: {len(r['found_above_3bg'])}/{r['n_prime_powers']};"
              f" tallest peak elsewhere {r['strongest_other']:.4f}")
        if name == "zeros":
            out["zeros_heights"] = dict(n=r["pp"], predicted=r["pred"], measured=r["meas"])
            rel = np.array(r["meas"]) / np.array(r["pred"])
            print(f"   zeros: pulse height / formula (Γ/2πK) Λ(n)/√n for n = {r['pp'][:10]} …: {np.round(rel[:10], 3).tolist()}")
            print(f"   found: {r['found_above_3bg']}")
    comb = sig["laser_comb"]
    d = (g[-1] - g[0]) / (len(g) - 1)
    out["comb_period"] = 2 * np.pi / d
    i = np.argmax(np.abs(comb) * (u > 1.0))
    print(f"laser comb: spacing {d:.4f}, expected pulse period 2π/Δ = {2 * np.pi / d:.4f}; first pulse found at u = {u[i]:.4f}"
          f" (height {np.abs(comb[i]):.3f})")
    files = [sound(*sp[k], os.path.join(HERE, "sound", f"zeta_{k}.wav")) for k in ("zeros", "random_phase", "gue")]
    print("wrote", [os.path.relpath(f, HERE) for f in files])
    keep = (u <= 9.0)
    out["u"] = np.round(u[keep][::5], 4).tolist()
    out["env"] = {k: np.round(np.abs(v[keep][::5]), 5).tolist() for k, v in sig.items()}
    json.dump(out, open(os.path.join(DATA, "zeta_sound.json"), "w"))
    figure(u, sig, out)


def figure(u, sig, out, fn="figures/zeta_sound_R6.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["path.simplify"] = False
    SURF, INK, INK2, GRID, C1, C2 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"
    fig = plt.figure(figsize=(14, 10.4), facecolor=SURF)
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.15, 1])

    def style(ax):
        ax.set_facecolor(SURF)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
    z = sig["zeros"]
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    ax.plot(u, z.real, color=C1, lw=0.4)
    for p in (2, 3, 5, 7, 11, 13):
        ax.annotate(str(p), (np.log(p), 0.21), ha="center", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.25, 0.25)
    ax.set_ylabel("Re A(u)", color=INK2)
    ax.set_xlabel("u = log x（「時間」）", color=INK2)
    ax.set_title("1000 個 ζ 零點同時發聲（等振幅、u = 0 時同相）：拉遠看像一段聲波；u = 0 的大脈衝已切掉，上限 ±0.25",
                 fontsize=10.5, color=INK, loc="left")
    ax = fig.add_subplot(gs[1, :])
    style(ax)
    sel = (u >= 0.25) & (u <= 4.2)
    ax.plot(u[sel], np.abs(z[sel]), color=C1, lw=0.8, label="零點合起來的包絡 |A(u)|")
    zh = out["zeros_heights"]
    ln = np.log(np.array(zh["n"]))
    ax.scatter(ln, zh["predicted"], color=C2, s=22, zorder=3, label="公式預測 (Γ/2πK) Λ(n)/√n（沒有參數）")
    for n, v, h in zip(zh["n"], ln, zh["measured"]):
        if h > 0.07 or n in (16, 27, 32):
            ax.annotate(str(n), (v, h + 0.006), ha="center", color=INK, fontsize=8)
    bg = out["results"]["zeros"]["background"]
    ax.axhline(bg, color=INK2, lw=0.8, ls="--")
    ax.annotate("背景（質數以外的地方）", (0.27, bg + 0.003), ha="left", color=INK2, fontsize=8.5)
    ax.set_xlim(0.25, 4.2)
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("|A(u)|", color=INK2)
    ax.set_xlabel("u = log x；數字是 x（質數與質數的次方）", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    r = out["results"]["zeros"]
    ax.set_title(f"放大：所有頻率只在 u = log 2、log 3、log 4、log 5、log 7…同時對齊，聚成一束（20 個最高的峰有 {r['top20_at_prime_powers']} 個在質數次方上）",
                 fontsize=10.5, color=INK, loc="left")
    NAME = {"random_phase": "同樣的頻率、相位打亂（分散的光譜）", "gue": "隨機矩陣能階（統計和零點一樣）",
            "poisson": "完全隨機的頻率", "laser_comb": "等間距頻率梳（鎖模雷射）"}
    for j, k in enumerate(("random_phase", "gue", "poisson", "laser_comb")):
        ax = fig.add_subplot(gs[2, j])
        style(ax)
        sel = (u >= 0.25) & (u <= 9.0)
        ax.plot(u[sel], np.abs(sig[k][sel]), color=C1, lw=0.5)
        for v in ln:
            ax.axvline(v, color=GRID, lw=0.6, zorder=0)
        ax.set_xlim(0.25, 9)
        ax.set_ylim(0, 0.2 if k != "laser_comb" else 1.05)
        rk = out["results"][k]
        extra = f"週期 2π/Δ = {out['comb_period']:.2f}" if k == "laser_comb" else f"20 高峰中 {rk['top20_at_prime_powers']} 個在質數次方"
        ax.set_title(f"{NAME[k]}\n{extra}", fontsize=9, color=INK)
        ax.set_xlabel("u", color=INK2)
    fig.suptitle("把 ζ 零點的頻譜融合成一個聲音：它只在質數的對數處聚成一束；打亂相位、換成統計相同的隨機頻譜，都不會",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(HERE, fn), dpi=110, facecolor=SURF)
    print("wrote", fn)


def cmd_cancel(lo=0.25, hi=4.2, tol=0.004):
    """Resonance and cancellation (the user's word 共振相消): where does each spectrum put its weight ∫|A|²?
    A random-matrix (GUE) spectrum spreads it as a smooth ramp; the zeros put it into the prime-power
    pulses and cancel in between (explicit formula)."""
    g = zeros()
    sp = spectra(g)
    u = np.arange(lo, hi, 2e-4)
    n, lam = prime_powers(int(np.exp(hi)) + 1)
    near = np.zeros(len(u), bool)
    for v in np.log(n):
        near |= np.abs(u - v) < tol
    out = {}
    for name in ("zeros", "gue", "poisson", "random_phase"):
        A = signal(*sp[name], u)
        w = np.abs(A) ** 2
        tot = float(np.trapezoid(w, u))
        at = float(np.trapezoid(w * near, u))
        out[name] = dict(total=tot, at_prime_powers=at, fraction_at_prime_powers=at / tot,
                         rms_between=float(np.sqrt(np.mean(w[~near]))))
        print(f"{name:13s} total ∫|A|² = {tot:.5f}; within ±{tol} of log(prime powers): {at / tot:6.1%};"
              f" rms between them {out[name]['rms_between']:.4f}  (those windows cover {near.mean():.1%} of u)")
    out["window_fraction"] = float(near.mean())
    json.dump(out, open(os.path.join(DATA, "zeta_cancel.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cancel":
        cmd_cancel()
    else:
        main()
