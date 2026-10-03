"""
The ringing of the Q = 2 ring without assuming a steady state (the user's objection, 2026-10-03:
nothing here is static, the energy keeps flowing out, the frequency need not be one fixed number).

For each run of lattice_dyn2.py ring (different kick strengths ε):
  * instantaneous frequency from successive full periods of the rms radius R(t) (zero crossings of
    R minus its running mean) and from the Hilbert phase; the amplitude envelope;
  * frequency against amplitude (an anharmonic oscillator shifts its frequency with amplitude, so a
    decaying vibration drifts in frequency);
  * the energy left in the box E(t) (the absorbing layer removes what is radiated) and the radiated
    power against amplitude;
  * the spectrum of the field at six probe points 3.6 from the centre: which frequencies leave.

Usage: python ringing_analysis.py tag [tag ...]     (tags of data/flagpair_dyn2_<tag>.json)
"""

import json
import os
import sys

import numpy as np
from scipy.signal import hilbert

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def running_mean(y, n):
    k = np.ones(n) / n
    pad = np.concatenate([np.full(n // 2, y[0]), y, np.full(n - 1 - n // 2, y[-1])])
    return np.convolve(pad, k, mode="valid")


def crossings(t, y):
    out = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            out.append(t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i]))
    return np.array(out)


def analyse(tag):
    d = json.load(open(os.path.join(DATA, f"flagpair_dyn2_{tag}.json")))
    tr = d["trace"]
    t = np.array([x["t"] for x in tr])
    R = np.array([x["R"][0] for x in tr])
    dt = t[1] - t[0]
    nper = int(round(9.2 / dt))                       # about one period of the bound mode
    y = R - running_mean(R, nper)
    zc = crossings(t, y)
    full = zc[2:] - zc[:-2]                            # full periods from every other crossing
    tf = 0.5 * (zc[2:] + zc[:-2])
    om_full = 2 * np.pi / full
    a = hilbert(y)
    amp = np.abs(a)
    ph = np.unwrap(np.angle(a))
    # amplitude at the times of the full-period estimates (mean envelope over that period)
    amp_f = np.array([amp[(t >= zc[i]) & (t <= zc[i + 2])].mean() for i in range(len(zc) - 2)])
    out = dict(tag=tag, eps=d.get("eps"), t_end=float(t[-1]), R_mean=float(R.mean()),
               t_period=tf.tolist(), omega_period=om_full.tolist(), amp_period=amp_f.tolist())
    print(f"{tag}: eps = {d.get('eps')}, t = 0..{t[-1]:g}, mean R = {R.mean():.4f}")
    for ti, oi, ai in zip(tf, om_full, amp_f):
        print(f"   t = {ti:6.2f}   omega(full period) = {oi:.4f}   amplitude of R = {ai:.4f}")
    if len(om_full) > 2:
        p = np.polyfit(tf, om_full, 1)
        out["drift"] = p[0]
        print(f"   drift d omega/dt = {p[0]:+.2e}")
    if "E" in tr[0]:
        E = np.array([x["E"] for x in tr])
        Es = running_mean(E, nper)
        P = -np.gradient(Es, t)                        # radiated power (smoothed over a period)
        out.update(E=E.tolist(), P_rad=P.tolist(), t=t.tolist())
        print(f"   energy in the box {E[0]:.4f} -> {E[-1]:.4f} (radiated {E[0] - E[-1]:.4f})")
        q = np.array([x["probe_q"] for x in tr]).mean(axis=1)
        late = t > 10
        qq = q[late] - q[late].mean()
        F = np.abs(np.fft.rfft(qq * np.hanning(len(qq))))
        f = 2 * np.pi * np.fft.rfftfreq(len(qq), dt)
        top = np.argsort(F[1:])[::-1][:5] + 1
        out["probe_peaks"] = [(float(f[i]), float(F[i])) for i in sorted(top)]
        print("   probe spectrum (t > 10) peaks at omega =", [round(f[i], 3) for i in sorted(top)],
              f"(resolution {2 * np.pi / (t[late][-1] - t[late][0]):.3f})")
    return out


if __name__ == "__main__":
    res = [analyse(tag) for tag in sys.argv[1:]]
    json.dump(res, open(os.path.join(DATA, "ringing_analysis.json"), "w"), indent=1)
