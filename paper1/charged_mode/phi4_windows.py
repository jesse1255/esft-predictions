"""
Where fractals live in soliton physics: resonance windows in kink–antikink collisions.

The user's suggestion (2026-09-27): try fractals instead of circles and waves.  Our vortex rings
are smooth (fractal_dims.py: no fractional dimension over any range of scales).  A fractal does
appear in the *outcome of collisions* when the solitons carry an internal vibration (a bound mode,
§4.4): part of the collision energy is parked in the vibration, and whether the pair escapes or is
captured depends on whether the vibration is in phase at the next contact — a resonance condition,
repeated at every bounce, which nests windows inside windows.

Model (1 + 1 dimensions, the standard example; not our model):
    L = ½ φ_t² − ½ φ_x² − ½ (1 − φ²)²,   kink φ = tanh x (mass 4/3), meson mass 2,
    one internal (shape) mode ω = √3 per kink.
Kink at −x0 moving right and antikink at +x0 moving left with speed v (Lorentz contracted).
Outcome from the field at the centre φ(0, t): every collision sends it below zero; at the end it
sits near +1 if the pair has escaped and oscillates about −1 if it was captured (a bion).
Label = number of collisions before escape (1, 2, 3, …) or 0 for capture.

Fractal test: the uncertainty exponent.  f(ε) = fraction of speeds v for which the label at v and at
v + ε differ.  Isolated boundaries give f ∝ ε (exponent 1); a fractal set of boundaries of box
dimension D gives f ∝ ε^(1 − D).

Usage: python phi4_windows.py scan v_min v_max n tag      → labels on a uniform grid of speeds
       python phi4_windows.py analyse tag [tag ...]         → windows, uncertainty exponent
"""

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def simulate(vs, L=80.0, dx=0.05, dt=0.025, T=160.0, x0=10.0, sponge=15.0, smax=1.0, rec=4):
    """Velocity-Verlet on a uniform grid for a batch of speeds; absorbing layer |x| > L − sponge.
    Returns φ(0, t) sampled every `rec` steps, shape (len(vs), steps // rec)."""
    x = jnp.arange(-L, L + 0.5 * dx, dx)
    v = jnp.asarray(vs, dtype=jnp.float64)[:, None]
    g = 1.0 / jnp.sqrt(1.0 - v ** 2)
    phi = jnp.tanh(g * (x + x0)) - jnp.tanh(g * (x - x0)) - 1.0
    phit = -v * g * (1.0 / jnp.cosh(g * (x + x0)) ** 2 + 1.0 / jnp.cosh(g * (x - x0)) ** 2)
    sig = smax * jnp.clip((jnp.abs(x) - (L - sponge)) / sponge, 0.0, None) ** 2
    i0 = int(round(L / dx))

    def force(phi):
        lap = (jnp.roll(phi, -1, -1) - 2.0 * phi + jnp.roll(phi, 1, -1)) / dx ** 2
        lap = lap.at[:, 0].set(0.0).at[:, -1].set(0.0)
        return lap + 2.0 * phi * (1.0 - phi ** 2)

    damp = jnp.exp(-sig * dt)

    def inner(carry, _):
        phi, phit, f = carry
        phit = phit + 0.5 * dt * f
        phi = phi + dt * phit
        f = force(phi)
        phit = (phit + 0.5 * dt * f) * damp
        return (phi, phit, f), None

    def outer(carry, _):
        carry, _ = jax.lax.scan(inner, carry, None, length=rec)
        return carry, carry[0][:, i0]

    n_out = int(round(T / dt)) // rec
    run = jax.jit(lambda c: jax.lax.scan(outer, c, None, length=n_out))
    _, centre = run((phi, phit, force(phi)))
    return np.asarray(centre).T, dt * rec


def classify(centre, dt_rec, tail=20.0):
    """Label: number of collisions (negative excursions of φ(0)) before escape; 0 = capture."""
    labels = np.zeros(centre.shape[0], dtype=int)
    ntail = int(round(tail / dt_rec))
    for i, c in enumerate(centre):
        neg = c < 0.0
        entries = int(np.sum(neg[1:] & ~neg[:-1]))
        if c[-ntail:].mean() > 0.3 and not neg[-ntail:].any():
            labels[i] = max(entries, 1)
        else:
            labels[i] = 0
    return labels


def cmd_scan(vmin, vmax, n, tag, batch=500):
    vs = np.linspace(vmin, vmax, n)
    labels = np.zeros(n, dtype=int)
    t0 = time.time()
    for s in range(0, n, batch):
        c, dtr = simulate(vs[s:s + batch])
        labels[s:s + batch] = classify(c, dtr)
        print(f"{s + len(vs[s:s + batch])}/{n} done, {time.time() - t0:.0f} s; labels so far "
              f"{np.bincount(labels[:s + batch], minlength=6).tolist()}", flush=True)
    json.dump(dict(tag=tag, vmin=vmin, vmax=vmax, n=n, v=vs.tolist(), label=labels.tolist()),
              open(os.path.join(DATA, f"phi4_windows_{tag}.json"), "w"))


def windows(v, lab):
    """Maximal runs of equal label: (label, v_start, v_end, count)."""
    out = []
    s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            out.append((int(lab[s]), float(v[s]), float(v[i - 1]), i - s))
            s = i
    return out


def uncertainty(lab, dv, ks):
    f = []
    for k in ks:
        f.append(float(np.mean(lab[k:] != lab[:-k])))
    return np.array(f)


def cmd_analyse(tags):
    res = {}
    for tag in tags:
        d = json.load(open(os.path.join(DATA, f"phi4_windows_{tag}.json")))
        v, lab = np.array(d["v"]), np.array(d["label"])
        dv = v[1] - v[0]
        w = windows(v, lab)
        esc = [x for x in w if x[0] > 0]
        print(f"{tag}: {len(v)} speeds in [{v[0]:.6f}, {v[-1]:.6f}], dv = {dv:.2e}; label counts "
              f"{np.bincount(lab, minlength=6).tolist()} (0 = capture, n = escape after n collisions)")
        # critical speed: lowest v above which every label is 1
        ones = np.where(lab != 1)[0]
        vc = v[ones[-1] + 1] if len(ones) and ones[-1] + 1 < len(v) else None
        print(f"   critical speed (all 1-bounce above): {vc}")
        for x in esc:
            if x[0] >= 2 and x[3] >= 3:
                print(f"   {x[0]}-bounce window  v = {x[1]:.6f} … {x[2]:.6f}  ({x[3]} points)")
        ks = np.unique(np.round(np.logspace(0, np.log10(len(v) // 8), 14)).astype(int))
        f = uncertainty(lab, dv, ks)
        ok = f > 0
        slope = np.polyfit(np.log(ks[ok] * dv), np.log(f[ok]), 1)[0] if ok.sum() > 2 else float("nan")
        print(f"   uncertainty exponent alpha = {slope:.3f}  → boundary dimension D = 1 − alpha = {1 - slope:.3f}")
        for k, fv in zip(ks, f):
            print(f"      eps = {k * dv:.2e}   f = {fv:.4e}")
        res[tag] = dict(n=len(v), dv=dv, counts=np.bincount(lab, minlength=6).tolist(), v_c=vc,
                        windows=[x for x in esc if x[0] >= 2], eps=(ks * dv).tolist(), f=f.tolist(),
                        alpha=slope, D=1 - slope)
    json.dump(res, open(os.path.join(DATA, "phi4_windows_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        cmd_scan(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
    elif sys.argv[1] == "analyse":
        cmd_analyse(sys.argv[2:])
