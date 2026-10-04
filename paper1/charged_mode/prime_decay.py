"""
What the instability of the axial rings means physically (the user's point, 2026-10-04: unstable is not the
same as wrong; ask which physical state an unstable configuration is, and compute it).

An unstable stationary configuration can be (a) a transition state (the top of a pass between basins),
(b) a short-lived resonance, (c) the high-symmetry state of a symmetry-breaking transition, or (d) a state
held by something else (rotation, charge, a constraint).  Which one it is shows in three numbers:
how fast the instability grows, where it ends, and how much energy it releases.

1. rate    γ = √(−λ) from the sector Hessians of prime_nonaxial.py (L² metric, without the Skyrme part
           of the kinetic metric, so an upper bound on the true rate); e-folding time 1/γ against the
           period 2π/μ of the mass gap.
2. decay   the axial ring put on the 3D lattice of the full three-line flag model (κ₃ = 2, N = 41,
           h = 0.3; embedded in link 01 with smith.cp1_at / embed_pts), its core displaced by
           ε cos(kφ) along z in its most unstable sector k, then relaxed (lattice3d.relax, L-BFGS).
           The flow may also leak into the third line: both decay channels of the model are open.
           Diagnostics: energy released, charge, leakage into line 2, and the shape of the core curve
           (|U₀₀|² small): strands per azimuthal bin and the Fourier content of its radius and height.

Usage: python prime_decay.py [mxn:k[:eps[:pc]] ...]      (default 5x1:3 7x1:3 4x1:2 3x2:2, eps = 0.2;
       eps = 0 relaxes the undeformed axial ring: the baseline for what the cubic lattice alone does;
       pc puts the ring's centre on a plaquette centre (h/2, h/2, 0) instead of a lattice site: a control
       for the lattice pinning (Peierls–Nabarro) energy, which the free relaxations can exploit by sliding)
       python prime_decay.py writhe      core curve of every relaxed state: Fourier phases of r(φ), z(φ) and the
                                         writhe Wr (Gauss integral) → data/prime_decay_writhe.json
       python prime_decay.py pinning     unrelaxed lattice energy of the FE 3 × 2 ring shifted and tilted
                                         rigidly: the size of the lattice's own energy landscape
                                         → data/prime_decay_pinning.json
       python prime_decay.py summary     lowest energy found per charge, before (axial, FE) and after the 3D
                                         relaxations, and the prime penalty against the neighbouring charges
                                         → data/prime_decay_summary.json

3. twist → writhe.  For the ribbon made of the core curve and a nearby preimage, Lk = Tw + Wr (Călugăreanu–
           White–Fuller), with Lk = Q fixed by the topology.  The axial ring has a planar core (Wr = 0): all of
           its linking is twist, Q turns per circuit for Q × 1 and m/n per strand for m × n.  A mirror-symmetric
           buckle (r and z modulated in phase) keeps Wr = 0; a helical wave (z a quarter period from r) does not.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")
REF = os.path.join(DATA, "prime_charges_ne32x48_a5.json")


def lattice_ring(key, k, eps, lat, shift=(0.0, 0.0, 0.0)):
    from run_hopf_pair import freeze_A, mirror_full
    from hopfion_axisym import Grid
    from smith import cp1_at, embed_pts
    d = json.load(open(REF))
    r = d["per_factorization"][key]
    Gh = freeze_A(Grid(d["ne_r"], d["ne_z"], p=2, a=d["a"], half=True))
    Uh = np.load(os.path.join(SCRATCH, f"prime_mn_state_m{r['m']}_n{r['n']}_{r['seed']}_ne{d['ne_r']}x{d['ne_z']}_a{d['a']:g}.npz"))["U"]
    Gf, Uf = mirror_full(Gh, Uh)
    X, Y, Z = lat.X - shift[0], lat.Y - shift[1], lat.Z - shift[2]
    rho, phi = np.hypot(X, Y), np.arctan2(Y, X)
    w = rho ** 2 / (rho ** 2 + 1.0)                      # vanishes on the axis (φ undefined there)
    z = Z - eps * np.cos(k * phi) * w
    U0 = embed_pts(cp1_at(Gf, Uf[:, :3], X, Y, Z, m=r["m"]), (0, 1))
    Us = embed_pts(cp1_at(Gf, Uf[:, :3], X, Y, z, m=r["m"]), (0, 1))
    return r, U0, Us


def core_shape(lat, U, level=0.1, nbin=72):
    """The core curve (|U₀₀|² < level): strands per azimuthal bin, Fourier amplitudes of r(φ), z(φ)."""
    c = np.abs(U[..., 0, 0]) ** 2 < level
    x, y, z = lat.X[c], lat.Y[c], lat.Z[c]
    if x.size == 0:
        return dict(points=0)
    r, ph = np.hypot(x, y), np.arctan2(y, x)
    b = ((ph + np.pi) / (2 * np.pi) * nbin).astype(int) % nbin
    strands, rmean, zmean = [], np.full(nbin, np.nan), np.full(nbin, np.nan)
    for i in range(nbin):
        s = b == i
        if not s.any():
            strands.append(0)
            continue
        rr, zz = r[s], z[s]
        rmean[i], zmean[i] = rr.mean(), zz.mean()
        # clusters in the (r, z) plane of this bin: single linkage with a gap of 2 h
        pts = np.stack([rr, zz], 1)
        lab = -np.ones(len(pts), int)
        nc = 0
        for j in range(len(pts)):
            if lab[j] >= 0:
                continue
            stack, lab[j] = [j], nc
            while stack:
                q = stack.pop()
                near = np.nonzero((lab < 0) & (np.hypot(*(pts - pts[q]).T) < 2.0 * lat.h))[0]
                lab[near] = nc
                stack.extend(near.tolist())
            nc += 1
        strands.append(nc)
    ok = np.isfinite(rmean)
    ang = (np.arange(nbin) + 0.5) / nbin * 2 * np.pi - np.pi
    four = {}
    for name, f in (("r", rmean), ("z", zmean)):
        if ok.sum() > 8:
            ff = f[ok] - f[ok].mean()
            four[name] = [float(abs(np.mean(ff * np.exp(-1j * m * ang[ok])))) * 2 for m in range(1, 7)]
    st = np.array(strands)
    return dict(points=int(x.size), strands_median=float(np.median(st[st > 0])) if (st > 0).any() else 0.0,
                strands_max=int(st.max()), bins_empty=int((st == 0).sum()), r_mean=float(np.nanmean(rmean)),
                z_spread=float(np.nanmax(zmean) - np.nanmin(zmean)), fourier=four)


def run(args):
    key, k, eps, pc = args
    from lattice3d import Lattice, relax
    from nlines import charge_n
    lat = Lattice(41, 0.3)
    shift = (0.5 * lat.h, 0.5 * lat.h, 0.0) if pc else (0.0, 0.0, 0.0)
    r, U0, Us = lattice_ring(key, k, eps, lat, shift)
    E_axial, E_seed = lat.energy(U0), lat.energy(Us)
    t0 = time.time()
    tag = f"{key}_k{k}" + ("" if eps == 0.2 else f"_eps{eps:g}") + ("_pc" if pc else "")
    log = os.path.join(SCRATCH, f"prime_decay_{tag}.log")
    if os.path.exists(log):
        os.remove(log)
    U, info = relax(lat, Us, cons=None, chunk=150, max_chunks=16, gtol=1e-3, verbose=False, log=log)
    U = np.asarray(U)
    E = lat.energy(U)
    Q, Qs = charge_n(lat.h, U, pad=1)
    leak = 1.0 - np.abs(U[..., 2, 2]) ** 2
    hist = [json.loads(l) for l in open(log)]
    np.savez_compressed(os.path.join(SCRATCH, f"prime_decay_{tag}.npz"), U=U)
    out = dict(key=key, m=r["m"], n=r["n"], Q=r["m"] * r["n"], k=k, eps=eps, shift=list(shift), E_axial_FE=r["E"], E_axial_lattice=E_axial,
               E_seed=E_seed, E_final=E, released=E_axial - E, charge=Q, charge_lines=Qs,
               leak_max=float(leak.max()), leak_int=float(leak.sum() * lat.h ** 3),
               shape_axial=core_shape(lat, U0), shape_final=core_shape(lat, U),
               history=[(h_["it"], h_["E"], h_["gmax"]) for h_ in hist], seconds=time.time() - t0)
    print(f"{key} (k = {k}): axial {E_axial:.2f} → final {E:.2f} (released {E_axial - E:+.2f}); charge {Q:+.3f};"
          f" leak into line 2: max {out['leak_max']:.3f}; core strands {out['shape_final'].get('strands_median')},"
          f" r ≈ {out['shape_final'].get('r_mean', 0):.2f}, z spread {out['shape_final'].get('z_spread', 0):.2f}"
          f"  ({time.time() - t0:.0f} s, {hist[-1]['it'] if hist else 0} iterations, gmax {hist[-1]['gmax'] if hist else 0:.1e})", flush=True)
    json.dump(out, open(os.path.join(DATA, f"prime_decay_{tag}.json"), "w"), indent=1)
    return out


def main(specs, procs=4, eps=0.2):
    procs = int(os.environ.get("PRIME_DECAY_PROCS", procs))
    from multiprocessing import Pool
    jobs = []
    for s in specs:
        parts = s.split(":")
        jobs.append((parts[0], int(parts[1]), float(parts[2]) if len(parts) > 2 else eps, len(parts) > 3 and parts[3] == "pc"))
    with Pool(procs) as pool:
        pool.map(run, jobs, chunksize=1)


def core_curve(U, h=0.3, level=0.1, nbin=48, mmax=4):
    """The core curve as r(φ), z(φ) about the core's own centroid (weights level − |U₀₀|²), least-squares
    Fourier series to order mmax through the filled azimuthal bins.  Returns centroid, coefficients
    [c₀, cos 1, sin 1, …] of r and of z, filled bins."""
    N = U.shape[0]
    x1 = (np.arange(N) - (N - 1) / 2) * h
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing="ij")
    a = np.abs(U[..., 0, 0]) ** 2
    c = a < level
    w = (level - a)[c]
    P = np.stack([X[c], Y[c], Z[c]], 1)
    cen = (P * w[:, None]).sum(0) / w.sum()
    q = P - cen
    r, ph, z = np.hypot(q[:, 0], q[:, 1]), np.arctan2(q[:, 1], q[:, 0]), q[:, 2]
    b = ((ph + np.pi) / (2 * np.pi) * nbin).astype(int) % nbin
    ang = (np.arange(nbin) + 0.5) / nbin * 2 * np.pi - np.pi
    rb, zb, ok = np.zeros(nbin), np.zeros(nbin), np.zeros(nbin, bool)
    for i in range(nbin):
        s = b == i
        if s.any():
            rb[i], zb[i], ok[i] = np.average(r[s], weights=w[s]), np.average(z[s], weights=w[s]), True
    A = np.column_stack([np.ones(ok.sum())] + [f(m * ang[ok]) for m in range(1, mmax + 1) for f in (np.cos, np.sin)])
    cr = np.linalg.lstsq(A, rb[ok], rcond=None)[0]
    cz = np.linalg.lstsq(A, zb[ok], rcond=None)[0]
    return cen, cr, cz, int(ok.sum())


def _fourier(c, t):
    out = c[0] + 0.0 * t
    for m in range(1, (len(c) - 1) // 2 + 1):
        out = out + c[2 * m - 1] * np.cos(m * t) + c[2 * m] * np.sin(m * t)
    return out


def writhe(cr, cz, n=600):
    """Writhe of the closed curve (r(t) cos t, r(t) sin t, z(t)): Gauss double integral, midpoint rule."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r, z = _fourier(cr, t), _fourier(cz, t)
    p = np.stack([r * np.cos(t), r * np.sin(t), z], 1)
    d = np.roll(p, -1, 0) - p
    mid = p + 0.5 * d
    W = 0.0
    for i in range(n):
        rij = mid[i] - mid
        dist = np.linalg.norm(rij, axis=1)
        dist[i] = np.inf
        W += np.sum(np.einsum("ij,ij->i", rij, np.cross(d[i], d)) / dist ** 3)
    return W / (4 * np.pi)


def writhe_all():
    import glob
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    checks = dict(circle=writhe(np.r_[1.7, np.zeros(8)], np.zeros(9)),
                  helical_k4=writhe(np.r_[1.7, np.zeros(6), 0.4, 0.0], np.r_[np.zeros(8), 0.4]),
                  in_phase_k4=writhe(np.r_[1.7, np.zeros(6), 0.4, 0.0], np.r_[np.zeros(7), 0.4, 0.0]))
    print("checks: planar circle Wr = %+.4f, helical wave r + iz ∝ e^{4iφ} Wr = %+.4f, in-phase buckle Wr = %+.4f"
          % (checks["circle"], checks["helical_k4"], checks["in_phase_k4"]))
    out = dict(checks=checks, states={})
    for f in sorted(glob.glob(os.path.join(SCRATCH, "prime_decay_*.npz"))):
        tag = os.path.basename(f)[len("prime_decay_"):-4]
        U = np.load(f)["U"]
        res = {}
        for level in (0.1, 0.2):
            cen, cr, cz, nok = core_curve(U, level=level)
            harm = {}
            for m in range(1, 5):
                ar, az = np.hypot(cr[2 * m - 1], cr[2 * m]), np.hypot(cz[2 * m - 1], cz[2 * m])
                ph = np.degrees(np.arctan2(cz[2 * m], cz[2 * m - 1]) - np.arctan2(cr[2 * m], cr[2 * m - 1]))
                harm[m] = dict(r=float(ar), z=float(az), phase_z_minus_r=float((ph + 180) % 360 - 180))
            res[str(level)] = dict(centroid=cen.tolist(), R=float(cr[0]), Wr=float(writhe(cr, cz)), bins=nok,
                                   harmonics=harm, cr=cr.tolist(), cz=cz.tolist())
        out["states"][tag] = res
        big = [f"m={m}: r {v['r']:.2f}, z {v['z']:.2f}, phase {v['phase_z_minus_r']:+.0f}°"
               for m, v in res["0.1"]["harmonics"].items() if max(v["r"], v["z"]) > 0.05]
        print(f"{tag:18s} Wr = {res['0.1']['Wr']:+.3f} (level 0.1), {res['0.2']['Wr']:+.3f} (0.2);  R {res['0.1']['R']:.2f};  "
              + "; ".join(big), flush=True)
    json.dump(out, open(os.path.join(DATA, "prime_decay_writhe.json"), "w"), indent=1)


def pinning(key="3x2"):
    """Unrelaxed lattice energy of the FE ring shifted / tilted rigidly (no relaxation): how much the
    lattice alone prefers one position or orientation over another."""
    from lattice3d import Lattice
    from run_hopf_pair import freeze_A, mirror_full
    from hopfion_axisym import Grid
    from smith import cp1_at, embed_pts
    lat = Lattice(41, 0.3)
    d = json.load(open(REF))
    r = d["per_factorization"][key]
    Gh = freeze_A(Grid(d["ne_r"], d["ne_z"], p=2, a=d["a"], half=True))
    Uh = np.load(os.path.join(SCRATCH, f"prime_mn_state_m{r['m']}_n{r['n']}_{r['seed']}_ne{d['ne_r']}x{d['ne_z']}_a{d['a']:g}.npz"))["U"]
    Gf, Uf = mirror_full(Gh, Uh)

    def E_at(shift=(0.0, 0.0, 0.0), tilt=0.0, axis=(1.0, 0.0, 0.0)):
        th = np.radians(tilt)
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        P = np.stack([lat.X - shift[0], lat.Y - shift[1], lat.Z - shift[2]], -1)
        # inverse rotation of the points (Rodrigues, angle −θ about a)
        P = P * np.cos(th) - np.cross(a, P) * np.sin(th) + (P @ a)[..., None] * a * (1 - np.cos(th))
        U = embed_pts(cp1_at(Gf, Uf[:, :3], P[..., 0], P[..., 1], P[..., 2], m=r["m"]), (0, 1))
        return float(lat.energy(U))
    E0 = E_at()
    out = dict(key=key, E0=E0, shifts={}, tilts={})
    print(f"{key} ring at a lattice site, axis along z: E = {E0:.3f} (unrelaxed)")
    for sh in ((0.05, 0, 0), (0.1, 0, 0), (0.15, 0, 0), (0.15, 0.15, 0), (0, 0, 0.15), (0.15, 0.15, 0.15)):
        out["shifts"][str(sh)] = E_at(shift=sh) - E0
        print(f"   shift {sh}: ΔE = {out['shifts'][str(sh)]:+.3f}", flush=True)
    for th in (3.0, 6.5, 10.0, 20.0, 45.0):
        out["tilts"][str(th)] = dict(about_x=E_at(tilt=th) - E0, about_diag=E_at(tilt=th, axis=(1, 1, 0)) - E0)
        print(f"   tilt {th:4.1f}°: ΔE = {out['tilts'][str(th)]['about_x']:+.3f} (about x), "
              f"{out['tilts'][str(th)]['about_diag']:+.3f} (about (1, 1, 0))", flush=True)
    json.dump(out, open(os.path.join(DATA, "prime_decay_pinning.json"), "w"), indent=1)


def summary():
    """Prime penalty before and after the 3D relaxations.  Before: the cheapest axial ring per charge (FE grid).
    After: the lowest lattice energy reached from any of the relaxations of that charge (an upper bound on the
    true minimum).  Penalty of Q: e(Q) / [(e(Q−1) + e(Q+1))/2] − 1 with e = E/Q^¾."""
    import glob
    fe = json.load(open(REF))["per_factorization"]
    axial, lattice = {}, {}
    for k, r in fe.items():
        Q = r["m"] * r["n"]
        if Q <= 8 and (Q not in axial or r["E"] < axial[Q][1]):
            axial[Q] = (k, r["E"])
    for f in sorted(glob.glob(os.path.join(DATA, "prime_decay_*_k*.json"))):
        r = json.load(open(f))
        tag = os.path.basename(f)[len("prime_decay_"):-5]
        if r["Q"] not in lattice or r["E_final"] < lattice[r["Q"]][1]:
            lattice[r["Q"]] = (tag, r["E_final"])
    out = dict(axial={}, lattice={}, penalty_axial={}, penalty_lattice={})
    print(f"{'Q':>3} {'axial (FE)':>22} {'e':>7}   {'after 3D (lattice)':>26} {'e':>7}")
    for Q in sorted(set(axial) | set(lattice)):
        a, l = axial.get(Q), lattice.get(Q)
        out["axial"][Q] = dict(key=a[0], E=a[1], e=a[1] / Q ** 0.75) if a else None
        out["lattice"][Q] = dict(tag=l[0], E=l[1], e=l[1] / Q ** 0.75) if l else None
        print(f"{Q:>3} {(a[0] + f' {a[1]:.1f}') if a else '—':>22} {(a[1] / Q ** 0.75) if a else float('nan'):7.1f}   "
              f"{(l[0] + f' {l[1]:.1f}') if l else '—':>26} {(l[1] / Q ** 0.75) if l else float('nan'):7.1f}")
    for Q in (5, 7):
        for name, tab in (("penalty_axial", out["axial"]), ("penalty_lattice", out["lattice"])):
            if all(tab.get(q) for q in (Q - 1, Q, Q + 1)):
                out[name][Q] = tab[Q]["e"] / (0.5 * (tab[Q - 1]["e"] + tab[Q + 1]["e"])) - 1
        print(f"prime {Q}: penalty against Q ± 1, axial {100 * out['penalty_axial'].get(Q, float('nan')):+.1f}%, "
              f"after 3D {100 * out['penalty_lattice'].get(Q, float('nan')):+.1f}%")
    json.dump(out, open(os.path.join(DATA, "prime_decay_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    if sys.argv[1:2] == ["summary"]:
        summary()
    elif sys.argv[1:2] == ["writhe"]:
        writhe_all()
    elif sys.argv[1:2] == ["pinning"]:
        pinning(*sys.argv[2:3])
    else:
        specs = [a for a in sys.argv[1:] if ":" in a] or ["5x1:3", "7x1:3", "4x1:2", "3x2:2"]
        main(specs)
