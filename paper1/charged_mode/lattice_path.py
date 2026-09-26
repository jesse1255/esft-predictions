"""
The 01 + 12 → 02 fusion path without axial symmetry (lattice3d), seeded by the k = 1
"zipper" mode of the coaxial saddle (sector_mode.py: the two rings tilt in opposite
senses, lines 0 and 2 meet on one side while line 1 is pushed out of it).

Reaction coordinate: the 3D distance s = |X̄₀ − X̄₂| between the centroids of the
(1 − |U₀₀|²)² and (1 − |U₂₂|²)² distributions (line 0 belongs to the 01 vortex, line 2
to the 12 vortex), held fixed with an augmented Lagrangian together with the pair
centre.  At each s every other degree of freedom relaxes, including tilts, slides
and bending, so E(s) is the lowest energy at that separation that the relaxation
finds from its seed; max_s E(s) − E_A − E_B bounds the barrier of this path from above.

    python lattice_path.py refs  N h              # A (01), B (12), fused (02) on the lattice
    python lattice_path.py seed  N h s eps        # saddle + eps·(zipper mode), relaxed at fixed s
    python lattice_path.py axial N h s            # saddle without seed at fixed s (stays coaxial?)
    python lattice_path.py scan  N h s_from s_to ds   # continue from the nearest saved state
    python lattice_path.py coax N h d [d ...]      # coaxial states at fixed centroid distance (from the
                                                  # axial cons states d = 3, 2, 1.5, 1.25, 1)
    python lattice_path.py oscan N h O_from O_to dO tag seed [theta_deg]
                                                  # overlap O = ∫ q₀ q₂ as the coordinate (0 apart, grows on
                                                  # contact and zipping); seed 'coax<d>' (+ hinge tilt) or .npz
    python lattice_path.py d1scan N h D_from D_to dD tag seed.npz
                                                  # same with the line-1 weight D₁ = ∫(1 − |U₁₁|²) as the
                                                  # coordinate (D₁ ≈ 28.4 separated, 0 fused), consecutive
                                                  # from seed.npz; rows kind = "d1_<tag>"

States: $FLAGPAIR_SCRATCH/lattice_<kind>_N<N>_h<h>_s<s>.npz;  rows: data/flagpair_lattice_path_N<N>_h<h>.json
"""

import glob
import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from lattice3d import Lattice, from_axial, relax, Constraint, SCRATCH, DATA

AX_SADDLE = os.path.join(DATA, "flagpair_state_cons_ne24x32_a4_k32_d1.npy")
AX_FUSED = os.path.join(DATA, "flagpair_state_A21in02_ne24x32_a4_k32.npy")
MODE = os.path.join(SCRATCH, "sector_mode_flagpair_state_cons_ne24x32_a4_k32_d1_k1_negative_1.npy")


def tagof(N, h):
    return f"N{N}_h{h:g}"


def out_json(N, h):
    return os.path.join(DATA, f"flagpair_lattice_path_{tagof(N, h)}.json")


def load_rows(N, h):
    fn = out_json(N, h)
    return json.load(open(fn)) if os.path.exists(fn) else dict(N=N, h=h, rows=[])


def add_row(N, h, row):
    d = load_rows(N, h)
    d["rows"].append(row)
    with open(out_json(N, h), "w") as fh:
        json.dump(d, fh, indent=1)


def axial_grid():
    from run_flag_pair import setup
    Gf, Uf = setup(24, 32, 4)
    return Gf, Uf


def zipper_mode(Gf):
    """Nodal (Xc, Xs) of the saved k = 1 negative mode (M-normalised sector vector)."""
    from run_flag_pair import fmodel
    from flag_sector import FlagSector, _x6_to_X
    from sector_mode import _fields_of
    fm = fmodel(Gf, 2.0)
    S = FlagSector(fm, np.load(AX_SADDLE), 1)
    v = np.load(MODE)
    xc, xs = _fields_of(S, jnp.asarray(v))
    return np.asarray(_x6_to_X(xc)), np.asarray(_x6_to_X(xs))


def describe(lat, U):
    g = lat.geometry(U)
    Un = np.asarray(U)
    q0, q1, q2 = (1.0 - np.abs(Un[..., c, c]) ** 2 for c in range(3))
    out = dict(parts=lat.parts(U), lines=[g[c]["weight"] for c in range(3)],
               overlap02=float(lat.h ** 3 * np.sum(q0 * q2)),
               triple_volume=float(lat.h ** 3 * np.sum((q0 > 0.1) & (q1 > 0.1) & (q2 > 0.1))),
               core_depth=[float(np.max(1.0 - np.abs(np.asarray(U)[..., c, c]) ** 2)) for c in range(3)])
    if "pair" in g:
        out["pair"] = g["pair"]
        out["normals"] = [g[0]["normal"], g[2]["normal"]]
        out["centroids"] = [g[c].get("centroid") for c in range(3)]
        out["radii"] = [g[c].get("radius") for c in range(3)]
    return out


def save_state(kind, N, h, s, U):
    os.makedirs(SCRATCH, exist_ok=True)
    fn = os.path.join(SCRATCH, f"lattice_{kind}_{tagof(N, h)}_s{s:.3f}.npz")
    np.savez_compressed(fn, U=U)
    return fn


def cmd_refs(N, h):
    from run_flag_pair import embed
    Gf, Uf = axial_grid()
    lat = Lattice(N, h)
    res = {}
    for name, Uax in (("A01", embed(Gf, Uf[:, :3], (0, 1))), ("B12", embed(Gf, Uf[:, :3], (1, 2))),
                      ("fused02", np.load(AX_FUSED))):
        U0 = from_axial(lat, Gf, Uax)
        E0 = lat.energy(U0)
        U, info = relax(lat, U0, chunk=100, max_chunks=20, gtol=1e-4, verbose=False)
        res[name] = dict(E_interp=E0, E=info["E"], gmax=info["gmax"], iterations=info["iterations"], **describe(lat, U))
        np.savez_compressed(os.path.join(SCRATCH, f"lattice_ref_{name}_{tagof(N, h)}.npz"), U=U)
        print(f"{name}: interpolated {E0:.5f} → relaxed {info['E']:.5f} ({info['iterations']} its, gmax {info['gmax']:.1e})",
              flush=True)
    res["binding_fused"] = res["fused02"]["E"] - res["A01"]["E"] - res["B12"]["E"]
    print(f"fused − A − B = {res['binding_fused']:+.4f}", flush=True)
    add_row(N, h, dict(kind="refs", **res))


def refs_energy(N, h):
    for r in load_rows(N, h)["rows"]:
        if r["kind"] == "refs":
            return r["A01"]["E"] + r["B12"]["E"]
    raise RuntimeError("run refs first")


def run_point(kind, N, h, s, U0, extra=None, coord="dist", K=200.0, outer=10, max_chunks=15, gtol=1e-3):
    lat = Lattice(N, h)
    E_ref = refs_energy(N, h)
    cons = Constraint(lat, coord, s)
    t0 = time.time()
    log = os.path.join(SCRATCH, f"lattice_{kind}_{tagof(N, h)}_s{s:.3f}.log")
    U, info = relax(lat, U0, cons=cons, K=K, outer=outer, chunk=100, max_chunks=max_chunks, gtol=gtol, verbose=False,
                    log=log, ctol=1e-3)
    row = dict(kind=kind, coord=coord, s=s, E=info["E"], E_int=info["E"] - E_ref, gmax=info["gmax"], iterations=info["iterations"],
               multipliers=info["lam"], c=info["hist"][-1].get("c"), t=time.time() - t0, **describe(lat, U))
    if extra:
        row.update(extra)
    row["state"] = save_state(kind, N, h, s, U)
    add_row(N, h, row)
    p = row.get("pair", {})
    print(f"{kind} s = {s:.3f}: E = {row['E']:.4f}  E_int = {row['E_int']:+.4f}  gmax {row['gmax']:.1e}  its {row['iterations']}"
          f"  | dist {p.get('dist', 0):.3f} rel.tilt {p.get('relative_tilt_deg', 0):.2f}° lateral {p.get('lateral', 0):.3f}"
          f"  lines {', '.join(f'{x:.2f}' for x in row['lines'])}  ({row['t']:.0f} s)", flush=True)
    return U, row


def cmd_seed(N, h, s, eps):
    Gf, _ = axial_grid()
    lat = Lattice(N, h)
    Xc, Xs = zipper_mode(Gf)
    print(f"mode max|X| = {max(np.abs(Xc).max(), np.abs(Xs).max()):.3f}, eps = {eps}", flush=True)
    U0 = from_axial(lat, Gf, np.load(AX_SADDLE), mode=(Xc, Xs), eps=eps, k=1)
    print("seed:", json.dumps(describe(lat, U0).get("pair")), flush=True)
    run_point("zip", N, h, s, U0, extra=dict(seed_eps=eps))


def cmd_axial(N, h, s):
    Gf, _ = axial_grid()
    lat = Lattice(N, h)
    U0 = from_axial(lat, Gf, np.load(AX_SADDLE))
    run_point("axial", N, h, s, U0)


def cmd_coax(N, h, d):
    """Coaxial pair at centroid distance d from the axial fixed-separation state (stays coaxial on
    the lattice by the C4 symmetry): the lattice version of the coaxial curve, and string anchors."""
    Gf, _ = axial_grid()
    lat = Lattice(N, h)
    U0 = from_axial(lat, Gf, np.load(os.path.join(DATA, f"flagpair_state_cons_ne24x32_a4_k32_d{d:g}.npy")))
    run_point("coax", N, h, d, U0, K=100.0, outer=6, max_chunks=6)


def cmd_oscan(N, h, o_from, o_to, do, tag, seed, theta=0.0):
    """Constrained minima at fixed overlap O = ∫ q₀ q₂, consecutive from a seed:
    seed = 'coax<d>' (axial fixed-separation state at d, optionally hinge-tilted by theta degrees)
    or an .npz file."""
    from lattice3d import hinge
    lat = Lattice(N, h)
    if seed.startswith("coax"):
        Gf, _ = axial_grid()
        d = float(seed[4:])
        U = from_axial(lat, Gf, np.load(os.path.join(DATA, f"flagpair_state_cons_ne24x32_a4_k32_d{d:g}.npy")),
                       coords=hinge(np.radians(theta)) if theta else None)
    else:
        U = np.load(seed)["U"]
    d0 = describe(lat, U)
    print("seed: O =", d0["overlap02"], "pair", json.dumps(d0.get("pair")), flush=True)
    for O in np.arange(o_from, o_to + 0.5 * do, do):
        U, row = run_point(f"ov_{tag}", N, h, float(round(O, 4)), U, extra=dict(seed=seed, theta=theta),
                           coord="overlap", K=100.0, outer=6, max_chunks=6)


def cmd_scan(N, h, s_from, s_to, ds, kind="zip"):
    ss = np.arange(s_from, s_to + 0.5 * np.sign(ds) * abs(ds), ds)
    for s in ss:
        # nearest saved state of this kind
        rows = [r for r in load_rows(N, h)["rows"] if r.get("kind") == kind and "state" in r]
        if not rows:
            raise RuntimeError("no seed state")
        near = min(rows, key=lambda r: abs(r["s"] - s))
        U0 = np.load(near["state"])["U"]
        run_point(kind, N, h, float(round(s, 4)), U0, extra=dict(start_s=near["s"]))


def cmd_d1scan(N, h, d_from, d_to, dd, tag, seed):
    if seed == "zipseed":                       # the coaxial saddle + the k = 1 zipper mode (eps = 1)
        Gf, _ = axial_grid()
        lat = Lattice(N, h)
        U = from_axial(lat, Gf, np.load(AX_SADDLE), mode=zipper_mode(Gf), eps=1.0, k=1)
    elif seed == "axialseed":                   # the coaxial saddle, no seed (C4-symmetric on the lattice)
        Gf, _ = axial_grid()
        U = from_axial(Lattice(N, h), Gf, np.load(AX_SADDLE))
    else:
        U = np.load(seed)["U"]
    print("seed D1 =", float(Lattice(N, h).weight(jnp.asarray(U), 1)), flush=True)
    for D in np.arange(d_from, d_to + 0.5 * dd, dd):
        U, row = run_point(f"d1_{tag}", N, h, float(round(D, 4)), U, extra=dict(seed=os.path.basename(seed)),
                           coord="D1", K=100.0, outer=6, max_chunks=6)


if __name__ == "__main__":
    cmd = sys.argv[1]
    N, h = int(sys.argv[2]), float(sys.argv[3])
    if cmd == "refs":
        cmd_refs(N, h)
    elif cmd == "seed":
        cmd_seed(N, h, float(sys.argv[4]), float(sys.argv[5]))
    elif cmd == "axial":
        cmd_axial(N, h, float(sys.argv[4]))
    elif cmd == "scan":
        cmd_scan(N, h, float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
                 kind=sys.argv[7] if len(sys.argv) > 7 else "zip")
    elif cmd == "coax":
        for d in sys.argv[4:]:
            cmd_coax(N, h, float(d))
    elif cmd == "oscan":
        cmd_oscan(N, h, float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8],
                  float(sys.argv[9]) if len(sys.argv) > 9 else 0.0)
    elif cmd == "d1scan":
        cmd_d1scan(N, h, float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), sys.argv[7], sys.argv[8])
    else:
        raise SystemExit(__doc__)
