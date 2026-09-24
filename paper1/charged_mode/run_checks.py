"""
Independent checks of the relaxed configuration (reads data/state_ne*.npz).

  python run_checks.py hessian 48      lowest eigenvalues, reflection-symmetric sector
  python run_checks.py fulldomain 32   no reflection symmetry imposed: gradient and
                                       lowest eigenvalues on the full meridional plane
  python run_checks.py starts 32       other initial shapes relax to the same state
  python run_checks.py mapping 64      compactification scale a = 2.5, 4 vs 3
  python run_checks.py transfer 48 96  shape probes re-evaluated on a finer grid
  python run_checks.py limits 48       e = 0 and μ = 0 comparison runs
  python run_checks.py betatest        β-solver convergence on a test source
  python run_checks.py summary         convergence tables → data/check_summary.json

Each command writes data/check_<name>.json.
"""

import glob
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hopfion_axisym import (AP, AR, AZ, Grid, Model, hopf_ansatz, probe_field,
                            sparse_solve)
from hansatz import HProfile
from newton import NewtonRelaxer, _frames

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PROBES = [(ell, R) for ell in (0, 2, 4) for R in (0.7, 1.4, 2.8)]


def load_state(ne):
    d = np.load(os.path.join(DATA, f"state_ne{ne}.npz"))
    G = Grid(int(d["ne"]), int(d["ne"]), p=int(d["p"]), a=float(d["a"]), half=True)
    return G, d


def save(name, obj):
    with open(os.path.join(DATA, f"check_{name}.json"), "w") as fh:
        json.dump(obj, fh, indent=1)
    print(json.dumps(obj, indent=1)[:4000])


def lumped_mass(R):
    """Nodal mass weights for the DOFs of a NewtonRelaxer (for H v = λ V v)."""
    G = R.G
    # diagonal of the consistent mass matrix ∫ N_n² d³x (strictly positive;
    # Q2 row sums ∫ N_n d³x can vanish on the axis)
    vol = np.asarray((G.PvT @ sp.diags(G.W) @ G.Pv).diagonal()).ravel()
    mass = np.zeros(R.ndof)
    for k, m in enumerate(R.masks):
        mass[R.offsets[k]:R.offsets[k + 1]] = vol[m]
    return mass


def lowest_eigs(R, U, k=8):
    M = R.M
    M.energy(U)
    frozen = (M.last["beta"], M.last["I"]) if M.charge else None
    e1, e2 = _frames(U[:, :3], R.G.sym_mask)
    H, g = R.hessian(U, e1, e2, frozen, central=True)
    mass = lumped_mass(R)
    Dm = sp.diags(1.0 / np.sqrt(mass))
    Hs = (Dm @ H @ Dm).tocsr()
    # shift-invert about a small negative shift so an exact zero mode is fine
    sigma = -1e-3
    A = (Hs - sigma * sp.identity(Hs.shape[0])).tocsc()
    lu = spla.splu(A)
    op = spla.LinearOperator(Hs.shape, matvec=lu.solve, dtype=float)
    vals, vecs = spla.eigsh(Hs, k=k, sigma=sigma, which="LM", OPinv=op)
    order = np.argsort(vals)
    return vals[order], (Dm @ vecs[:, order]), H, g, e1, e2


def mode_overlaps(R, U, vecs, e1, e2):
    """Overlap of eigenvectors with z-translation and global phase rotation."""
    G = R.G
    u = U[:, :3]
    T = np.stack([-u[:, 1], u[:, 0], np.zeros(G.nn)], axis=1)
    # ∂_z u by central differences along the node columns
    uu = u.reshape(G.nr, G.nz, 3)
    zz = G.z_n.reshape(G.nr, G.nz)
    d = np.zeros_like(uu)
    with np.errstate(invalid="ignore"):
        d[:, 1:-1] = (uu[:, 2:] - uu[:, :-2]) / (zz[:, 2:] - zz[:, :-2])[:, :, None]
    dz = np.nan_to_num(d.reshape(-1, 3))

    def to_x(vec_u):
        x = np.zeros(R.ndof)
        x[R.offsets[0]:R.offsets[1]] = np.sum(vec_u * e1, axis=1)[R.masks[0]]
        x[R.offsets[1]:R.offsets[2]] = np.sum(vec_u * e2, axis=1)[R.masks[1]]
        return x

    out = []
    gens = dict(phase=to_x(T), z_translation=to_x(dz))
    for i in range(vecs.shape[1]):
        v = vecs[:, i] / np.linalg.norm(vecs[:, i])
        row = {}
        for name, gvec in gens.items():
            n = np.linalg.norm(gvec)
            row[name] = float(abs(v @ gvec) / n) if n > 0 else 0.0
        out.append(row)
    return out


def ring_geometry(G, U):
    """Vortex-ring geometry on the z = 0 plane: position of the core (u_3 = +1,
    the preimage of the antivacuum, around which the phase of u_1 + i u_2
    winds) and the radii where u_3 crosses 0; plus the half-height of the
    u_3 > 0 tube on the ring."""
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    rho = np.linspace(0.0, 6.0, 6001)
    P = G.interp_points(rho, np.zeros_like(rho))
    u3 = P @ U[:, 2]
    i = int(np.argmax(u3))
    f = interp1d(rho, u3, kind="cubic")
    inner = brentq(lambda x: f(x), rho[1], rho[i]) if u3[0] < 0 < u3[i] else float("nan")
    outer = brentq(lambda x: f(x), rho[i], rho[-1]) if u3[i] > 0 > u3[-1] else float("nan")
    zz = np.linspace(0.0, 6.0, 6001)
    Pz = G.interp_points(np.full_like(zz, rho[i]), zz)
    u3z = Pz @ U[:, 2]
    j = np.nonzero(u3z < 0)[0]
    half_height = float(zz[j[0]]) if j.size else float("nan")
    return dict(core_radius=float(rho[i]), u3_max=float(u3[i]), inner_zero=float(inner),
                outer_zero=float(outer), tube_half_height=half_height)


# ---------------------------------------------------------------------------

def cmd_hessian(ne):
    G, d = load_state(ne)
    M = Model(G, e=0.3, N=1.0, kappa=1.0)
    U = d["V_full"]
    R = NewtonRelaxer(M, U, verbose=False)
    t = time.time()
    vals, vecs, H, g, e1, e2 = lowest_eigs(R, U)
    # where do the lowest modes live?  fraction of the (mass-weighted) norm in the
    # gauge field and beyond r = 6 (outside the soliton)
    mass = lumped_mass(R)
    r, _, _ = G.node_r_theta()
    node_of = np.concatenate([np.nonzero(m)[0] for m in R.masks])
    far = ~(np.isfinite(r[node_of]) & (r[node_of] < 6.0))
    is_A = np.arange(R.ndof) >= R.offsets[2]
    loc = []
    for i in range(vecs.shape[1]):
        w = mass * vecs[:, i] ** 2
        w /= w.sum()
        loc.append(dict(fraction_in_A=float(w[is_A].sum()), fraction_beyond_r6=float(w[far].sum())))
    # matter-only block (A frozen): the soliton's own lowest modes
    nm = R.offsets[2]
    Hm = H[:nm, :nm]
    Dm = sp.diags(1.0 / np.sqrt(mass[:nm]))
    Hs = (Dm @ Hm @ Dm).tocsc()
    lu = spla.splu((Hs + 1e-3 * sp.identity(nm)).tocsc())
    op = spla.LinearOperator(Hs.shape, matvec=lu.solve, dtype=float)
    vm = np.sort(spla.eigsh(Hs, k=6, sigma=-1e-3, which="LM", OPinv=op)[0])
    save(f"hessian_ne{ne}", dict(
        mode_localisation=loc, eigenvalues_matter_block_A_frozen=vm.tolist(),
        description="Lowest eigenvalues of the discrete Hessian (a1, a2, A_ρ, A_z, A_φ), "
                    "mass-normalised by nodal volumes, reflection-symmetric half plane, "
                    "Coulomb-gauge term κ = 1, fixed-charge term included locally "
                    "(the omitted non-local part is positive semi-definite). Central "
                    "differences. mode_localisation: share of each eigenvector in A and "
                    "beyond r = 6.",
        ne=ne, ndof=R.ndof, eigenvalues=vals.tolist(), gradient_max=float(np.abs(g).max()),
        seconds=time.time() - t))


def mirror_to_full(Gh, Uh):
    """Reflect a half-plane state to the full meridional plane."""
    Gf = Grid(Gh.ne_r, Gh.ne_z, p=Gh.p, a=Gh.a, half=False)
    Uf = Gf.vacuum()
    nzh = Gh.nz
    Uh3 = Uh.reshape(Gh.nr, nzh, 6)
    Uf3 = Uf.reshape(Gf.nr, Gf.nz, 6)
    mid = nzh - 1
    Uf3[:, mid:, :] = Uh3
    lower = Uh3[:, ::-1, :].copy()
    lower[:, :, 1] *= -1.0      # u_2 odd
    lower[:, :, AR] *= -1.0     # A_ρ odd
    Uf3[:, :mid + 1, :] = lower
    return Gf, Uf3.reshape(-1, 6)


def cmd_fulldomain(ne):
    Gh, d = load_state(ne)
    Mh = Model(Gh, e=0.3, N=1.0, kappa=1.0)
    Eh = Mh.energy(d["V_full"], want_grad=False)[0]
    Gf, Uf = mirror_to_full(Gh, d["V_full"])
    Mf = Model(Gf, e=0.3, N=1.0, kappa=1.0)
    Ef, gf = Mf.energy(Uf)
    R = NewtonRelaxer(Mf, Uf, verbose=False)
    e1, e2 = _frames(Uf[:, :3], Gf.sym_mask)
    gx = R._grad_x(Uf, gf, np.ones(Gf.nn), e1, e2)
    vals, vecs, H, g, e1, e2 = lowest_eigs(R, Uf, k=8)
    ov = mode_overlaps(R, Uf, vecs, e1, e2)
    # relax on the full plane after an odd (symmetry-breaking) kick
    kick = Uf.copy()
    r, ct, st = Gf.node_r_theta()
    rr = np.where(np.isfinite(r), r, 0.0)
    bump = 0.05 * np.exp(-(rr - 1.5) ** 2) * ct          # odd in z
    kick[:, 1] += bump * st
    kick[:, 0] += 0.02 * bump
    kick[:, :3] /= np.linalg.norm(kick[:, :3], axis=1)[:, None]
    kick[~Gf.free] = Uf[~Gf.free]
    Ek = Mf.energy(kick, want_grad=False)[0]
    R2 = NewtonRelaxer(Mf, kick, verbose=True)
    Ur, Er = R2.run(max_iter=40, tol=1e-9)
    diag = Mf.diagnostics(Ur)
    z_center = float(np.dot(Gf.W, (Gf.zq) * Mf.local(*Mf.interp(Ur), want_grad=False)[0])
                     / np.dot(Gf.W, Mf.local(*Mf.interp(Ur), want_grad=False)[0]))
    save(f"fulldomain_ne{ne}", dict(
        description="State reflected to the full meridional plane (no reflection symmetry "
                    "imposed). Gradient of the full problem, lowest mass-normalised Hessian "
                    "eigenvalues with overlaps on the exact phase zero mode and the "
                    "(lattice-broken) z-translation mode, and relaxation after an odd kick.",
        ne=ne, E_half=Eh, E_full_plane=Ef, grad_max_full_plane=float(np.abs(gx).max()),
        eigenvalues=vals.tolist(), overlaps=ov, E_after_kick=Ek, E_relaxed=Er,
        relaxed_converged=R2.converged, relaxed_Q_H=diag["Q_H"],
        relaxed_energy_centroid_z=z_center, relaxed_rms_charge=diag["rms_charge"]))


def relax_from(G, U0, label):
    M = Model(G, e=0.3, N=1.0, kappa=1.0)
    V1 = M.solve_A_fixed_u(U0)
    E0 = M.energy(V1, want_grad=False)[0]
    R = NewtonRelaxer(M, V1, verbose=True)
    V, E = R.run(max_iter=80, tol=1e-9)
    dg = M.diagnostics(V)
    print(f"  [{label}] start {E0:.6f} -> {E:.10f}  (it {len(R.history)})", flush=True)
    return dict(label=label, E_start=E0, E=E, iterations=len(R.history),
                converged=R.converged, rms_charge=dg["rms_charge"], Dzz=dg["Dzz_charge"],
                mu_volume=dg["mu_volume"], Q_H=dg["Q_H"])


def cmd_starts(ne):
    G, d = load_state(ne)
    prof = HProfile(G)
    cf = d["cf"]
    r, ct, st = G.node_r_theta()
    out = []
    # (i)/(ii) rescaled profiles; (iii)/(iv) squashed / stretched along z
    for lab, sr in (("h(r/1.3)", 1 / 1.3), ("h(1.3 r)", 1.3)):
        U, _ = hopf_ansatz(G, prof.h_of_r(cf, np.where(np.isfinite(r), r * sr, 1e9)))
        out.append(relax_from(G, U, lab))
    for lab, qz in (("h(r'), r'^2 = rho^2 + (z/0.7)^2", 0.7), ("h(r'), r'^2 = rho^2 + (z/1.4)^2", 1.4)):
        rho = np.nan_to_num(G.rho_n, posinf=1e9)
        z = np.nan_to_num(G.z_n, posinf=1e9)
        rp = np.sqrt(rho ** 2 + (z / qz) ** 2)
        # build u with the squashed radius but the true polar angle
        hq = prof.h_of_r(cf, rp)
        U, _ = hopf_ansatz(G, hq)
        out.append(relax_from(G, U, lab))
    ref = json.load(open(os.path.join(DATA, f"round_ne{ne}.json")))["full"]["E"]
    save(f"starts_ne{ne}", dict(description="Full relaxation from four other initial shapes "
                                             "(same grid) versus the run started from the "
                                             "h(r) optimum.", ne=ne, E_reference=ref, runs=out,
                                max_abs_dE=max(abs(o["E"] - ref) for o in out)))


def cmd_mapping(ne):
    Gr, d = load_state(ne)
    prof_ref = HProfile(Gr)
    cf = d["cf"]
    out = []
    for a in (2.5, 4.0):
        G = Grid(ne, ne, p=2, a=a, half=True)
        prof = HProfile(G)
        U, _ = hopf_ansatz(G, prof.h_nodes(cf))
        res = relax_from(G, U, f"a={a}")
        out.append(dict(a=a, **res))
    ref = json.load(open(os.path.join(DATA, f"round_ne{ne}.json")))["full"]
    save(f"mapping_ne{ne}", dict(description="Compactification scale a varied at fixed "
                                              "element count.", ne=ne, a_reference=3.0,
                                 E_reference=ref["E"], rms_reference=ref["diag"]["rms_charge"],
                                 runs=out))


def transfer_state(Gc, Uc, Gf):
    """Interpolate a coarse state to a finer grid (u renormalised)."""
    fin = np.isfinite(Gf.rho_n) & np.isfinite(Gf.z_n)
    P = Gc.interp_points(Gf.rho_n[fin], Gf.z_n[fin])
    U = Gf.vacuum()
    U[fin] = P @ Uc
    U[:, :3] /= np.linalg.norm(U[:, :3], axis=1)[:, None]
    U[~Gf.free] = Gf.vacuum()[~Gf.free]
    return U


def cmd_transfer(ne_c, ne_f):
    Gc, d = load_state(ne_c)
    Gf = Grid(ne_f, ne_f, p=2, a=Gc.a, half=True)
    M = Model(Gf, e=0.3, N=1.0, kappa=1e-8)
    out = {}
    for key in ("V_h", "V_full"):
        U = transfer_state(Gc, d[key], Gf)
        V = M.solve_A_fixed_u(U)
        E, g = M.energy(V)
        out[key] = dict(E_on_fine_grid=E,
                        probes={f"l{ell}_R{R}": float(np.sum(g * probe_field(Gf, V, ell, R)))
                                for ell, R in PROBES})
    save(f"transfer_{ne_c}_to_{ne_f}", dict(
        description="Coarse-grid states interpolated (Q2) to a finer grid; u kept fixed, "
                    "A and β re-solved there (gradient part of A free), then the note's "
                    "latitude probes dE/dε evaluated with the fine-grid energy.",
        ne_coarse=ne_c, ne_fine=ne_f, **out))


def cmd_limits(ne):
    """Uncharged / ungauged comparisons with the same grid and solver."""
    G, d = load_state(ne)
    out = []
    for lab, e, N, mu in (("e=0.3, N=0 (gauged, uncharged)", 0.3, 0.0, 1.0),
                          ("e=0 (no Maxwell field), mu=1", 0.0, 0.0, 1.0),
                          ("e=0, mu=0 (Faddeev-Skyrme)", 0.0, 0.0, 0.0)):
        M = Model(G, e=e, N=N, mu=mu, kappa=1.0)
        U0 = d["V_full"].copy()
        if e == 0.0:
            U0[:, 3:] = 0.0
        R = NewtonRelaxer(M, U0, verbose=True)
        V, E = R.run(max_iter=80, tol=1e-9)
        dg = M.diagnostics(V)
        br = M.breakdown(V)
        out.append(dict(label=lab, e=e, N=N, mu=mu, E=E, converged=R.converged,
                        breakdown=br, Q_H=dg["Q_H"], virial_relative=dg.get("virial_relative"),
                        mu_volume=dg["mu_volume"], mu_far=dg["mu_far"],
                        ring=ring_geometry(G, V),
                        energy_density_rms_radius=dg["energy_density_rms_radius"],
                        E_over_16pi2=E / (16 * np.pi ** 2),
                        E_over_16sqrt2pi2=E / (16 * np.sqrt(2) * np.pi ** 2)))
        print(f"  [{lab}] E = {E:.8f}  E/16π² = {E / (16 * np.pi ** 2):.6f}", flush=True)
    save(f"limits_ne{ne}", dict(description="Same grid, other couplings. E/16π² and E/(16√2π²) "
                                             "are the energy in the normalisations "
                                             "(1/32π²)∫[(∂n)² + ½(∂n×∂n)²] and "
                                             "(1/32√2π²)∫[...] used in the Hopfion literature.",
                                ne=ne, runs=out))


def cmd_betatest():
    """β solver on a spherical test source against a fine 1-D reference."""
    import scipy.sparse as sps
    e = 0.3
    Kf = lambda r: 3.0 * np.exp(-r ** 2 / 2)
    R, n = 40.0, 400001
    r = np.linspace(0, R, n)
    h = r[1] - r[0]
    rm = r[:-1] + h / 2
    Aw = rm ** 2 / e ** 2 / h
    main = np.zeros(n)
    main[:-1] += Aw
    main[1:] += Aw
    vol = np.zeros(n)
    vol[1:-1] = r[1:-1] ** 2 * h
    vol[0] = (h / 2) ** 3 / 3
    vol[-1] = R ** 2 * h / 2
    main += Kf(r) * vol
    rhs = np.zeros(n)
    main[-1] += R / e ** 2       # exact exterior (monopole) energy beyond R
    rhs[-1] += R / e ** 2
    A = sps.diags([-Aw, main, -Aw], [-1, 0, 1], format="csc")
    b = spla.spsolve(A, rhs)
    Iref = float(4 * np.pi * np.sum(Kf(r) * b * vol))
    rows = []
    for ne in (32, 48, 64, 96):
        G = Grid(ne, ne, p=2, a=3.0, half=True)
        M = Model(G, e=e, N=1.0)
        _, I = M.solve_beta(Kf(np.sqrt(G.rho ** 2 + G.zq ** 2)))
        rows.append(dict(ne=ne, I=I, error=I - Iref, error_times_ne=(I - Iref) * ne))
    ex = [dict(pair=[rows[i]["ne"], rows[j]["ne"]],
               I_extrapolated=rows[j]["I"] - (rows[i]["I"] - rows[j]["I"]) * (1 / rows[j]["ne"])
               / (1 / rows[i]["ne"] - 1 / rows[j]["ne"]))
          for i, j in ((1, 3), (2, 3))]
    for x in ex:
        x["error"] = x["I_extrapolated"] - Iref
    save("betatest", dict(description="Spherical source K = 3 exp(-r²/2), e = 0.3: the "
                                      "compactified-grid β solve has an O(1/ne) error "
                                      "(error·ne ≈ const); first-order extrapolation removes it.",
                          I_reference=Iref, rows=rows, extrapolations=ex))


def cmd_summary():
    rows = []
    for fn in sorted(glob.glob(os.path.join(DATA, "round_ne*.json")),
                     key=lambda s: int(s.split("ne")[-1].split(".")[0])):
        r = json.load(open(fn))
        ne = r["grid"]["ne"]
        G, d = load_state(ne)
        Mf = Model(G, e=0.3, N=1.0, kappa=1.0)
        Mh = Model(G, e=0.3, N=1.0, kappa=1e-8)
        vf = Mf.diagnostics(d["V_full"])
        vh = Mh.diagnostics(d["V_h"])
        rows.append(dict(ne=ne, nodes=r["grid"]["nodes"],
                         h=dict(E=r["h_family"]["opt"]["E"], **{k: vh[k] for k in (
                             "rms_charge", "r2_charge", "Dzz_charge", "I_eff", "omega",
                             "mu_volume", "mu_far", "G0_q1", "Gz_q1", "Gx_q1", "Q_H",
                             "virial_relative")},
                                dE_mer=r["h_family"]["contributions"]["dE_mer"],
                                dE_az=r["h_family"]["contributions"]["dE_az"],
                                probe_l2_R1p4=r["h_family"]["probes"]["l2_R1.4"]),
                         full=dict(E=r["full"]["E"], **{k: vf[k] for k in (
                             "rms_charge", "r2_charge", "Dzz_charge", "I_eff", "omega",
                             "mu_volume", "mu_far", "G0_q1", "Gz_q1", "Gx_q1", "Q_H",
                             "virial_relative", "gauge_fix_energy")},
                                   gauge_check_diff=r["full"]["gauge_check"]["diff"],
                                   breakdown=r["full"]["breakdown"])))

    def extrap(key, sub):
        """Richardson extrapolation v(h) = v_inf + C h^q from the last three grids.

        q is solved from (h1^q - h2^q)/(h2^q - h3^q) = (v1 - v2)/(v2 - v3); the
        quoted error is max(|v_inf - v3|, |v3 - v2| h3^q/(h2^q - h3^q)).
        """
        from scipy.optimize import brentq
        vals = [row[sub][key] for row in rows]
        hs = [1.0 / row["ne"] for row in rows]
        if len(vals) < 3:
            return None
        (h1, h2, h3), (v1, v2, v3) = hs[-3:], vals[-3:]
        d12, d23 = v1 - v2, v2 - v3
        out = dict(last=v3, last_step_change=d23)
        if d23 == 0 or d12 * d23 <= 0:
            out.update(value=v3, order=None, error_estimate=abs(d23))
            return out
        f = lambda q: (h1 ** q - h2 ** q) / (h2 ** q - h3 ** q) - d12 / d23
        try:
            q = brentq(f, 0.3, 10.0)
        except ValueError:
            out.update(value=v3, order=None, error_estimate=abs(d23))
            return out
        C = d23 / (h2 ** q - h3 ** q)
        v_inf = v3 - C * h3 ** q
        out.update(value=float(v_inf), order=float(q), error_estimate=float(abs(v_inf - v3)))
        return out

    def extrap_first_order(key, sub):
        """I_eff has an O(1/ne) error from the compactified corner (see
        check_betatest.json); extrapolate with q = 1 from the last two grids."""
        v2, v3 = rows[-2][sub][key], rows[-1][sub][key]
        h2, h3 = 1.0 / rows[-2]["ne"], 1.0 / rows[-1]["ne"]
        v_inf = v3 - (v2 - v3) * h3 / (h2 - h3)
        return dict(last=v3, value=float(v_inf), order=1.0, error_estimate=float(abs(v_inf - v3)))

    extr = {}
    for sub in ("h", "full"):
        extr[sub] = {k: extrap(k, sub) for k in ("E", "rms_charge", "Dzz_charge",
                                                 "mu_volume", "G0_q1", "Gz_q1", "Gx_q1")}
        extr[sub]["I_eff"] = extrap_first_order("I_eff", sub)
        extr[sub]["omega"] = dict(value=1.0 / extr[sub]["I_eff"]["value"])
    extr["delta_full_minus_h"] = {k: extr["full"][k]["value"] - extr["h"][k]["value"]
                                  for k in ("E", "rms_charge", "Dzz_charge", "mu_volume", "I_eff")}
    save("summary", dict(description="Grid sequence (Q2 elements, a = 3) and Richardson "
                                     "extrapolation from the last three grids.",
                         rows=rows, extrapolated=extr))


if __name__ == "__main__":
    cmd, args = sys.argv[1], [int(x) for x in sys.argv[2:]]
    dict(hessian=cmd_hessian, fulldomain=cmd_fulldomain, starts=cmd_starts,
         mapping=cmd_mapping, transfer=cmd_transfer, limits=cmd_limits, betatest=cmd_betatest,
         summary=cmd_summary)[cmd](*args)
