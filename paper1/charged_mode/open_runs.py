"""
Runs of the openable-core flag model (lattice_open.py).

    python open_runs.py single N h lam [lam ...]   → relax the Q = 1 vortex; is it still there?
    python open_runs.py cross  N h lam seed.npz tag → relax a two-ring seed (e.g. the touching
                                                     Smith configuration) with openable cores

Rows: data/flagpair_open_<kind>_N<N>_h<h>.json; states: $FLAGPAIR_SCRATCH/open_<kind>_<tag>_lam<lam>.npz
"""

import json
import os
import sys

import numpy as np

from lattice3d import Lattice, charge, SCRATCH, DATA
from lattice_open import OpenLattice, relax_open


def describe_factory(lat):
    def describe(U, R):
        g = lat.geometry(U)
        q = [1.0 - np.abs(U[..., c, c]) ** 2 for c in range(3)]
        w = [float(lat.h ** 3 * np.sum(R[..., c] ** 2 * q[c])) for c in range(3)]
        return dict(line_weights=w, open_volume=[float(lat.h ** 3 * np.sum(R[..., c] < 0.5)) for c in range(3)])
    return describe


def save_json(kind, N, h, row):
    fn = os.path.join(DATA, f"flagpair_open_{kind}_N{N}_h{h:g}.json")
    d = json.load(open(fn)) if os.path.exists(fn) else dict(rows=[])
    d["rows"].append(row)
    json.dump(d, open(fn, "w"), indent=1)


def cmd_single(N, h, lams):
    U0 = np.load(os.path.join(SCRATCH, f"lattice_ref_A01_N{N}_h{h:g}.npz"))["U"]
    for lam in lams:
        lat = OpenLattice(N, h, lam=lam)
        U, R, hist = relax_open(lat, U0, chunk=100, max_chunks=30, gtol=1e-4, describe=describe_factory(lat))
        Q, Qa, _ = charge(lat, U, pad=1)
        row = dict(lam=lam, E=hist[-1]["E"], parts=lat.parts2(U, R), rho_min=hist[-1]["rho_min"], Q_frame=Q,
                   hist=hist)
        save_json("single", N, h, row)
        np.savez_compressed(os.path.join(SCRATCH, f"open_single_lam{lam:g}_N{N}_h{h:g}.npz"), U=U, R=R)
        print(f"lambda = {lam:g}: E = {row['E']:.4f}  rho_min = {[round(v, 3) for v in row['rho_min']]}  Q(frame) = {Q:+.4f}",
              flush=True)


def cmd_cross(N, h, lam, seed, tag):
    U0 = np.load(seed)["U"]
    lat = OpenLattice(N, h, lam=lam)
    log = os.path.join(SCRATCH, f"open_cross_{tag}_lam{lam:g}.log")
    U, R, hist = relax_open(lat, U0, chunk=100, max_chunks=40, gtol=1e-4, log=log, describe=describe_factory(lat))
    Q, Qa, _ = charge(lat, U, pad=1)
    row = dict(tag=tag, lam=lam, seed=os.path.basename(seed), E=hist[-1]["E"], parts=lat.parts2(U, R),
               rho_min=hist[-1]["rho_min"], Q_frame=Q, Q_per_line=Qa, hist=hist)
    save_json("cross", N, h, row)
    np.savez_compressed(os.path.join(SCRATCH, f"open_cross_{tag}_lam{lam:g}_N{N}_h{h:g}.npz"), U=U, R=R)
    print(f"{tag} lambda = {lam:g}: E = {row['E']:.4f}  rho_min = {[round(v, 3) for v in row['rho_min']]}  Q(frame) = {Q:+.4f}",
          flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    N, h = int(sys.argv[2]), float(sys.argv[3])
    if cmd == "single":
        cmd_single(N, h, [float(v) for v in sys.argv[4:]])
    elif cmd == "cross":
        cmd_cross(N, h, float(sys.argv[4]), sys.argv[5], sys.argv[6])
