"""
Fractal (fractional) dimension: what it is, and whether our vortex rings have one.

The user's suggestion (2026-09-27): instead of circles and waves, try fractals; "the fractal hidden
in binary", what a fractional dimension is.

Box-counting dimension.  Cover a set with boxes of side s and count the boxes N(s) that contain
part of it.  A curve needs N ∝ 1/s boxes, a surface 1/s², a solid 1/s³: D = −d log N / d log s.
For a fractal the slope is the same non-integer number over many scales.

1. binary   The fractal hidden in binary: the cells (n, k) with k AND (n − k) == 0 are the odd
            binomial coefficients (Lucas / Kummer), i.e. Pascal's triangle mod 2 = the Sierpiński
            triangle.  Its dimension is log 3 / log 2 = 1.585 (each doubling of the size triples the
            count).  Used here to validate the box counter.
2. vortex   The same counter on the core set of our lattice fields (sites where 1 − |U_cc|² > thr):
            the single vortex, the Q = 2 ring, and collision snapshots.  The local slope of
            log N vs log s gives an effective dimension at each scale.

Usage: python fractal_dims.py binary
       python fractal_dims.py vortex      (the lattice fields, h = 0.3: too coarse, see the notes)
       python fractal_dims.py fine        (the single vortex on a fine grid, two decades of scale)
"""

import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def box_count(mask, sizes):
    """Number of boxes of side s (in cells) that contain at least one True cell, for each s.
    The grid is padded with False up to a multiple of s; boxes are aligned at the origin, and the
    count is minimised over the s^d possible offsets taken in steps of max(1, s // 4)."""
    d = mask.ndim
    out = []
    for s in sizes:
        best = None
        step = max(1, s // 4)
        offs = range(0, s, step)
        for off in (np.array(o) for o in np.stack(np.meshgrid(*([list(offs)] * d), indexing="ij"), -1).reshape(-1, d)):
            pad = [(int(off[i]), int((-(mask.shape[i] + off[i])) % s)) for i in range(d)]
            m = np.pad(mask, pad)
            shp = []
            for i in range(d):
                shp += [m.shape[i] // s, s]
            n = int(m.reshape(shp).any(axis=tuple(range(1, 2 * d, 2))).sum())
            best = n if best is None else min(best, n)
        out.append(best)
    return np.array(out)


def slopes(sizes, counts):
    ls, lc = np.log(sizes), np.log(counts)
    loc = -np.diff(lc) / np.diff(ls)
    return loc


def fit(sizes, counts, lo, hi):
    sel = (sizes >= lo) & (sizes <= hi)
    p = np.polyfit(np.log(sizes[sel]), np.log(counts[sel]), 1)
    return -p[0]


def cmd_binary():
    """Pascal's triangle mod 2 on 2^n rows: the Sierpiński triangle hidden in binary."""
    rows = []
    for n_exp in (10, 11, 12):
        n = 2 ** n_exp
        i = np.arange(n)
        N_, K_ = np.meshgrid(i, i, indexing="ij")
        mask = (K_ <= N_) & ((K_ & (N_ - K_)) == 0)
        sizes = np.array([2 ** j for j in range(0, n_exp - 1)])
        counts = box_count(mask, sizes)
        D = fit(sizes, counts, 1, sizes[-1])
        rows.append(dict(rows=n, sizes=sizes.tolist(), counts=counts.tolist(), D_fit=D,
                         local=[round(v, 4) for v in slopes(sizes, counts)]))
        print(f"Pascal mod 2, {n} rows: D = {D:.4f}  (log 3/log 2 = {np.log(3) / np.log(2):.4f});"
              f" local slopes {[round(v, 3) for v in slopes(sizes, counts)]}", flush=True)
    # control: a filled triangle (dimension 2) and a straight line (dimension 1)
    n = 2 ** 11
    i = np.arange(n)
    N_, K_ = np.meshgrid(i, i, indexing="ij")
    sizes = np.array([2 ** j for j in range(0, 10)])
    D_tri = fit(sizes, box_count(K_ <= N_, sizes), 1, sizes[-1])
    D_line = fit(sizes, box_count(K_ == N_, sizes), 1, sizes[-1])
    print(f"controls: filled triangle D = {D_tri:.4f}, diagonal line D = {D_line:.4f}", flush=True)
    json.dump(dict(pascal_mod2=rows, filled_triangle=D_tri, line=D_line, exact=np.log(3) / np.log(2)),
              open(os.path.join(DATA, "fractal_binary_sierpinski.json"), "w"), indent=1)


def core_mask(U, thr):
    q = [1.0 - np.abs(U[..., c, c]) ** 2 for c in range(3)]
    return np.max(np.stack(q), axis=0) > thr


def cmd_vortex(h=0.3):
    cases = [("單一漩渦 Q = 1", os.path.join(SCRATCH, "lattice_ref_A01_N41_h0.3.npz")),
             ("Q = 2 環", os.path.join(SCRATCH, "lattice_ref_fused02_N41_h0.3.npz"))]
    for V in ("0.8",):
        for t in ("5.00", "10.00", "30.00"):
            cases.append((f"對撞 V = {V}，t = {float(t):g}", os.path.join(SCRATCH, f"dyn2_collide_touch_V{V}_N41_sponge_t{t}.npz")))
    sizes = np.array([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20])
    out = []
    for name, fn in cases:
        if not os.path.exists(fn):
            print("missing", fn)
            continue
        U = np.load(fn)["U"]
        for thr in (0.5, 0.9):
            m = core_mask(U, thr)
            if m.sum() == 0:
                continue
            c = box_count(m, sizes)
            loc = slopes(sizes, c)
            mids = np.sqrt(sizes[1:] * sizes[:-1]) * h
            row = dict(case=name, file=os.path.basename(fn), thr=thr, cells=int(m.sum()), sizes=(sizes * h).tolist(),
                       counts=c.tolist(), local_slope=[round(v, 3) for v in loc], scale_mid=[round(v, 3) for v in mids])
            out.append(row)
            print(f"{name:22s} thr {thr}: cells {m.sum():6d}  local D at scales {np.round(mids, 2).tolist()}:"
                  f" {[round(v, 2) for v in loc]}", flush=True)
    json.dump(out, open(os.path.join(DATA, "fractal_vortex_boxcount.json"), "w"), indent=1)


def cmd_fine(hf=0.04, L=4.0):
    """The single vortex from the axisymmetric solution (§5.4 reference), evaluated on a fine grid
    (spacing hf over ±L): core sets q = 1 − |U_00|² > thr, box counts over two decades, and the
    local slope (effective dimension) at each scale.  A smooth ring of tube radius a and ring radius
    R should read 3 below a, about 1 between a and R, and 0 above R."""
    sys.path.insert(0, HERE)
    from run_flag_pair import setup
    from smith import cp1_at
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    x1 = np.arange(-L, L + 1e-9, hf)
    n = len(x1)
    X, Y = np.meshgrid(x1, x1, indexing="ij")
    q = np.zeros((n, n, n), dtype=np.float32)
    for k, z in enumerate(x1):
        nv = cp1_at(Gf, u, X, Y, np.full_like(X, z))
        q[:, :, k] = 0.5 * (1.0 + nv[..., 2])
    print("grid", n, "^3, q max", float(q.max()), flush=True)
    sizes = np.unique(np.round(np.logspace(0, np.log10(n // 2), 18)).astype(int))
    out = []
    for thr in (0.5, 0.9, 0.99):
        m = q > thr
        c = box_count(m, sizes)
        loc = slopes(sizes, c)
        mids = np.sqrt(sizes[1:] * sizes[:-1]) * hf
        out.append(dict(thr=thr, cells=int(m.sum()), sizes=(sizes * hf).tolist(), counts=c.tolist(),
                        local_slope=[round(v, 3) for v in loc], scale_mid=[round(v, 4) for v in mids]))
        print(f"thr {thr}: cells {int(m.sum())}", flush=True)
        for s, d in zip(mids, loc):
            print(f"   scale {s:6.3f}   local D {d:5.2f}", flush=True)
    # geometry of the core for reference: ring radius and tube radius at thr 0.5
    idx = np.argwhere(q > 0.5)
    r = np.hypot(x1[idx[:, 0]], x1[idx[:, 1]])
    print("core (q > 0.5): ring radius range", float(r.min()), float(r.max()), "z range",
          float(x1[idx[:, 2]].min()), float(x1[idx[:, 2]].max()), flush=True)
    json.dump(dict(hf=hf, L=L, cases=out), open(os.path.join(DATA, "fractal_vortex_fine.json"), "w"), indent=1)


if __name__ == "__main__":
    {"binary": cmd_binary, "vortex": cmd_vortex, "fine": cmd_fine}[sys.argv[1]]()
