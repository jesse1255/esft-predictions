"""
"Smith rings": two vortex rings of different links (01 and 12) placed in perpendicular
planes so that their cores meet, and the Smith chart pulled back through one vortex.

The Smith chart (史密斯圓圖) is the Cartesian grid of normalised impedances z = r + ix seen
through Γ = (z − 1)/(z + 1): constant-r circles and constant-x circles, all through Γ = 1
(z = ∞), crossing at right angles.  In the field language, Γ is a stereographic coordinate
on the target sphere of one line (CP¹ ⊂ F₂) with Γ = 1 at the vacuum value; pulling the two
circle families back through a vortex gives two families of surfaces in space that all
start from the vacuum's preimage (the axis and infinity): one object, two families.

Two-ring configurations (product ansatz U = U_A U_B, A in block (0, 1), B in block (1, 2);
the relative phases do not matter at this level, §2.1), R = core radius of one vortex:
    touch   A in the xy plane centred at (−R, 0, 0), B in the xz plane centred at (+R, 0, 0):
            the cores meet only at the origin, with perpendicular tangents, and open to
            opposite sides (two circles through one point, as in the Smith chart)
    link+   A in the xy plane at the origin, B in the xz plane centred at (R, 0, 0): chain
            links (each core passes through the other's disc); link− the same with B's
            orientation reversed (the sign of the linking number flips)
    cross2  both centred at the origin in perpendicular planes: the cores cross at (±R, 0, 0)

Usage:
    python smith.py seeds N h [smooth] → $FLAGPAIR_SCRATCH/lattice_seed_smith_<cfg>[_gs]_N<N>_h<h>.npz, energies
                                         (smooth: line-by-line construction smooth_pair instead of U_A U_B)
    python smith.py chart              → figures/smith_chart_R6.png (2D chart and its pull-back)
"""

import json
import os
import sys

import numpy as np

from lattice3d import Lattice, SCRATCH, DATA

HERE = os.path.dirname(os.path.abspath(__file__))


def _lagrange2(t):
    return np.stack([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], axis=-1)


def cp1_at(Gf, u, x, y, z, m=1):
    """Axial CP¹ vortex n = R₃(mφ) u(ρ, z) (round-5 convention: vacuum u₃ = −1) at 3D points."""
    a = Gf.a
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    xi = (2.0 / np.pi) * np.arctan(rho / a)
    eta = (2.0 / np.pi) * np.arctan(z / a)
    ne_r, ne_z = (Gf.nr - 1) // 2, (Gf.nz - 1) // 2
    er = np.clip(np.floor(xi * ne_r).astype(int), 0, ne_r - 1)
    tr = 2.0 * (xi * ne_r - er) - 1.0
    ez = np.clip(np.floor((eta + 1.0) * 0.5 * ne_z).astype(int), 0, ne_z - 1)
    tz = 2.0 * ((eta + 1.0) * 0.5 * ne_z - ez) - 1.0
    Br, Bz = _lagrange2(tr), _lagrange2(tz)
    v = np.zeros(x.shape + (3,))
    for i in range(3):
        for j in range(3):
            node = (2 * er + i) * Gf.nz + (2 * ez + j)
            v += (Br[..., i] * Bz[..., j])[..., None] * u[node]
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    c, s = np.cos(m * phi), np.sin(m * phi)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], axis=-1)


def embed_pts(n, block):
    """Unit vectors (round-5 convention) → flag field in block (a, b) (as run_flag_pair.embed)."""
    n1, n2, n3 = n[..., 0], n[..., 1], -n[..., 2]
    z1 = np.sqrt(np.clip(0.5 * (1.0 + n3), 0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(z1 > 1e-12, (n1 + 1j * n2) / (2.0 * np.maximum(z1, 1e-300)), 1.0 + 0j)
    nrm = np.sqrt(z1 ** 2 + np.abs(z2) ** 2)
    z1, z2 = z1 / nrm, z2 / nrm
    a, b = block
    c = 3 - a - b
    U = np.zeros(n.shape[:-1] + (3, 3), complex)
    U[..., a, a] = z1
    U[..., b, a] = z2
    U[..., a, b] = -np.conj(z2)
    U[..., b, b] = z1
    U[..., c, c] = 1.0
    return U


def rot_x(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ring(Gf, u, lat, R=np.eye(3), a=(0.0, 0.0, 0.0), block=(0, 1)):
    """One vortex, rotated by R (its normal ẑ → R ẑ) and centred at a: n'(x) = n(R⁻¹(x − a))."""
    P = np.stack([lat.X - a[0], lat.Y - a[1], lat.Z - a[2]], axis=-1) @ R      # rows: (R⁻¹ p)ᵀ = pᵀ R
    n = cp1_at(Gf, u, P[..., 0], P[..., 1], P[..., 2])
    return embed_pts(n, block)


def smooth_pair(UA, UB):
    """Smooth two-vortex flag field: line 0 from A (block (0, 1)), line 2 from B (block (1, 2)),
    Gram–Schmidt, line 1 = the conjugated cross product.  Lines are gauge invariant, so this is
    continuous wherever the two cores do not meet (the product U_A U_B is not: near A's core its
    third column carries the winding phase of A's gauge, times B's off-diagonal amplitude)."""
    l0 = UA[..., :, 0]
    l2 = UB[..., :, 2]
    l2 = l2 - np.einsum("...i,...i->...", np.conj(l0), l2)[..., None] * l0
    nrm = np.linalg.norm(l2, axis=-1, keepdims=True)
    bad = nrm[..., 0] < 1e-8
    l2 = l2 / np.maximum(nrm, 1e-300)
    l1 = np.conj(np.cross(l2, l0))
    l1 /= np.linalg.norm(l1, axis=-1, keepdims=True)
    U = np.stack([l0, l1, l2], axis=-1)
    return U, int(bad.sum())


def configs(Rc):
    Ry = rot_x(-90.0)          # normal ẑ → ŷ (ring in the xz plane)
    Rym = rot_x(90.0)          # normal ẑ → −ŷ (same plane, reversed orientation)
    I = np.eye(3)
    return {
        "touch": ((I, (-Rc, 0.0, 0.0)), (Ry, (Rc, 0.0, 0.0))),
        "linkp": ((I, (-0.5 * Rc, 0.0, 0.0)), (Ry, (0.5 * Rc, 0.0, 0.0))),
        "linkm": ((I, (-0.5 * Rc, 0.0, 0.0)), (Rym, (0.5 * Rc, 0.0, 0.0))),
        "cross2": ((I, (0.0, 0.0, 0.0)), (Ry, (0.0, 0.0, 0.0))),
    }


def cmd_seeds(N, h, Rc=0.93, smooth=False):
    from run_flag_pair import setup
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    lat = Lattice(N, h)
    out = {}
    for name, ((RA, aA), (RB, aB)) in configs(Rc).items():
        UA = ring(Gf, u, lat, RA, aA, (0, 1))
        UB = ring(Gf, u, lat, RB, aB, (1, 2))
        if smooth:
            U, nbad = smooth_pair(UA, UB)
            name = name + "_gs"
        else:
            U, nbad = UA @ UB, -1
        for s in (U,):
            s[0], s[-1], s[:, 0], s[:, -1], s[:, :, 0], s[:, :, -1] = (np.eye(3),) * 6
        fn = os.path.join(SCRATCH, f"lattice_seed_smith_{name}_N{N}_h{h:g}.npz")
        np.savez_compressed(fn, U=U)
        g = lat.geometry(U)
        E = lat.energy(U)
        out[name] = dict(E=E, lines=[g[c]["weight"] for c in range(3)], pair=g.get("pair"), file=fn,
                         degenerate_sites=nbad)
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()
                                if k != "file"}), flush=True)
    return out


def gamma_to_n(G):
    """Inverse of Γ(n): stereographic from the rotated sphere m = (−n₃, n₂, n₁), Γ = (m₁ + i m₂)/(1 − m₃)."""
    G = np.asarray(G, complex)
    d = np.abs(G) ** 2 + 1.0
    m1, m2, m3 = 2 * G.real / d, 2 * G.imag / d, (np.abs(G) ** 2 - 1.0) / d
    return np.stack([m3, m2, -m1], axis=-1)


def n_to_gamma(n):
    m1, m2, m3 = -n[..., 2], n[..., 1], n[..., 0]
    return (m1 + 1j * m2) / (1.0 - m3 + 1e-15)


def preimage_loop(Gf, u, nstar, tri=None):
    """The closed curve {x : n(x) = n*} of the axial vortex n = R₃(φ)u(ρ, z): the contour
    u₃ = n*₃ in the meridional half plane, with φ = arg(n*₁ + i n*₂) − arg(u₁ + i u₂)."""
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from plot_double_helix import triangulation
    if tri is None:
        tri, ok = triangulation(Gf)
    else:
        tri, ok = tri
    uu = u[ok]
    fig = plt.figure()
    cs = plt.tricontour(tri, uu[:, 2], levels=[nstar[2]])
    segs = cs.allsegs[0]
    plt.close(fig)
    if not segs:
        return None
    path = max(segs, key=len)
    i1 = mtri.LinearTriInterpolator(tri, uu[:, 0])
    i2 = mtri.LinearTriInterpolator(tri, uu[:, 1])
    rho, z = path[:, 0], path[:, 1]
    arg = np.asarray(np.arctan2(np.asarray(i2(rho, z)), np.asarray(i1(rho, z))))
    phi = np.arctan2(nstar[1], nstar[0]) - arg
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def cmd_chart():
    """The Smith chart, its pull-back through the axial Q = 1 vortex (section through the axis),
    and the 3D version: every point of the chart is a closed loop around the core; a constant-r
    circle sweeps one family of loops, a constant-x circle the other."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_flag_pair import setup
    from plot_double_helix import triangulation
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    fig = plt.figure(figsize=(17, 6.2))
    # (a) the chart itself
    ax = fig.add_subplot(1, 3, 1)
    t = np.linspace(0, 2 * np.pi, 800)
    for r in (0.0, 0.2, 0.5, 1.0, 2.0, 5.0):
        c, rad = r / (1 + r), 1 / (1 + r)
        ax.plot(c + rad * np.cos(t), rad * np.sin(t), color="crimson", lw=1.2 if r == 1.0 else 0.8)
    for x in (0.2, 0.5, 1.0, 2.0, 5.0):
        for sx in (1, -1):
            cy, rad = sx / x, 1 / x
            px, py = 1 + rad * np.cos(t), cy + rad * np.sin(t)
            inside = np.hypot(px, py) <= 1.0001
            ax.plot(np.where(inside, px, np.nan), np.where(inside, py, np.nan), color="navy",
                    lw=1.2 if (x == 1.0 and sx == 1) else 0.8)
    ax.plot([1], [0], "ko", ms=7)
    ax.plot([-1], [0], "k*", ms=11)
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_title("史密斯圓圖：紅＝等電阻圓，藍＝等電抗圓\n全部通過 Γ = 1（●，場的真空值）；★ Γ = −1（核心的值）", fontsize=10)
    ax.axis("off")
    # (b) pull-back through the vortex, section y = 0
    ax = fig.add_subplot(1, 3, 2)
    xs = np.linspace(-3.0, 3.0, 601)
    X, Z = np.meshgrid(xs, xs)
    n = cp1_at(Gf, u, X, np.zeros_like(X), Z)
    zimp = (1 + n_to_gamma(n)) / (1 - n_to_gamma(n) + 1e-15)
    ax.contour(X, Z, np.log10(np.clip(zimp.real, 1e-3, 1e3)), levels=np.linspace(-1.5, 1.5, 13),
               colors="crimson", linewidths=0.9, linestyles="solid")
    ax.contour(X, Z, np.arcsinh(zimp.imag), levels=np.linspace(-3, 3, 13), colors="navy", linewidths=0.9,
               linestyles="solid")
    ax.plot([-0.93, 0.93], [0, 0], "k*", ms=11)
    ax.set_aspect("equal")
    ax.set_title("拉回到空間（過對稱軸的剖面，Q = 1 漩渦）\n兩族曲面都從軸線與無窮遠（真空的原像）發出，包住核心 ★", fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("z")
    # (c) 3D: loops of the points on the r = 1 circle (red) and on the x = +1 circle (blue)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    tri = triangulation(Gf)
    for fam, col in (("r", "crimson"), ("x", "navy")):
        for k in range(9):
            if fam == "r":            # |Γ − ½| = ½, avoiding the common point Γ = 1
                th = np.pi * (0.25 + 1.5 * k / 8)
                G = 0.5 + 0.5 * np.exp(1j * th)
            else:                     # |Γ − (1 + i)| = 1, inside the unit disc
                th = np.pi * (1.0 + 0.5 * (0.1 + 0.8 * k / 8))
                G = (1 + 1j) + np.exp(1j * th)
            loop = preimage_loop(Gf, u, gamma_to_n(G), tri)
            if loop is not None:
                ax.plot(*loop.T, color=col, lw=1.1, alpha=0.85)
    ph = np.linspace(0, 2 * np.pi, 100)
    ax.plot(0.93 * np.cos(ph), 0.93 * np.sin(ph), 0 * ph, "k-", lw=2.5)
    ax.plot([0, 0], [0, 0], [-3, 3], color="0.5", lw=1, ls="--")
    lim = 2.4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-60)
    ax.set_title("三維版本：圖上每一點＝空間中一條繞核心（黑）的閉環\n紅＝r = 1 圓上各點的環，藍＝x = 1 圓上各點的環（虛線：對稱軸）",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fn = os.path.join(HERE, "figures", "smith_chart_R6.png")
    fig.savefig(fn, dpi=120)
    print("wrote", fn)


if __name__ == "__main__":
    if sys.argv[1] == "seeds":
        cmd_seeds(int(sys.argv[2]), float(sys.argv[3]), smooth=len(sys.argv) > 4 and sys.argv[4] == "smooth")
    elif sys.argv[1] == "chart":
        cmd_chart()
