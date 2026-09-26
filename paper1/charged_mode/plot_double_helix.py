"""
The Hopf solitons of rounds 5–6 drawn as closed double helices.

For a map n: ℝ³ → S², the preimage of a point of S² is a closed curve, and the
preimages of two different points link Q times (Q = Hopf degree).  For the axial
solitons n = R₃(mφ) u(ρ, z) (round-5 convention: vacuum u₃ = −1, core u₃ = +1),
the preimage of the equator point with phase θ is
    u₃(ρ, z) = 0   and   arg(u₁ + iu₂)(ρ, z) + mφ = θ (mod 2π):
it winds once around the core ring per meridional circuit and closes after m of
them.  The preimages of θ = 0 and θ = π are the two strands of a double helix
wound around the ring; they twist around each other Q = m·n times.

Top row: the meridional cross-section (the x–z plane through the axis) coloured by
the sign of n₁ (the two "halves") with the phase as hue; the two cores (u₃ = +1)
are a vortex and an antivortex.  Bottom row: the two strands in 3D.
Left: the single soliton (Q = 1, round 5); right: the Q = 2 ring with azimuthal
winding 2 (A₂,₁ — in round 6 the fused state of the links 01 + 12 → 02).

Usage: python plot_double_helix.py   → figures/double_helix_R6.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from run_hopf_pair import relax_single

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


def triangulation(G, box=4.0):
    ok = np.isfinite(G.rho_n) & np.isfinite(G.z_n) & (G.rho_n <= box) & (np.abs(G.z_n) <= box)
    idx = -np.ones(G.nn, int)
    idx[ok] = np.arange(ok.sum())
    tris = []
    for i in range(G.nr - 1):
        for j in range(G.nz - 1):
            a, b, c, d = i * G.nz + j, (i + 1) * G.nz + j, i * G.nz + j + 1, (i + 1) * G.nz + j + 1
            if min(idx[a], idx[b], idx[c], idx[d]) >= 0:
                tris += [(idx[a], idx[b], idx[d]), (idx[a], idx[d], idx[c])]
    return mtri.Triangulation(G.rho_n[ok], G.z_n[ok], np.array(tris)), ok


def core_contour(tri, u3, level=0.0):
    fig = plt.figure()
    cs = plt.tricontour(tri, u3, levels=[level])
    segs = cs.allsegs[0]
    plt.close(fig)
    return max(segs, key=len)          # the closed curve around the core ring


def strands(tri, u, m, thetas=(0.0, np.pi)):
    path = core_contour(tri, u[:, 2])
    iu1 = mtri.LinearTriInterpolator(tri, u[:, 0])
    iu2 = mtri.LinearTriInterpolator(tri, u[:, 1])
    rho, z = path[:, 0], path[:, 1]
    arg = np.unwrap(np.arctan2(np.asarray(iu2(rho, z)), np.asarray(iu1(rho, z))))
    wind = (arg[-1] - arg[0]) / (2 * np.pi)
    out = []
    for th in thetas:
        pts = []
        for k in range(m):                     # m meridional circuits close the curve
            phi = (th - arg + 2 * np.pi * k) / m
            pts.append(np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1))
        out.append(np.concatenate(pts))
    return out, path, wind


def linking_number(c1, c2):
    """Gauss linking integral of two closed polylines (midpoint rule)."""
    a, b = c1, c2
    da, db = np.roll(a, -1, 0) - a, np.roll(b, -1, 0) - b
    ma, mb = a + 0.5 * da, b + 0.5 * db
    r = ma[:, None, :] - mb[None, :, :]
    cr = np.cross(da[:, None, :], db[None, :, :])
    return float(np.sum(np.einsum("ijk,ijk->ij", cr, r) / np.linalg.norm(r, axis=2) ** 3) / (4 * np.pi))


def main():
    os.makedirs(FIG, exist_ok=True)
    cases = []
    for m, title in ((1, "Q = 1：一個漩渦"), (2, "Q = 2：融合後的雙繞環")):
        _, _, Gf, Uf = relax_single(24, 32, 4, m=m, verbose=False)
        tri, ok = triangulation(Gf)
        cases.append((m, title, Gf, Uf[ok, :3], tri))
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
    fig = plt.figure(figsize=(12, 11))
    for col, (m, title, G, u, tri) in enumerate(cases):
        # --- meridional cross-section: x > 0 is φ = 0, x < 0 is φ = π (n = R₃(mπ)u) ---
        ax = fig.add_subplot(2, 2, col + 1)
        xs = np.linspace(-3.2, 3.2, 481)
        zs = np.linspace(-3.2, 3.2, 481)
        X, Z = np.meshgrid(xs, zs)
        R = np.abs(X)
        f = [mtri.LinearTriInterpolator(tri, u[:, k]) for k in range(3)]
        n1, n2, n3 = (np.asarray(fk(R, Z)) for fk in f)
        sgn = np.where(X < 0, np.cos(m * np.pi), 1.0)
        n1, n2 = sgn * n1, sgn * n2
        hue = (np.arctan2(n2, n1) / (2 * np.pi)) % 1.0
        light = 0.5 - 0.35 * n3                   # core (n₃ = +1) dark, vacuum light
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(np.stack([hue, 0.55 + 0 * hue, np.clip(light + 0.25, 0, 1)], axis=-1))
        ax.imshow(rgb, extent=(xs[0], xs[-1], zs[0], zs[-1]), origin="lower")
        ax.contour(X, Z, n1, levels=[0.0], colors="k", linewidths=2.0)
        ax.contour(X, Z, n3, levels=[0.0], colors="w", linewidths=1.0, linestyles="--")
        ax.set_title(f"{title}\n剖面（含軸）：顏色＝相位，黑線 n₁ = 0，白虛線 n₃ = 0", fontsize=10)
        ax.set_xlabel("x（x < 0 是 φ = π 那一側）")
        ax.set_ylabel("z")
        # --- the two strands in 3D ---
        ax3 = fig.add_subplot(2, 2, col + 3, projection="3d")
        (s0, s1), path, wind = strands(tri, u, m)
        Lk = linking_number(s0, s1)
        extra, _, _ = strands(tri, u, m, thetas=[k * np.pi / 3 for k in (1, 2, 4, 5)])
        ph = np.linspace(0, 2 * np.pi, 60)
        Pr, Pz = path[::4, 0], path[::4, 1]
        ax3.plot_surface(Pr[:, None] * np.cos(ph)[None, :], Pr[:, None] * np.sin(ph)[None, :],
                         np.repeat(Pz[:, None], ph.size, 1), color="0.7", alpha=0.12, linewidth=0)
        for c in extra:
            ax3.plot(*c.T, color="0.45", lw=0.8, alpha=0.6)
        ax3.plot(*s0.T, color="crimson", lw=2.5, label="相位 0 的那一股")
        ax3.plot(*s1.T, color="navy", lw=2.5, label="相位 π 的那一股")
        ax3.set_box_aspect((1, 1, 0.6))
        ax3.set_xlim(-2.4, 2.4)
        ax3.set_ylim(-2.4, 2.4)
        ax3.set_zlim(-1.5, 1.5)
        ax3.view_init(elev=38, azim=-55)
        ax3.set_title(f"兩股（紅、藍）互相纏繞 {abs(Lk):.2f} 次（Gauss 連結數）\n灰線：其他相位的股，整體是一條扭轉的繩", fontsize=10)
        ax3.legend(loc="upper left", fontsize=8)
        print(f"m = {m}: meridional winding {wind:+.3f}, linking number of the two strands {Lk:+.4f}")
    fig.tight_layout()
    fn = os.path.join(FIG, "double_helix_R6.png")
    fig.savefig(fn, dpi=130)
    print("wrote", fn)


if __name__ == "__main__":
    main()
