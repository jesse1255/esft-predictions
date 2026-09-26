"""
Cross-sections of lattice states (lattice3d / lattice_path): where the three lines sit.

For each state, the plane through the pair axis that contains the tilt (the plane
spanned by the mean ring normal and the relative displacement / tilt direction) is
sampled, and the defect densities q_c = 1 − |U_cc|² of line 0 (the 01 vortex, red),
line 1 (shared, green) and line 2 (the 12 vortex, blue) are drawn as filled contours.
A coaxial pair shows four blobs (two per ring, mirror images); a zipped pair shows
lines 0 and 2 merged on one side with line 1 pushed out of it.

Usage: python plot_zipper.py out.png state1.npz [state2.npz ...]   (titles from the file names)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

from lattice3d import Lattice


def section(U, h, e1, e2, extent=3.2, n=241):
    N = U.shape[0]
    s = np.linspace(-extent, extent, n)
    A, B = np.meshgrid(s, s, indexing="xy")
    P = A[..., None] * e1 + B[..., None] * e2                       # (n, n, 3)
    idx = P / h + 0.5 * (N - 1)
    out = []
    for c in range(3):
        q = 1.0 - np.abs(U[..., c, c]) ** 2
        out.append(map_coordinates(q, [idx[..., 0], idx[..., 1], idx[..., 2]], order=1))
    return s, out


def frame(lat, U):
    g = lat.geometry(U)
    n0, n2 = np.array(g[0]["normal"]), np.array(g[2]["normal"])
    nm = n0 + n2
    nm /= np.linalg.norm(nm)
    t = n0 - n2                                                     # relative tilt direction
    t -= (t @ nm) * nm
    if np.linalg.norm(t) < 1e-6:
        d = np.array(g[0]["centroid"]) - np.array(g[2]["centroid"])
        t = d - (d @ nm) * nm
    if np.linalg.norm(t) < 1e-6:
        t = np.array([1.0, 0.0, 0.0]) - nm[0] * nm
    t /= np.linalg.norm(t)
    return t, nm, g


def main():
    out = sys.argv[1]
    files = sys.argv[2:]
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "Noto Sans CJK TC", "DejaVu Sans"]
    fig, axes = plt.subplots(1, len(files), figsize=(5.2 * len(files), 5.0), squeeze=False)
    for ax, fn in zip(axes[0], files):
        U = np.load(fn)["U"]
        N = U.shape[0]
        name = os.path.basename(fn)
        h = float(name.split("_h")[1].split("_")[0].replace(".npz", ""))
        lat = Lattice(N, h)
        t, nm, g = frame(lat, U)
        s, (q0, q1, q2) = section(U, h, t, nm)
        img = 1.0 - 0.6 * (np.clip(q0, 0, 1)[..., None] * np.array([0, 1, 1])
                           + np.clip(q1, 0, 1)[..., None] * np.array([1, 0, 1])
                           + np.clip(q2, 0, 1)[..., None] * np.array([1, 1, 0]))
        ax.imshow(np.clip(img, 0, 1), extent=(s[0], s[-1], s[0], s[-1]), origin="lower")
        for q, col in ((q0, "red"), (q1, "green"), (q2, "blue")):
            ax.contour(s, s, q, levels=[0.5], colors=col, linewidths=1.8)
        p = g.get("pair", {})
        ax.set_title(f"{name[:-4]}\n相對傾角 {p.get('relative_tilt_deg', 0):.1f}°，中心距 {p.get('dist', 0):.2f}", fontsize=9)
        ax.set_xlabel("傾斜方向（合上的一側在 +）")
        ax.set_ylabel("平均環法向")
    fig.suptitle("紅：線 0（01 漩渦）　綠：線 1（共享）　藍：線 2（12 漩渦）；等值線 1 − |U_cc|² = 0.5", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
