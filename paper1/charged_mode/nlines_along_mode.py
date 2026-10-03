"""
Walk along the negative off-block mode of the coaxial four-line pair (01 ⊕ 23 on top of each other):
E(s) for U = U₀ exp(ε X_mode), s = largest mixing angle, against the quadratic prediction E₀ + ½ λ ε²;
the structure of the mode (entries 02, 03, 12, 13; Pauli components; frame-rotation part); the charge
at the lowest point.  Needs nlines.py refine coaxial_on_top first.

Usage: python nlines_along_mode.py
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, ".")
os.environ.setdefault("FLAGPAIR_SCRATCH", "/tmp/flagpair")
from nlines import NLattice, embed, charge_n, SCRATCH, DATA
N, h = 41, 0.3
UA = np.load(os.path.join(SCRATCH, "lattice_ref_A01_N41_h0.3.npz"))["U"]
U0 = embed(UA, (0, 1), 4) @ embed(UA, (2, 3), 4)
x = np.load(os.path.join(SCRATCH, "nlines_offblock_vec_coaxial_on_top_refined.npy")).reshape(N, N, N, 8)
Y = (x[..., 0:4] + 1j * x[..., 4:8]).reshape(N, N, N, 2, 2)
amp = np.sqrt((np.abs(Y) ** 2).sum((-1, -2)))
# structure of the mode: Pauli components of Y (frame rotation = real identity part)
P = {"1": np.eye(2), "sx": np.array([[0, 1], [1, 0]]), "sy": np.array([[0, -1j], [1j, 0]]), "sz": np.diag([1, -1])}
w = {k: float((np.abs(np.einsum("...ij,ji->...", Y, M.conj().T) / 2) ** 2).sum()) for k, M in P.items()}
tot = sum(w.values())
wr = {k: round(v / tot, 3) for k, v in w.items()}
ent = {f"y{a}{c}": float((np.abs(Y[..., i, j]) ** 2).sum() / (np.abs(Y) ** 2).sum()) for i, a in enumerate((0, 1)) for j, c in enumerate((2, 3))}
re_id = float((np.real(np.einsum("...ii->...", Y) / 2) ** 2).sum() / (np.abs(Y) ** 2).sum())
print("mode weight per entry:", {k: round(v, 3) for k, v in ent.items()}, " Pauli components:", wr, " real-identity (frame rotation):", round(re_id, 3))
nl = NLattice(N, h, 4)

def expX(Ys):
    X = np.zeros(Ys.shape[:3] + (4, 4), complex)
    X[..., :2, 2:] = Ys
    X[..., 2:, :2] = -np.conj(np.swapaxes(Ys, -1, -2))
    H = 1j * X
    lam, V = np.linalg.eigh(H)
    return np.einsum("...ij,...j,...kj->...ik", V, np.exp(-1j * lam), V.conj())

t0 = time.time()
E0 = nl.energy(U0)
print("E(0) =", round(E0, 3), f"(compile+run {time.time() - t0:.0f} s)")
lam_mode = json.load(open(os.path.join(DATA, "nlines_offblock.json")))["coaxial_on_top"]["refined"]["lam"]
rows = []
for s in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3, 1.6):
    eps = s / amp.max()
    U = U0 @ expX(eps * Y)
    E = nl.energy(U)
    quad = E0 + 0.5 * lam_mode * eps ** 2
    rows.append((s, E, quad))
    print(f"   max mixing angle {s:4.2f}: E = {E:9.3f}   (quadratic prediction {quad:9.3f})", flush=True)
smin = min(rows, key=lambda r: r[1])
Umin = U0 @ expX(smin[0] / amp.max() * Y)
Q, Qs = charge_n(h, Umin, pad=1)
print(f"lowest along the line: s = {smin[0]}, E = {smin[1]:.3f} (drop {smin[1] - E0:+.3f}); charge there {Q:+.3f} {np.round(Qs, 3).tolist()}")
json.dump(dict(E0=E0, rows=rows, mode_entries=ent, pauli=wr, frame_rotation_part=re_id, charge_at_min=Q, charge_lines=Qs),
          open(os.path.join(DATA, "nlines_offblock_line.json"), "w"), indent=1)
