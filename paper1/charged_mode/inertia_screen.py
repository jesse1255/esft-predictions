"""
Cartan inertia of the static relaxed pairs (hypothesis sweep §4, item 1).

T(Ω) = ½ Ω·Λ[U]·Ω is exactly quadratic, so for each saved static state the
co-rotating (Ω = (ω, 0, −ω)) and counter-rotating (Ω = (ω, 0, ω)) kinetic energies
at ω = 1 give the small-ω coefficients of the interaction at fixed Ω:
    E_int(Ω) ≈ E_int(0) − ω² [T_pair(Ω̂) − T_A(Ω̂) − T_B(Ω̂)]  (no re-relaxation)
    c2_co(d), c2_ctr(d) = T_pair − T_A − T_B for the unit co / counter rotation.
Seconds per state.  Usage: python inertia_screen.py → data/flagpair_inertia_screen.json
"""
import json, os
import numpy as np
from run_flag_pair import setup, fmodel, embed, shifted, DATA

Gf, Uf = setup(24, 32, 4)
T = "ne24x32_a4"
out = []
for d in (3.0, 2.0, 1.5, 1.25, 1.0):
    fn = os.path.join(DATA, f"flagpair_state_cons_{T}_k32_d{d:g}.npy")
    if not os.path.exists(fn):
        continue
    U = np.load(fn)
    UA = embed(Gf, shifted(Gf, Uf, +d / 2), (0, 1))
    UB = embed(Gf, shifted(Gf, Uf, -d / 2), (1, 2))
    row = dict(d=d)
    for name, Om in (("co", (1.0, 0.0, -1.0)), ("ctr", (1.0, 0.0, 1.0))):
        fm = fmodel(Gf, 2.0, Om)
        TA, TB, Tp = fm.kinetic(UA), fm.kinetic(UB), fm.kinetic(U)
        row[f"c2_{name}"] = Tp - TA - TB
        row[f"T_pair_{name}"] = Tp
    row["T_single"] = TA
    out.append(row)
    print(f"d = {d:4.2f}:  c2_co = {row['c2_co']:+8.3f}   c2_ctr = {row['c2_ctr']:+8.3f}   (single T = {TA:.3f})", flush=True)
json.dump(dict(note=__doc__, rows=out), open(os.path.join(DATA, "flagpair_inertia_screen.json"), "w"), indent=1)
