"""
Tables for FLAG_PAIR_R6.md from the run_flag_pair.py outputs.

1. Shared-line pair at the superposition level: E_int(d; κ₃) for κ₃ = 0, 1, 2,
   split into sigma / quartic / potential parts.  The three-cycle term enters
   linearly, E_int(κ₃) = E_int(0) + κ₃ K(d); K(d) ≥ 0 is checked against the
   κ₃ = 1 run (the residual of the straight line).
2. Same line versus shared line: the shared-line interaction is compared with
   the square of the same-line interference amplitude B1(d) of the flag model
   (first order in each tail versus second order).
3. Q = 2 in block (0, 2) and the lowest Hessian eigenvalues (subcommand q2).
4. Shared-line pair relaxed at fixed separation (subcommand cons).

Usage: python summarize_flag_pair.py   → data/flagpair_summary.json
"""

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
T = "ne24x32_a4"


def load(name):
    fn = os.path.join(DATA, name)
    return json.load(open(fn)) if os.path.exists(fn) else None


def main():
    out = {}
    parts = load(f"flagpair_parts_{T}.json")
    same = load(f"flagpair_map_same_AB_{T}_k32.json")
    B1 = {round(r["d"], 3): r["B1"] for r in same["rows"]} if same else {}
    if parts:
        rows = []
        print("== shared-line pair, superposition (E_int, parts)")
        k0, k1, k2 = parts["0.0"], parts["1.0"], parts["2.0"]
        for r0, r1, r2 in zip(k0, k1, k2):
            d = r0["d"]
            K = (r2["E_int"] - r0["E_int"]) / 2.0
            lin_res = r1["E_int"] - (r0["E_int"] + K)
            row = dict(d=d, E_int={"0": r0["E_int"], "1": r1["E_int"], "2": r2["E_int"]},
                       sigma=r0["parts_int"]["sigma"], quartic_k0=r0["parts_int"]["skyrme"],
                       pot=r0["parts_int"]["pot"], K=K, linearity_residual=lin_res,
                       order_AB_minus_BA_k2=r2["E_int"] - r2["E_int_BA"])
            if round(d, 3) in B1:
                row["B1_same"] = B1[round(d, 3)]
                row["ratio_k2_over_B1sq"] = r2["E_int"] / B1[round(d, 3)] ** 2
                row["ratio_k0_over_B1sq"] = r0["E_int"] / B1[round(d, 3)] ** 2
            rows.append(row)
            print(f"d = {d:3.1f}  E_int(κ₃=0,1,2) = {r0['E_int']:+9.5f} {r1['E_int']:+9.5f} {r2['E_int']:+9.5f}"
                  f"  σ {row['sigma']:+8.5f}  quartic(0) {row['quartic_k0']:+8.5f}  pot {row['pot']:+8.5f}"
                  f"  K {K:8.5f}  lin.res {lin_res:+.1e}"
                  + (f"  B1 {row['B1_same']:8.5f}  E/B1² (κ₃=2) {row['ratio_k2_over_B1sq']:.4f}"
                     f" (κ₃=0) {row['ratio_k0_over_B1sq']:.4f}" if "B1_same" in row else ""))
        ds = np.array([r["d"] for r in rows])
        for key, vals in (("K", [r["K"] for r in rows]), ("sigma", [r["sigma"] for r in rows]),
                          ("B1_same", [B1.get(round(d, 3), np.nan) for d in ds])):
            v = np.log(np.array(vals))
            sl = np.diff(v) / np.diff(ds)
            print(f"   log-slope of {key:7s}: " + "  ".join(f"{s:+.2f}" for s in sl))
        out["superposition_shared"] = rows
    shared = load(f"flagpair_map_shared_AB_{T}_k32.json")
    if shared:
        out["phase_dependence_shared"] = [dict(d=r["d"], spread=r["E_max"] - r["E_min"], mean=r["E_mean"])
                                          for r in shared["rows"]]
        print("== phase dependence of the shared-line pair (max − min over the 12×12 (β, γ) torus)")
        print("   " + "  ".join(f"d={x['d']:.1f}: {x['spread']:.1e}" for x in out["phase_dependence_shared"]))
    q2 = load(f"flagpair_q2_{T}.json")
    if q2:
        print("== Q = 2 in block (0, 2)  (CP¹ values: single "
              f"{q2['E_cp1']['single']:.4f}, A21 {q2['E_cp1']['A21']:.4f})")
        for r in q2["rows"]:
            print(f"κ₃ = {r['k3']:g}: E1 {r['E1']:.5f}  E_A21 {r['E_A21']:.5f}  binding {r['binding']:.4f}"
                  f" ({100 * r['binding_fraction']:.2f}%)  Q {r['Q1']:+.4f} {r['Q2']:+.4f}  conv {r['converged']}")
            for nm in ("single", "A21"):
                print(f"   {nm:6s} " + "  ".join(
                    f"{s['ev']:+.4f}[{s['frac_01']:.2f},{s['frac_02']:.2f},{s['frac_12']:.2f}]"
                    for s in r[f"spectrum_{nm}"]))
        out["q2"] = q2["rows"]
    for fn in sorted(glob.glob(os.path.join(DATA, f"flagpair_cons_{T}_k3*.json"))):
        c = json.load(open(fn))
        key = os.path.basename(fn)[len("flagpair_cons_"):-5]
        print(f"== {key}: shared-line pair at fixed separation")
        rows = []
        for d in sorted({r["d"] for r in c["rows"]}, reverse=True):
            rs = [r for r in c["rows"] if r["d"] == d]
            ok = [r for r in rs if r["converged"] and abs(r["Q"] + 2) < 0.02]
            best = min(ok, key=lambda r: r["E"]) if ok else None
            row = dict(d=d, E_sup_int=rs[0]["E_sup_int"], starts={r["start"]: dict(
                E_int=r["E_int"], converged=r["converged"], Q=r["Q"], lines=r["lines"]) for r in rs})
            if best:
                row.update(E_int=best["E_int"], best_start=best["start"], lines=best["lines"],
                           Q_3cycle=best["Q_3cycle"], parts=best["parts"])
            rows.append(row)
            print(f"d = {d:4.2f}  superposition {row['E_sup_int']:+9.4f}  relaxed "
                  + ("  ".join(f"{k}: {v['E_int']:+9.4f} (Q {v['Q']:+.3f}, conv {v['converged']},"
                               f" lines {', '.join(f'{x:.2f}' for x in v['lines'])})" for k, v in row["starts"].items())))
        out[f"cons_{key}"] = rows
    with open(os.path.join(DATA, "flagpair_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
