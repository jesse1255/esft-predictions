"""
Can soliton rest energies produce the lepton mass hierarchy?  (round 6)

Hypotheses tested against m_μ/m_e = 206.768, m_τ/m_e = 3477.2 (PDG values quoted
from memory; not re-checked in this environment):
  H1  the three links (or lines with unequal masses μ_ab) are three generations:
      E₁ depends on μ only weakly (local exponent from μ = 0.9, 1.0, 1.1);
  H2  generations are Hopf charges Q = 1, 2, 3…: E(Q) grows like Q^{3/4}
      (Vakulenko–Kapitanski); E(2)/E(1) is computed.
Both need absurd inputs (μ ratios ~10¹³, Q ~ 10³–10⁵): rest energies scale as powers,
the hierarchy needs exponentials.

Usage: python hierarchy_check.py   → data/hierarchy_check.json
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
R = {"mu/e": 206.768, "tau/e": 3477.2}


def main():
    E = {mu: json.load(open(os.path.join(DATA, f"pair_single_ne24x32_a4_mu{mu:g}.json")))["full"]["E"]
         for mu in (0.9, 1.0, 1.1)}
    p = math.log(E[1.1] / E[0.9]) / math.log(1.1 / 0.9)
    q2 = json.load(open(os.path.join(DATA, "pair_q2_ne24x32_a4_mu1.json")))
    r2 = q2["A21"]["E"] / q2["E1"]
    out = dict(E1_vs_mu={str(k): v for k, v in E.items()}, exponent_dlnE_dlnmu=p,
               H1_mu_ratio_needed={k: v ** (1 / p) for k, v in R.items()},
               E_Q2_over_E_Q1=r2, H2_Q_needed_at_Q34={k: v ** (4 / 3) for k, v in R.items()},
               lepton_ratios=R)
    print(json.dumps(out, indent=1))
    with open(os.path.join(DATA, "hierarchy_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
