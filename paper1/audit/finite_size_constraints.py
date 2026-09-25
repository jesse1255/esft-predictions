"""
Finite-size constraints on a lepton charge radius, redone model by model
(second-round audit, 2026-09-25).

The original Paper One (ESFT_FoP_SPRINGER.tex, eqs. 7–16) treats the electron as
having an F₁ form factor F₁ = 1 − q²a²/6 (static q) and bounds a from g − 2,
the 2S–2P Lamb shift and 1S–2S.  Here every bound is tied to an explicit
physical model:

  M-e  only the electron has a charge radius a (muon, proton as usual);
  M-u  all charged leptons share the same radius a;
  and in both, the structure is a pure F₁ form factor (no extra magnetic
  structure).  A model with an intrinsic magnetic form factor has a tree-level
  δa and none of the g − 2 numbers below apply.

Covariant form factor (metric +−−−):  F₁(q²) = 1 + q²⟨r²⟩/6,  ⟨r²⟩ = a².

Literature inputs are marked LIT; several were only available through search
summaries in this session and must be checked against the original papers.

Usage: python finite_size_constraints.py   → finite_size_constraints.json
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ALPHA = 1 / 137.035999084            # CODATA 2018
HBARC = 197.3269804                   # MeV fm
ME = 0.51099895000                    # MeV
MMU = 105.6583755
MP = 938.27208816
LAMBDA_E = HBARC / ME                 # reduced Compton wavelength of the electron, fm
HZ_PER_MEV = 2.417989242e20           # 1 MeV / h in Hz

out = {}

# ---------------------------------------------------------------------------
# 1  g − 2: Schwinger topology with F₁(k²) at both ends of the internal photon
# ---------------------------------------------------------------------------
# F₁(k²)²/k² = 1/k² + ⟨r²⟩/3 + O(k²).  The constant is a contact term; it equals
# the k² ≪ M² limit of a vector of mass M with coupling g_V if g_V²/M² = −e²⟨r²⟩/3.
# A vector gives a_V = (g_V²/8π²) ∫₀¹ 2x²(1−x) dx / [x² + (1−x)M²/m²] → g_V² m²/(12π² M²).
# Hence δa = −e² m² ⟨r²⟩/(36π²) = −(α/9π) m² ⟨r²⟩.
def vector_a(M_over_m, n=200000):
    """Exact one-loop a_V/(g_V²/8π²) by the midpoint rule (checks the heavy limit)."""
    s = 0.0
    r2 = M_over_m ** 2
    for i in range(n):
        x = (i + 0.5) / n
        s += 2 * x * x * (1 - x) / (x * x + (1 - x) * r2)
    return s / n


heavy = {str(M): vector_a(M) * M * M for M in (30.0, 100.0, 300.0)}   # → 2/3
coef_new = ALPHA / (9 * math.pi)
coef_old = ALPHA * 0.04675225418 / (12 * math.pi)                   # original eq. 10
out["g2"] = dict(
    heavy_vector_limit_check=heavy, expected="2/3 = 0.6667",
    coefficient_alpha_over_9pi=coef_new, coefficient_original_eq10=coef_old,
    ratio_new_over_original=coef_new / coef_old,
)
# data (LIT, via search summary of arXiv:2504.21179; to be checked):
#   Δa_e = a_exp − a_SM = +4.8(3.0)e-13 with α(Rb),  −8.7(3.6)e-13 with α(Cs)
g2_rows = []
for lab, (d, s) in dict(Rb=(4.8e-13, 3.0e-13), Cs=(-8.7e-13, 3.6e-13)).items():
    lo = d - 1.96 * s            # the model shift is δa = −coef (m a)² ≤ 0
    amax = LAMBDA_E * math.sqrt(max(-lo, 0.0) / coef_new)
    row = dict(alpha_from=lab, delta_ae=d, sigma=s, lower_95=lo, a_max_fm=amax)
    if d < 0:
        row["a_central_fm"] = LAMBDA_E * math.sqrt(-d / coef_new)
    g2_rows.append(row)
out["g2"]["bounds"] = g2_rows

# ---------------------------------------------------------------------------
# 2  LEP: the charge radius is a vector–vector contact interaction
# ---------------------------------------------------------------------------
# e⁺e⁻ → μ⁺μ⁻ at q² = s > 0:  e² F_e(s) F_μ(s)/s = e²/s + e²(⟨r_e²⟩ + ⟨r_μ²⟩)/6 + O(s).
# The extra term has the same sign as the photon term (F₁(s) > 1 for timelike q²):
# constructive interference.  Matching to (g²/Λ²)(ēγe)(μ̄γμ), g²/4π = 1:
#   M-e:  ⟨r_e²⟩ = 6/(αΛ²);   M-u:  2a² = 6/(αΛ²).
# LIT (search summary; the source table was not opened): e⁺e⁻ → μ⁺μ⁻, VV model, 95% CL,
#   Λ⁺ (constructive) = 21.6 TeV, Λ⁻ (destructive) = 13.7 TeV.
lep = []
for lab, Lam in (("VV constructive (applies)", 21.6e6), ("VV destructive (wrong sign)", 13.7e6)):
    a_e = math.sqrt(6 / ALPHA) * HBARC / Lam
    lep.append(dict(model=lab, Lambda_TeV=Lam / 1e6, a_max_Me_fm=a_e, a_max_Mu_fm=a_e / math.sqrt(2),
                    sqrt_s_a_at_200GeV=200e3 * a_e / HBARC))
out["LEP"] = lep

# ---------------------------------------------------------------------------
# 3  Atomic spectroscopy: the electron radius is degenerate with r_p
# ---------------------------------------------------------------------------
# δE_nS = (2/3)(Zα)⁴ m_r³ (r_p² + r_e²)/n³ at leading order (Z = 1, ℓ = 0).
def c_fs_hz_per_fm2(n, m_lep):
    mr = m_lep * MP / (m_lep + MP)
    return (2 / 3) * ALPHA ** 4 * mr ** 3 / HBARC ** 2 / n ** 3 * HZ_PER_MEV


c1 = c_fs_hz_per_fm2(1, ME)
c2 = c_fs_hz_per_fm2(2, ME)
out["atomic_coefficients"] = dict(C_1S_Hz_per_fm2=c1, C_2S_Hz_per_fm2=c2,
                                  C_1S2S_Hz_per_fm2=c1 - c2)
# original eq. 15–16: |δf(1S–2S)| = (7/12) α⁴ m (m a)²/h ≤ 10 Hz
a_orig = LAMBDA_E * math.sqrt(10.0 / ((7 / 12) * ALPHA ** 4 * ME * HZ_PER_MEV))
out["original_1S2S_bound_fm"] = a_orig
out["proton_term_1S2S_Hz"] = (c1 - c2) * 0.84060 ** 2
# μH (LIT): r_p = 0.84060(39) fm  (updated μH theory; CODATA 2022 quotes 0.84060 from μH)
rmu, drmu = 0.84060, 0.00039
s_mu, ds_mu = rmu ** 2, 2 * rmu * drmu
# electronic H determinations (LIT; each already marginalises R∞ and includes theory):
H = [
    ("CODATA 2022, electronic H+D only (value/uncertainty from a search summary: "
     "0.8529, relative 5.1e-3 and 2.8σ from μH)", 0.8529, 0.0044),
    ("Beyer 2017, 2S–4P + 1S–2S", 0.8335, 0.0095),
    ("Fleurbaey 2018, 1S–3S + 1S–2S", 0.877, 0.013),
    ("Bezginov 2019, 2S–2P", 0.833, 0.010),
    ("Grinin 2020, 1S–3S + 1S–2S", 0.8482, 0.0038),
    ("2S–nS (n = 8, 9, 10) + 1S–2S, arXiv:2604.26401", 0.8433, 0.0031),
]
rows = []
for lab, r, dr in H:
    s_h, ds_h = r * r, 2 * r * dr
    d = s_h - s_mu
    sd = math.hypot(ds_h, ds_mu)
    up = max(d, 0.0) + 1.645 * sd          # one-sided 95 %, physical boundary r_e² ≥ 0
    rows.append(dict(input=lab, r_H=r, dr_H=dr, re2_fm2=d, sigma_fm2=sd,
                     nsigma=d / sd, re_max_95_fm=math.sqrt(up)))
out["atomic_Me"] = dict(
    model="M-e (only the electron has a radius; muon pointlike)",
    relation="r_e² = s_H − s_μ,  s_H = (r_p² + r_e²) from electronic H,  s_μ = r_p² from μH",
    muH=dict(r_p=rmu, dr_p=drmu), rows=rows,
    note="In M-u (r_e = r_μ) the difference is r_e² − r_μ² = 0: no bound at all.")
with open(os.path.join(HERE, "finite_size_constraints.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
