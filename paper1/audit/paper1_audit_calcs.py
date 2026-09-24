"""
Independent calculations for the Paper One audit (2026-09-24).

Every number quoted in PAPER1_REVIEW.md is produced here and written to
paper1_audit_results.json.  Only numpy/scipy are needed.

Sections
  A  constants
  B  Two-Range profile |∇Θ| = n/√(r²+a²): energy, flux ("charge"), Fourier transform
  C  electron g-2 from a charge form factor: the paper's coefficient versus the
     gauge-invariant modified-photon-propagator result (exact one loop)
  D  atomic spectroscopy: reproduction of the paper's bounds, proton-radius
     degeneracy, bound from electronic vs muonic hydrogen
  E  LEP: charge-radius operator = four-fermion contact interaction
  F  KT critical slowing numbers (dark-matter section) and 3D dynamic scaling
  G  decoherence arithmetic, Lorentz-violation arithmetic, rank-3 moduli
  H  the July back-reaction extrapolation (numbers quoted in the task brief)
  I  dimensionless invariants of the charged Hopfion (charged_mode results)

Literature values that are not derived here are marked "lit" and must be
checked against the sources; this environment has no web access.
"""

import json
import math
import os

import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------------------------------------------------------------------------
# A  constants (CODATA 2018)
# ---------------------------------------------------------------------------
alpha = 1 / 137.035999084
me_MeV = 0.51099895000
hbarc_MeVfm = 197.3269804
h_eVs = 4.135667696e-15
lamC_fm = hbarc_MeVfm / me_MeV                    # reduced Compton wavelength
a0_fm = lamC_fm / alpha                            # Bohr radius
OUT["A_constants"] = dict(alpha=alpha, me_MeV=me_MeV, hbarc_MeVfm=hbarc_MeVfm,
                          lambdabar_C_fm=lamC_fm, bohr_radius_fm=a0_fm)
a_paper = 2.7e-3      # fm, the paper's 1S-2S "bound"
a_q = 3.0e-3          # fm, the value used downstream (Λ = 65.8 GeV)

# ---------------------------------------------------------------------------
# B  Two-Range profile
# ---------------------------------------------------------------------------
n, a = 1.0, 1.0


def energy_two_range(R, a=1.0, n=1.0):
    """E(R) = ∫_{r<R} |∇Θ|² d³x for |∇Θ| = n/√(r²+a²)."""
    return 4 * np.pi * n * n * (R - a * np.arctan(R / a))


rows = []
for R in (1e1, 1e2, 1e3, 1e4):
    num, _ = quad(lambda r: 4 * np.pi * r * r * n * n / (r * r + a * a), 0, R, limit=400)
    rows.append(dict(R_over_a=R, E_closed_form=energy_two_range(R), E_quad=num,
                     E_over_R=energy_two_range(R) / R))
# unregularised n²/r²: contribution of 0 < r < ε is finite (= 4π n² ε)
eps = 1e-3
uv_part, _ = quad(lambda r: 4 * np.pi * r * r * n * n / (r * r), 0, eps)
OUT["B1_energy"] = dict(
    statement="E = ∫|∇Θ|² d³x with |∇Θ| = n/√(r²+a²) diverges linearly at r → ∞ (IR), "
              "not at the origin; the core changes E only by the finite amount -2π²n²a.",
    rows=rows, unregularised_contribution_r_below_1em3=uv_part,
    dE_da_large_R=-2 * np.pi ** 2 * n * n,
    derrick="For a pure gradient energy in 3D, E[Θ(x/λ)] = λ E[Θ]: no stationary size.")

# flux of ∇Θ through the sphere of radius R ("charge" Q = flux/2π)
flux = [dict(R_over_a=R, Q=4 * np.pi * R * R * n / np.sqrt(R * R + 1) / (2 * np.pi))
        for R in (1e1, 1e2, 1e3)]
OUT["B2_charge_flux"] = dict(statement="(1/2π)∮∇Θ·dS = 2nR²/√(R²+a²) → 2nR: not n, diverges.",
                             rows=flux)


# Fourier transform of ρ(r) = 1/(r²+a²):  ∫ e^{iq·x} ρ d³x = 2π² e^{-qa}/q
def ft_numeric(q, a=1.0):
    val, _ = quad(lambda r: r / (r * r + a * a), 0, np.inf, weight="sin", wvar=q)
    return 4 * np.pi / q * val


ft = [dict(qa=q, numeric=ft_numeric(q), closed_form=2 * np.pi ** 2 * np.exp(-q) / q)
      for q in (0.1, 0.5, 1.0, 2.0)]
OUT["B3_form_factor"] = dict(
    statement="FT[1/(r²+a²)] = 2π² e^{-qa}/q = 2π²(1/q - a + q a²/2 - ...): a 1/q pole "
              "(infinite charge) and odd powers of q; F = 1 - q²a²/6 does not follow.",
    rows=ft)

# ---------------------------------------------------------------------------
# C  electron g-2
# ---------------------------------------------------------------------------
I_paper, _ = quad(lambda x: x ** 4 * (1 - x) / (x * x * (1 - x) ** 2 + x * (1 - x) + x * x),
                  0, 1, epsabs=1e-15, epsrel=1e-13)
C_paper = alpha * I_paper / (12 * np.pi)          # δa = -C (m a)²


def aV_deriv(M2, k):
    """(π/α) ∂^k a_V/∂(M²)^k for a vector of mass² M² (units m_e = 1).

    a_V = (α/π) ∫_0^1 x²(1-x) / (x² + (1-x) M²) dx   (Schwinger at M = 0)
    Integrated in y = 1 - x on a log grid (robust for M² up to 1e12).
    """
    def f(logy):
        y = math.exp(logy)
        x = 1.0 - y
        den = x * x + y * M2
        return x * x * y * ((-1) ** k) * math.factorial(k) * y ** k / den ** (k + 1) * y

    val, _ = quad(f, math.log(1e-40), 0.0, limit=500, epsabs=0, epsrel=1e-11)
    return val


def delta_a_formfactor(ma, p):
    """Exact one-loop Δa/(α/π) for F(k²) = (1 - k²/Λ²)^{-p} at both photon vertices.

    ⟨r²⟩ = 6p/Λ² ≡ a².  F²/k² = 1/k² + Σ_{j=1}^{2p} Λ^{2(j-1)}/(Λ²-k²)^j, and a
    term 1/(k²-M²)^j contributes (1/(j-1)!) ∂^{j-1} a_V/∂(M²)^{j-1}.
    """
    L2 = 6.0 * p / ma ** 2
    tot = 0.0
    for j in range(1, 2 * p + 1):
        tot += ((-1) ** j) * L2 ** (j - 1) / math.factorial(j - 1) * aV_deriv(L2, j - 1)
    return tot


schwinger = aV_deriv(0.0, 0)      # should be 1/2
grow = []
for ma in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
    row = dict(m_a=ma, leading_universal=-ma * ma / 9.0)
    for p in (1, 2, 3):
        row[f"exact_p{p}"] = delta_a_formfactor(ma, p)
    row["paper"] = -(I_paper / 12.0) * ma * ma
    grow.append(row)
ma_paper = a_paper / lamC_fm
bound = lambda coef, da: lamC_fm * math.sqrt(da / coef)      # a such that coef (m a)² = da
OUT["C_g_minus_2"] = dict(
    I_paper=I_paper, coefficient_paper=C_paper, coefficient_universal=alpha / (9 * np.pi),
    ratio_universal_over_paper=(alpha / (9 * np.pi)) / C_paper,
    schwinger_check_aV0_over_alpha_over_pi=schwinger,
    note="Δa in units of α/π; 'exact_p' = one-loop with F=(1-k²/Λ²)^-p on both internal "
         "vertices (gauge invariant: only the photon propagator changes).",
    rows=grow,
    at_a_2p7em3_fm=dict(m_a=ma_paper, delta_a_paper=C_paper * ma_paper ** 2,
                       delta_a_universal=alpha / (9 * np.pi) * ma_paper ** 2),
    bounds_fm=dict(paper_coeff_1e_13=bound(C_paper, 1e-13),
                   universal_1e_13=bound(alpha / (9 * np.pi), 1e-13),
                   universal_1e_12=bound(alpha / (9 * np.pi), 1e-12)))

# ---------------------------------------------------------------------------
# D  atomic spectroscopy
# ---------------------------------------------------------------------------
a4m_eV = alpha ** 4 * me_MeV * 1e6


def shift_Hz(r_fm, n):
    """Finite-size shift of nS: (2/3) α⁴ m (m r)²/n³ (reduced mass ignored)."""
    return (2.0 / 3.0) * a4m_eV * (r_fm / lamC_fm) ** 2 / n ** 3 / h_eVs


lamb_bound = lamC_fm * math.sqrt(2e3 * h_eVs * 12 / a4m_eV)
s12_bound = lamC_fm * math.sqrt(10 * h_eVs * 12 / (7 * a4m_eV))
rp_muH, drp_muH = 0.84087, 0.00039            # lit: Antognini et al. 2013
eH = {"Bezginov2019_Lamb": (0.833, 0.010),    # lit
      "Grinin2020_1S3S": (0.8482, 0.0038)}    # lit
deg = {}
for k, (r, dr) in eH.items():
    d = r * r - rp_muH ** 2
    sd = math.hypot(2 * r * dr, 2 * rp_muH * drp_muH)
    ub = max(d, 0) + 2 * sd
    deg[k] = dict(r_p_eH=r, r2_diff_fm2=d, sigma_fm2=sd, upper_2sigma_fm2=ub,
                  a_upper_fm=math.sqrt(ub))
OUT["D_atomic"] = dict(
    reproduce_paper=dict(lamb_2kHz_bound_fm=lamb_bound, s1s2s_10Hz_bound_fm=s12_bound),
    proton_size_terms_Hz=dict(r_p=rp_muH, shift_1S=shift_Hz(rp_muH, 1), shift_2S=shift_Hz(rp_muH, 2),
                              one_s_two_s=shift_Hz(rp_muH, 1) - shift_Hz(rp_muH, 2),
                              one_s_two_s_from_drp_muH=(shift_Hz(rp_muH + drp_muH, 1) - shift_Hz(rp_muH + drp_muH, 2))
                              - (shift_Hz(rp_muH, 1) - shift_Hz(rp_muH, 2))),
    electron_size_term_at_a_2p7em3_fm_Hz=shift_Hz(a_paper, 1) - shift_Hz(a_paper, 2),
    degeneracy="electron and proton finite-size terms enter as (2/3)α⁴m³(r_p² + a²)/n³ δ_l0: "
               "identical n and l dependence; electronic hydrogen alone cannot separate them.",
    eH_vs_muH=deg)

# ---------------------------------------------------------------------------
# E  LEP contact-interaction equivalence
# ---------------------------------------------------------------------------
# Γ^μ = γ^μ (1 + q²⟨r²⟩/6): the O(q²) term cancels the photon pole →
# e²⟨r²⟩/6 (ēγe)(f̄γf).  Match to (4π/Λ²)(ēγe)(f̄γf):  ⟨r²⟩ = 6/(α Λ²).
lep = []
for Lam_TeV in (2.0, 10.0, 20.0):
    Lam_MeV = Lam_TeV * 1e6
    lep.append(dict(Lambda_TeV=Lam_TeV,
                    a_fm=math.sqrt(6 / alpha) / Lam_MeV * hbarc_MeVfm))
Lam_equiv_TeV = math.sqrt(6 / alpha) * hbarc_MeVfm / a_paper / 1e6
qa_Z = 91.1876e3 * a_paper / hbarc_MeVfm
OUT["E_LEP"] = dict(
    matching="⟨r²⟩ = 6/(α Λ²) for the VV contact model with g² = 4π",
    rows=lep, Lambda_equivalent_to_a_2p7em3_fm_TeV=Lam_equiv_TeV,
    lit="LEP combined ee→ff contact-interaction limits Λ ~ 10-20 TeV (model dependent); check.",
    z_pole_illustration=dict(q_a=qa_Z, q2a2_over_6=qa_Z ** 2 / 6))

# ---------------------------------------------------------------------------
# F  KT critical slowing
# ---------------------------------------------------------------------------
c_ms = 299792458.0
tau0 = a_paper * 1e-15 / c_ms
target = 1e15
x = math.log(target / tau0)
t_match = (math.pi / x) ** 2
tau_of = lambda t: tau0 * math.exp(math.pi / math.sqrt(t))
nu3, z_A, z_F = 0.6717, 2.0, 1.5              # 3D XY: ν (lit), model A / model F
OUT["F_KT"] = dict(
    tau0_s=tau0, orders=math.log10(target / tau0), t_match=t_match,
    log10_tau_at_t=dict(t_1e_4=math.log10(tau_of(1e-4)), t_1e_3=math.log10(tau_of(1e-3)),
                        t_1e_2=math.log10(tau_of(1e-2))),
    paper_claims_log10_tau=dict(t_1e_4=60, t_1e_2=4),
    dlntau_dlnt=-math.pi / (2 * math.sqrt(t_match)), paper_formula_1_over_2t=1 / (2 * t_match),
    fractional_t_change_for_factor_10=math.log(10) / (math.pi / (2 * math.sqrt(t_match))),
    xy3d_power_law=dict(nu=nu3, tau_ratio_modelA=t_match ** (-nu3 * z_A),
                        tau_ratio_modelF=t_match ** (-nu3 * z_F),
                        t_needed_for_1e41_modelA=10 ** (-41 / (nu3 * z_A))),
    dm_amplitude_check_km_s=math.sqrt(4 / 3 * math.pi * 6.674e-11 * 0.25 * 1.78266e-27 * 1e6)
    * 30 * 3.0857e19 / 1e3)

# ---------------------------------------------------------------------------
# G  decoherence, Lorentz violation, rank-3 moduli
# ---------------------------------------------------------------------------
m_kg, v, hbar = 9.1093837015e-31, 1e6, 1.054571817e-34
a_m = 3e-18
G_bare = m_kg ** 2 * v ** 2 * a_m * c_ms / hbar ** 2
suppr = (v / c_ms) ** 2 * (m_kg * v * a_m / hbar) ** 2
Lam_core_eV = hbarc_MeVfm / a_paper * 1e6
OUT["G_misc"] = dict(
    decoherence=dict(Gamma_bare=G_bare, suppression_without_rho_s=suppr,
                     Gamma_times_rho_s=G_bare * suppr,
                     rho_s_needed_for_1em9_to_1em8=[G_bare * suppr / 1e-9, G_bare * suppr / 1e-8]),
    lorentz=dict(Lambda_core_GeV=Lam_core_eV / 1e9,
                 delta_LV_atomic=(alpha ** 2 * me_MeV * 1e6 / Lam_core_eV) ** 2,
                 optical=(1.0 / Lam_core_eV) ** 2, hard_xray=(1e5 / Lam_core_eV) ** 2,
                 note="A Lorentz-invariant F□F term changes the propagator to "
                      "1/(k²(1 + c k²/Λ²)): the massless pole keeps ω = |k|; a second "
                      "pole appears at k² = -Λ²/c. It cannot give ω² = k²(1 + c k²/Λ²)."),
    rank3_moduli={r: dict(dim_SU=r * r - 1, dim_stab=((r - 2) ** 2 - 1 + 1) if r > 2 else 0,
                          orientation=4 * r - 5) for r in (2, 3, 4, 5)},
    rank3_paper_overcount=415 / 8)

# ---------------------------------------------------------------------------
# H  July back-reaction extrapolation (numbers from the task brief)
# ---------------------------------------------------------------------------
b1, b2 = 0.03866, 0.04211          # 60×120, 120×240 (h ratio 2)
b_fixed, b_lin = 0.041974, 0.042581
OUT["H_july"] = dict(
    inputs=dict(beta_60x120=b1, beta_120x240=b2, beta_fixed_background=b_fixed,
                linear_threshold_fixed_background=b_lin),
    extrapolated={f"p={p}": b2 + (b2 - b1) / (2 ** p - 1) for p in (1, 2, 3, 4)},
    note="Two grids cannot fix the order p; the brief's 0.0433-0.0434 corresponds to p ≈ 2.")

# ---------------------------------------------------------------------------
# I  charged Hopfion invariants (from ../charged_mode/data)
# ---------------------------------------------------------------------------
cm = os.path.join(HERE, "..", "charged_mode", "data")
try:
    s = json.load(open(os.path.join(cm, "check_summary.json")))
    lim = json.load(open(os.path.join(cm, "check_limits_ne96.json")))
    E = s["extrapolated"]["full"]["E"]["value"]
    rms = s["extrapolated"]["full"]["rms_charge"]["value"]
    mu = abs(s["extrapolated"]["full"]["mu_volume"]["value"])
    Q = 0.3
    mu_N0 = abs([r for r in lim["runs"] if r["N"] == 0.0 and r["e"] == 0.3][0]["mu_volume"])
    OUT["I_charged_hopfion"] = dict(
        E=E, rms=rms, mu=mu, mu_at_N0=mu_N0, M_times_rms=E * rms,
        rms_if_M_is_me_fm=E * rms * lamC_fm, in_bohr_radii=E * rms * lamC_fm / a0_fm,
        g_like_J_half=2 * E * mu / (Q * 0.5), g_like_J_one=2 * E * mu / (Q * 1.0),
        scaling="E = (f/g) Ẽ(μ/(fg), e/g), R = R̃/(fg)  ⇒  M·R ∝ 1/g² at fixed μ/(fg), e/g")
except FileNotFoundError:
    OUT["I_charged_hopfion"] = "charged_mode data not found"

with open(os.path.join(HERE, "paper1_audit_results.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False, default=float)
print(json.dumps(OUT, indent=1, ensure_ascii=False, default=float))
