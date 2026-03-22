"""
ESFT Prediction Verification Script
====================================
Reproduces all 24 quantitative predictions from:
  - Paper I:   Topological soliton coherence scale (Zenodo 10.5281/zenodo.19154442)
  - Paper II:  Emergent gauge symmetry (Zenodo 10.5281/zenodo.19159255)
  - Paper III: Electron mass from topological string tension (Zenodo 10.5281/zenodo.19159513)

Requirements: Python 3.x (standard library only)
Usage: python verify_predictions.py

Author: Jesse C. P. Ting
Date: March 21, 2026
"""

import math

# ============================================================
# Fundamental constants
# ============================================================
hbar_c = 197.3269804   # MeV·fm (CODATA 2018)
qoppa_fm = 0.003       # fm (Paper I, fitted to 7 domains)
sqrt_sigma = 459.0     # MeV (Paper II, topological derivation)
eta = 0.25             # BKT exact (Nelson & Kosterlitz 1977)
Nc = 3                 # color SU(3)
Nw = 2                 # weak isospin SU(2)

# Derived scales
Lambda_core = hbar_c / qoppa_fm   # MeV
sigma = sqrt_sigma**2              # MeV^2
epsilon = sqrt_sigma / Lambda_core # dimensionless IR/UV ratio

# Intermediate quantities
M_Q = (sqrt_sigma / (2*math.pi)) * math.log(1/epsilon)
alpha_s = 18*math.pi / (11*math.log(1/epsilon**2))

# ============================================================
# Experimental values (PDG 2024 + lattice QCD)
# ============================================================
exp = {
    'alpha_s':   0.518,      # non-perturbative (lattice estimate at core scale)
    'm_b':       4.18,       # GeV (PDG, MSbar at self-energy scale)
    'm_e':       0.51100,    # MeV (PDG)
    'm_omega':   782.66,     # MeV (PDG)
    'qq13':      250.0,      # MeV (FLAG 2021, at 2 GeV)
    'F':         1.79,       # numerical MC simulation (this work)
    'ms_mq':     27.07,      # PDG ratio
    'NDelta':    294.0,      # MeV (PDG: m_Delta - m_N)
    'mZ_mW':     1.13461,    # PDG: 91.1876/80.3692
    'inv_alpha': 137.036,    # CODATA
    'm_H':       125.25,     # GeV (PDG)
    'm_c':       1.27,       # GeV (PDG, MSbar)
    'm_t':       172.69,     # GeV (PDG)
    'm_pi':      138.0,      # MeV (PDG average pi+/pi0)
    'm_s':       93.4,       # MeV (PDG, MSbar at 2 GeV)
    'm_q':       3.45,       # MeV (PDG, (m_u+m_d)/2 at 2 GeV)
    'v':         246.22,     # GeV (Higgs VEV)
    'sin2tw':    0.23122,    # PDG
    'm_tau':     1776.9,     # MeV (PDG)
    'y_t':       0.9919,     # PDG: m_t*sqrt(2)/v
    'f_pi':      92.07,      # MeV (PDG)
    'M_Q':       313.0,      # MeV (constituent quark mass, phenomenological)
    'm_mu':      105.66,     # MeV (PDG)
    'm_N':       938.0,      # MeV (PDG)
}

# ============================================================
# ESFT predictions
# ============================================================
pred = {}

# Hadronic sector
pred['alpha_s']  = ('18π/(11·ln(1/ε²))',           alpha_s)
pred['m_pi']     = ('√σ · ε^η',                     sqrt_sigma * epsilon**eta)
pred['m_omega']  = ('2M_Q(1+α_s/2π)',               2*M_Q*(1+alpha_s/(2*math.pi)))
pred['qq13']     = ('(√σ³/2π)^{1/3}',               (sqrt_sigma**3/(2*math.pi))**(1/3))
pred['m_q']      = ('σ/Λ_core',                      sigma/Lambda_core)
pred['m_s']      = ('Λε^{4/3}  [2-2/Nc]',           Lambda_core*epsilon**(4/3))
pred['ms_mq']    = ('ε^{-2/Nc}',                     epsilon**(-2/Nc))
pred['M_Q']      = ('(√σ/2π)ln(1/ε)',               M_Q)
pred['m_N']      = ('3·M_Q',                          3*M_Q)
pred['NDelta']   = ('σ/(2M_Q)',                       sigma/(2*M_Q))
pred['m_c']      = ('Λε^{4/5}  [2Nw/(Nc+Nw)]',     Lambda_core*epsilon**(4/5)/1000)
pred['m_b']      = ('Λε^{5/9}  [(Nc+Nw)/Nc²]',     Lambda_core*epsilon**(5/9)/1000)

# Leptonic sector
pred['m_e']      = ('σ/(2πΛ)',                        sigma/(2*math.pi*Lambda_core))
pred['m_mu']     = ('Λε^{4/3}',                      Lambda_core*epsilon**(4/3))
pred['m_tau']    = ('√σ · ε^{-η}',                   sqrt_sigma*epsilon**(-eta))
pred['m_t']      = ('Λε^{-1/5} [−1/(Nc+Nw)]',      Lambda_core*epsilon**(-1/(Nc+Nw))/1000)
pred['y_t']      = ('√2 · ε^{1/20}',                 math.sqrt(2)*epsilon**(1/20))

# Electroweak sector
pred['inv_alpha']= ('64/α_s + 16.4 (SM RG)',        64/alpha_s + 16.4)
pred['m_H']      = ('Λε^{-η/2}',                     Lambda_core*epsilon**(-eta/2)/1000)
pred['mZ_mW']    = ('2/√3',                           2/math.sqrt(3))
pred['sin2tw']   = ('1/(1+dim SU(2)) = 1/4',         0.25)
pred['v']        = ('Λε^{-1/4}',                      Lambda_core*epsilon**(-0.25)/1000)
pred['f_pi']     = ('√σ/√(2πNc)',                     sqrt_sigma/math.sqrt(2*math.pi*Nc))

# Three-field BKT
a = 0.893
pred['F']        = ('(a−1+√(a²+10a+1))/2a',         (a-1+math.sqrt(a**2+10*a+1))/(2*a))

# ============================================================
# Output
# ============================================================
print("=" * 75)
print("ESFT PREDICTION VERIFICATION")
print("=" * 75)
print(f"Qoppa  = {qoppa_fm} fm")
print(f"Λ_core = {Lambda_core:.2f} MeV = {Lambda_core/1000:.4f} GeV")
print(f"√σ     = {sqrt_sigma} MeV")
print(f"ε      = {epsilon:.6f}")
print(f"η      = {eta}")
print(f"M_Q    = {M_Q:.2f} MeV")
print(f"α_s(ϙ) = {alpha_s:.4f}")
print()

results = []
for key in pred:
    formula, esft_val = pred[key]
    exp_val = exp[key]
    dev = abs(esft_val - exp_val) / abs(exp_val) * 100
    results.append((key, formula, esft_val, exp_val, dev))

results.sort(key=lambda x: x[4])

print(f"{'#':>2} {'Grade':>5} {'Dev':>7} {'Name':>12}  {'ESFT':>12}  {'Exp':>12}  Formula")
print("-" * 95)

u1 = u5 = u15 = u25 = 0
for i, (key, formula, esft_val, exp_val, dev) in enumerate(results):
    grade = '***' if dev < 1 else '** ' if dev < 5 else '*  ' if dev < 15 else '   '
    if dev < 1: u1 += 1
    elif dev < 5: u5 += 1
    elif dev < 15: u15 += 1
    else: u25 += 1
    print(f"{i+1:2d} {grade} {dev:6.2f}% {key:>12}  {esft_val:12.4f}  {exp_val:12.4f}  {formula}")

print()
print(f"< 1%:   {u1}")
print(f"1-5%:   {u5}")
print(f"5-15%:  {u15}")
print(f"15-25%: {u25}")
print(f"Total:  {len(results)}")
print()
print("Inputs: Qoppa = 0.003 fm (Paper I), √σ = 459 MeV (Paper II), η = 1/4 (BKT)")
print("Free parameters beyond Paper I: ZERO")
print()

# Cross-checks
print("CROSS-CHECKS:")
mpi = sqrt_sigma * epsilon**eta
mtau = sqrt_sigma * epsilon**(-eta)
print(f"  m_π × m_τ = {mpi*mtau:.0f} vs σ = {sigma:.0f} (internal: {abs(mpi*mtau-sigma)/sigma*100:.6f}%)")
me = sigma/(2*math.pi*Lambda_core)
mq = sigma/Lambda_core
print(f"  m_e = m_q/(2π): {me:.5f} = {mq/(2*math.pi):.5f} (match: {abs(me-mq/(2*math.pi))/me*100:.8f}%)")
yt1 = Lambda_core*epsilon**(-1/5)/1000 * math.sqrt(2) / (Lambda_core*epsilon**(-0.25)/1000)
yt2 = math.sqrt(2)*epsilon**(1/20)
print(f"  y_t consistency: {yt1:.6f} = {yt2:.6f} (match: {abs(yt1-yt2)/yt1*100:.8f}%)")
print(f"  -1/(Nc+Nw) + η = {-1/(Nc+Nw)+eta:.4f} = 1/20 = {1/20:.4f}")
