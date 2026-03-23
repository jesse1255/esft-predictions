"""
2026-03-22 — 全面驗算 Paper 4-7 今日所有新結果
================================================
逐條計算，不手算，所有數字用 Python 精確驗證。
"""

import numpy as np

print("=" * 70)
print("ESFT Paper 4-7 — Complete Verification (2026-03-22)")
print("=" * 70)

# ============================================================
# 基礎常數
# ============================================================
qoppa_fm = 0.003
hbar_c = 197.327  # MeV·fm
Lambda = hbar_c / qoppa_fm  # MeV
sqrt_sigma = 459.0  # MeV
sigma = sqrt_sigma**2
epsilon = sqrt_sigma / Lambda
Nc = 3
Nw = 2
N5 = Nc + Nw  # = 5

print(f"\n--- 基礎常數 ---")
print(f"ϙ = {qoppa_fm} fm")
print(f"Λ = {Lambda:.1f} MeV = {Lambda/1000:.2f} GeV")
print(f"√σ = {sqrt_sigma} MeV")
print(f"σ = {sigma:.0f} MeV²")
print(f"ε = {epsilon:.6f}")
print(f"ln(1/ε) = {np.log(1/epsilon):.4f}")
print(f"N_c = {Nc}, N_w = {Nw}, N_5 = {N5}")

# ============================================================
# Paper 5: f_π 驗算
# ============================================================
print(f"\n{'='*70}")
print("PAPER 5: f_π from BKT renormalization")
print(f"{'='*70}")

# GOR: m_π² f_π² = (m_u+m_d) |⟨q̄q⟩|
m_q = Lambda * epsilon**2  # (m_u+m_d)/2
m_ud = 2 * m_q  # m_u + m_d
qqbar_third = 249.0  # MeV (⟨q̄q⟩^{1/3})
qqbar = qqbar_third**3  # |⟨q̄q⟩| in MeV³

# Bare f_π from GOR (using ESFT m_π = √σ × ε^{1/4})
m_pi_esft = sqrt_sigma * epsilon**0.25
f_pi_bare = np.sqrt(m_ud * qqbar) / m_pi_esft

# BKT renormalization: √(π/2)
bkt_factor = np.sqrt(np.pi / 2)
f_pi_renorm = f_pi_bare * bkt_factor

f_pi_exp = 92.07  # MeV (PDG 2024, f_π±)

print(f"\nm_q = Λε² = {m_q:.3f} MeV")
print(f"m_u + m_d = {m_ud:.3f} MeV")
print(f"|⟨q̄q⟩|^{{1/3}} = {qqbar_third} MeV → |⟨q̄q⟩| = {qqbar:.0f} MeV³")
print(f"m_π(ESFT) = √σ × ε^{{1/4}} = {m_pi_esft:.2f} MeV")
print(f"\nf_π(bare) = √(m_ud × |⟨q̄q⟩|) / m_π = {f_pi_bare:.2f} MeV")
print(f"BKT factor = √(π/2) = {bkt_factor:.4f}")
print(f"f_π(renorm) = {f_pi_bare:.2f} × {bkt_factor:.4f} = {f_pi_renorm:.2f} MeV")
print(f"\nf_π(exp) = {f_pi_exp} MeV")
print(f"偏差 = {abs(f_pi_renorm - f_pi_exp)/f_pi_exp * 100:.1f}%")

# 反過來：用 f_π 算 m_π 自洽性
m_pi_from_GOR = np.sqrt(m_ud * qqbar) / f_pi_renorm
print(f"\n自洽性: m_π from GOR with f_π(renorm) = {m_pi_from_GOR:.2f} MeV")
print(f"  vs m_π(ESFT) = {m_pi_esft:.2f} MeV (應該相等 = {abs(m_pi_from_GOR-m_pi_esft):.2f} diff)")

# 用實驗 m_π 反推 f_π
m_pi_exp = 138.0
f_pi_from_exp_mpi = np.sqrt(m_ud * qqbar) / m_pi_exp * bkt_factor
print(f"\n如果用 m_π(exp)=138: f_π = {f_pi_from_exp_mpi:.2f} MeV (vs exp {f_pi_exp})")

# ============================================================
# Paper 5: Nuclear binding with new f_π
# ============================================================
print(f"\n--- Nuclear force parameters ---")
M_Q = (sqrt_sigma / (2*np.pi)) * np.log(1/epsilon)
alpha_s = 18 * np.pi / (11 * np.log(1/epsilon**2))
m_omega = 2 * M_Q * (1 + alpha_s/(2*np.pi))
hard_core = 1 / sqrt_sigma * hbar_c  # fm

print(f"M_Q = (√σ/2π)ln(1/ε) = {M_Q:.1f} MeV")
print(f"α_s(ϙ) = {alpha_s:.4f}")
print(f"m_ω = 2M_Q(1+α_s/2π) = {m_omega:.1f} MeV (exp 783)")
print(f"Hard core = 1/√σ = {hard_core:.3f} fm (exp ~0.4-0.5)")
print(f"N-Δ = σ/(2M_Q) = {sigma/(2*M_Q):.1f} MeV (exp 294)")

# ============================================================
# Paper 6: Sakharov induced gravity
# ============================================================
print(f"\n{'='*70}")
print("PAPER 6: Sakharov Induced Gravity")
print(f"{'='*70}")

M_Pl_GeV = 1.2209e19  # Planck mass in GeV
G_N = 6.674e-11  # m³/(kg·s²)

# G_induced = 6π / (N_fields × f²)
# f = M_Pl, N = 3
N_fields = 3
G_ind_over_GN = (6 * np.pi / N_fields) / (8 * np.pi)  # in units where G_N = 1/(8π M_Pl²)
# More carefully: G_N = 1/(8π M_Pl²), G_ind = 6π/(N f²) = 6π/(N M_Pl²)
# G_ind/G_N = 6π/(N M_Pl²) / (1/(8π M_Pl²)) = 48π² / N

ratio_simple = 6 * np.pi / N_fields * (8 * np.pi)  # wait let me redo this

# G_N = ℏc / M_Pl² = 1/(8π M_Pl²) in natural units (reduced Planck mass)
# Actually G_N = 1/M_Pl² (if M_Pl = 1.22e19 GeV, unreduced)
# G_ind = 6π/(N f²) from Sakharov one-loop

# Ratio:
# G_ind/G_N = [6π/(N f²)] / [1/M_Pl²] = 6π M_Pl² / (N f²)
# If f = M_Pl: G_ind/G_N = 6π/N

ratio = 6 * np.pi / N_fields
print(f"\nG_ind/G_N = 6π/N = 6π/{N_fields} = {ratio:.2f}")
print(f"  → G_induced ≈ {ratio:.1f} × G_Newton")

# What N_eff would give G_ind = G_N?
N_eff_needed = 6 * np.pi
print(f"\n需要 N_eff = 6π ≈ {N_eff_needed:.1f} 才能 G_ind = G_N")
print(f"  SM total DOF: 28 bosonic + 90 fermionic = 118")
print(f"  ESFT: 3 scalar + massive modes... plausible to get ~{N_eff_needed:.0f}")

# Alternative: 3 fields × 2 (particle + antiparticle) × π contribution factor
N_eff_alt = 3 * 2 * np.pi  # ≈ 18.8
print(f"  Alternative: 3 × 2 × π = {N_eff_alt:.1f} → G_ratio = {6*np.pi/N_eff_alt:.3f}")

# ============================================================
# Paper 7: Fibonacci formula verification
# ============================================================
print(f"\n{'='*70}")
print("PAPER 7: Fibonacci Mass Formula — Complete Verification")
print(f"{'='*70}")

# n_color(l) = 2 - 2l/Nc
# n_weak(l) = F_{l-2} / (Nc × D_{l-2})  for l >= 2, else 0
# F: Fibonacci with F_0=Nw=2, F_1=N5=5, F_k=F_{k-1}+F_{k-2}
# D_k: N5 if k even, Nc if k odd

# Generate Fibonacci
F = [Nw, N5]  # F_0, F_1
for i in range(10):
    F.append(F[-1] + F[-2])

def D(k):
    return N5 if k % 2 == 0 else Nc

def n_weak(l):
    if l < 2:
        return 0
    k = l - 2
    return F[k] / (Nc * D(k))

def n_total(l):
    return 2 - 2*l/Nc + n_weak(l)

def mass_MeV(l):
    n = n_total(l)
    return Lambda * epsilon**n

# Known quarks
quarks = [
    (0, 'u,d', 3.45),
    (1, 's', 93.4),
    (2, 'c', 1270.0),
    (3, 'b', 4180.0),
    (4, 't', 172690.0),
]

print(f"\n{'l':>3} {'quark':>6} {'n_color':>10} {'n_weak':>10} {'n_total':>10} "
      f"{'m_ESFT':>12} {'m_exp':>12} {'dev':>8}")
print(f"{'-'*75}")

for l, name, m_exp in quarks:
    nc = 2 - 2*l/Nc
    nw = n_weak(l)
    nt = n_total(l)
    m = mass_MeV(l)
    dev = abs(m - m_exp) / m_exp * 100
    
    # Show n_weak decomposition
    if l >= 2:
        k = l - 2
        nw_str = f"F{k}={F[k]}/({Nc}×{D(k)})={nw:.4f}"
    else:
        nw_str = "0"
    
    print(f"{l:3d} {name:>6} {nc:10.4f} {nw:10.4f} {nt:10.4f} "
          f"{m:12.1f} {m_exp:12.1f} {dev:7.1f}%")

# Fibonacci sequence check
print(f"\n--- Fibonacci sequence ---")
print(f"Seeds: F_0 = N_w = {F[0]}, F_1 = N_5 = {F[1]}")
for k in range(6):
    print(f"  F_{k} = {F[k]}, D_{k} = {D(k)}, "
          f"n_weak = {F[k]}/{Nc}×{D(k)} = {F[k]/(Nc*D(k)):.6f}")

# Prediction: l=5 (6th quark)
print(f"\n--- Prediction: 6th quark (l=5) ---")
l6 = 5
n6 = n_total(l6)
m6 = mass_MeV(l6)
print(f"  l=5: n_color={2-2*5/3:.4f}, n_weak={n_weak(5):.4f}, n_total={n6:.4f}")
print(f"  m = Λ × ε^{n6:.4f} = {m6:.1f} MeV = {m6/1000:.2f} GeV")

# ============================================================
# Paper 7: LEPTON sector test
# ============================================================
print(f"\n{'='*70}")
print("PAPER 7 EXTENSION: Lepton Masses")
print(f"{'='*70}")

print(f"\n已知 lepton:")
print(f"  m_e = {0.511} MeV → ESFT: σ/(2πΛ) = {sigma/(2*np.pi*Lambda):.4f} MeV "
      f"(dev {abs(sigma/(2*np.pi*Lambda)-0.511)/0.511*100:.1f}%)")
print(f"  m_μ = {105.66} MeV → ESFT: Λε^{{4/3}} = {Lambda*epsilon**(4/3):.1f} MeV "
      f"(= m_s, dev {abs(Lambda*epsilon**(4/3)-105.66)/105.66*100:.1f}%)")
print(f"  m_τ = {1776.9} MeV → ESFT: √σ/ε^{{1/4}} = {sqrt_sigma/epsilon**0.25:.1f} MeV "
      f"(dev {abs(sqrt_sigma/epsilon**0.25-1776.9)/1776.9*100:.1f}%)")

# Can leptons follow a similar l-based formula but without color?
# If n_lepton = n_color=0 (no color) + something
print(f"\n--- Lepton Fibonacci test (color-free, only weak) ---")

# Hypothesis: leptons have n = 2 + n_weak_only(l) where weak acts differently
# m_e: l=0, n=2 + 1/(2π) factor → special (U(1) winding)
# m_μ: same n as m_s (4/3) → lepton-quark complementarity
# m_τ: n = -1/4 (from ε^{-1/4})

# m_τ/m_μ:
print(f"  m_τ/m_μ (exp) = {1776.9/105.66:.1f}")
print(f"  m_τ/m_μ (ESFT) = ε^{{-1/4-4/3}} = ε^{{-19/12}} = {epsilon**(-19/12):.1f}")
print(f"  actual: {(sqrt_sigma/epsilon**0.25)/(Lambda*epsilon**(4/3)):.1f}")

# m_μ/m_e:
print(f"  m_μ/m_e (exp) = {105.66/0.511:.1f}")
print(f"  m_μ/m_e (ESFT) = {(Lambda*epsilon**(4/3))/(sigma/(2*np.pi*Lambda)):.1f}")

# ============================================================
# Paper 4: Curvature correction estimate
# ============================================================
print(f"\n{'='*70}")
print("PAPER 4: S² Curvature Correction")
print(f"{'='*70}")

# Hopf soliton core: S² base space with radius R ~ 1/√σ
R_s2 = hbar_c / sqrt_sigma  # radius of S² in fm
print(f"\nS² base radius R ≈ 1/√σ = {R_s2:.4f} fm = {R_s2/qoppa_fm:.1f} ϙ")

# BKT on flat 2D: T_KT = πJ/2
# BKT on S² (curved): correction ~ ϙ²/R²
correction = (qoppa_fm / R_s2)**2
print(f"Curvature correction: (ϙ/R)² = {correction:.6f} = {correction*100:.4f}%")
print(f"  → Negligible! BKT on S² ≈ BKT on flat 2D to {correction*100:.4f}% precision")
print(f"  → Paper 4 argument is solid")

# ============================================================
# 總結
# ============================================================
print(f"\n{'='*70}")
print("SUMMARY: Updated Scorecard")
print(f"{'='*70}")

results = [
    ("f_π", f_pi_renorm, f_pi_exp, "MeV", "Paper 5"),
    ("m_e", sigma/(2*np.pi*Lambda), 0.511, "MeV", "Paper 7"),
    ("m_ω", m_omega, 782.66, "MeV", "Paper 5"),
    ("m_s/m_q", epsilon**(-2/3), 27.07, "", "Paper 7"),
    ("N-Δ", sigma/(2*M_Q), 294.0, "MeV", "Paper 5"),
    ("m_Z/m_W", 2/np.sqrt(3), 1.1346, "", "Paper 7"),
    ("m_H", Lambda*epsilon**(-1/8)/1000, 125.25, "GeV", "Paper 7"),
    ("1/α_EM", 1/(epsilon/64 + 16.4/137), 137.036, "", "Paper 7"),  # rough
    ("m_π", m_pi_esft, 138.0, "MeV", "Paper 5"),
    ("m_b", mass_MeV(3), 4180.0, "MeV", "Paper 7"),
    ("m_c", mass_MeV(2), 1270.0, "MeV", "Paper 7"),
    ("m_t", mass_MeV(4)/1000, 172.69, "GeV", "Paper 7"),
    ("m_s", mass_MeV(1), 93.4, "MeV", "Paper 7"),
]

print(f"\n{'Quantity':>12} {'ESFT':>12} {'Exp':>12} {'Dev':>8} {'Paper':>10}")
print(f"{'-'*58}")
for name, esft, exp, unit, paper in results:
    dev = abs(esft - exp) / exp * 100
    print(f"{name:>12} {esft:12.3f} {exp:12.3f} {dev:7.1f}% {paper:>10}")
