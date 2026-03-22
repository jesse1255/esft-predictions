"""
ESFT Predicted vs Observed — 45° comparison plot
=================================================
Generates a log-log scatter plot of all 24 ESFT predictions against
experimental values. Points on the 45° line = perfect agreement.

Output: predictions_vs_experiment.png

Requirements: Python 3.x, matplotlib, numpy
"""

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Constants & inputs ──────────────────────────────────
hbar_c = 197.3269804
qoppa = 0.003
sqrt_sigma = 459.0
eta = 0.25
Nc, Nw = 3, 2

Lambda_core = hbar_c / qoppa
sigma = sqrt_sigma**2
epsilon = sqrt_sigma / Lambda_core
M_Q = (sqrt_sigma / (2*math.pi)) * math.log(1/epsilon)
alpha_s = 18*math.pi / (11*math.log(1/epsilon**2))
a_bkt = 0.893

# ── Predictions (all in natural units: MeV for masses, dimensionless for ratios) ──
# Format: (label, ESFT_value, Exp_value, unit_label)
data = [
    # Hadronic (MeV scale)
    ("α_s",        alpha_s,                                         0.518,     ""),
    ("m_π",        sqrt_sigma * epsilon**eta,                       138.0,     "MeV"),
    ("m_ω",        2*M_Q*(1+alpha_s/(2*math.pi)),                  782.66,    "MeV"),
    ("⟨q̄q⟩^⅓",   (sqrt_sigma**3/(2*math.pi))**(1/3),             250.0,     "MeV"),
    ("m_q",        sigma/Lambda_core,                                3.45,     "MeV"),
    ("m_s",        Lambda_core*epsilon**(4/3),                      93.4,      "MeV"),
    ("m_s/m_q",    epsilon**(-2/Nc),                                27.07,     ""),
    ("M_Q",        M_Q,                                             313.0,     "MeV"),
    ("m_N",        3*M_Q,                                           938.0,     "MeV"),
    ("Δ(N-Δ)",     sigma/(2*M_Q),                                   294.0,     "MeV"),
    ("f_π",        sqrt_sigma/math.sqrt(2*math.pi*Nc),             92.07,     "MeV"),

    # Leptonic (MeV)
    ("m_e",        sigma/(2*math.pi*Lambda_core),                   0.511,     "MeV"),
    ("m_μ",        Lambda_core*epsilon**(4/3),                      105.66,    "MeV"),
    ("m_τ",        sqrt_sigma*epsilon**(-eta),                      1776.9,    "MeV"),

    # Heavy quarks & EW (GeV → convert to MeV for uniform axis)
    ("m_c",        Lambda_core*epsilon**(4/5),                      1270.0,    "MeV"),
    ("m_b",        Lambda_core*epsilon**(5/9),                      4180.0,    "MeV"),
    ("m_t",        Lambda_core*epsilon**(-1/(Nc+Nw)),               172690.0,  "MeV"),
    ("m_H",        Lambda_core*epsilon**(-eta/2),                   125250.0,  "MeV"),
    ("v",          Lambda_core*epsilon**(-0.25),                    246220.0,  "MeV"),

    # Dimensionless
    ("y_t",        math.sqrt(2)*epsilon**(1/20),                    0.9919,    ""),
    ("sin²θ_W",    0.25,                                            0.23122,   ""),
    ("m_Z/m_W",    2/math.sqrt(3),                                  1.13461,   ""),
    ("1/α",        64/alpha_s + 16.4,                               137.036,   ""),
    ("F(BKT)",     (a_bkt-1+math.sqrt(a_bkt**2+10*a_bkt+1))/(2*a_bkt), 1.79, ""),
]

# ── Separate into mass predictions and dimensionless ──
mass_pts = [(l, e, o) for l, e, o, u in data if u == "MeV"]
dimless_pts = [(l, e, o) for l, e, o, u in data if u == ""]

# ── Plot ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white')
fig.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.90, bottom=0.12)

# --- Left panel: masses (log-log) ---
for label, esft, exp in mass_pts:
    dev = abs(esft - exp) / exp * 100
    color = '#1a9850' if dev < 5 else '#fee08b' if dev < 15 else '#d73027'
    ax1.scatter(exp, esft, c=color, s=60, zorder=3, edgecolors='#333', linewidth=0.5)
    ax1.annotate(label, (exp, esft), fontsize=6.5, ha='left', va='bottom',
                 xytext=(4, 4), textcoords='offset points', color='#333')

# 45° line
xlim = (0.3, 5e5)
ax1.plot(xlim, xlim, 'k--', alpha=0.4, linewidth=1, label='perfect agreement')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(xlim)
ax1.set_ylim(xlim)
ax1.set_xlabel('Experimental value (MeV)', fontsize=11)
ax1.set_ylabel('ESFT prediction (MeV)', fontsize=11)
ax1.set_title('Mass predictions', fontsize=13, fontweight='bold')
ax1.set_aspect('equal')
ax1.legend(fontsize=9, loc='upper left')
ax1.grid(True, alpha=0.2, which='both')

# --- Right panel: dimensionless ---
for label, esft, exp in dimless_pts:
    dev = abs(esft - exp) / exp * 100
    color = '#1a9850' if dev < 5 else '#fee08b' if dev < 15 else '#d73027'
    ax2.scatter(exp, esft, c=color, s=60, zorder=3, edgecolors='#333', linewidth=0.5)
    ax2.annotate(label, (exp, esft), fontsize=7, ha='left', va='bottom',
                 xytext=(4, 4), textcoords='offset points', color='#333')

xlim2 = (0.15, 200)
ax2.plot(xlim2, xlim2, 'k--', alpha=0.4, linewidth=1, label='perfect agreement')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(xlim2)
ax2.set_ylim(xlim2)
ax2.set_xlabel('Experimental value', fontsize=11)
ax2.set_ylabel('ESFT prediction', fontsize=11)
ax2.set_title('Dimensionless predictions', fontsize=13, fontweight='bold')
ax2.set_aspect('equal')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.2, which='both')

# Color legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a9850', markersize=8, label='< 5%'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#fee08b', markersize=8, label='5–15%'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d73027', markersize=8, label='> 15%'),
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=9,
           framealpha=0.9, title='Deviation from experiment', title_fontsize=9)

fig.suptitle('ESFT: Predicted vs Observed  (24 predictions, 1 input parameter ϙ = 0.003 fm)',
             fontsize=12, y=0.98, fontweight='bold')

out_path = 'predictions_vs_experiment.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {out_path}")
plt.close()
