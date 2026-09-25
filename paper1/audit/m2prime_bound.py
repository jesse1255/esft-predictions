"""
Explicit constants in the M2' topological lower bound (REVERSE_REVIEW_R2 §3.4).

    |Q| ≤ C₁ E^{4/3} + C₂ E,
    C₁ = 3 S₃ / (8π² (2 κ_min r_min)^{2/3}),     (Chern–Simons part, three diagonal U(1)s)
    C₂ = 1 / (24π² √(r_min κ₃)),                 (three-cycle part)

S₃ = sharp Sobolev constant in ‖u‖_{L⁶(ℝ³)} ≤ S₃ ‖∇u‖_{L²} (Aubin–Talenti):
S₃ = (3π)^{-1/2} (Γ(3)/Γ(3/2))^{1/3} = 3^{-1/2} (2/π)^{2/3}.

The lower bound E* is the root of C₁E^{4/3} + C₂E = |Q|.

Usage: python m2prime_bound.py   → m2prime_bound.json
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))

S3 = (3 * math.pi) ** -0.5 * (math.gamma(3) / math.gamma(1.5)) ** (1 / 3)
assert abs(S3 - 3 ** -0.5 * (2 / math.pi) ** (2 / 3)) < 1e-12


def constants(r_min, kappa_min, kappa3):
    c1 = 3 * S3 / (8 * math.pi ** 2 * (2 * kappa_min * r_min) ** (2 / 3))
    c2 = 1 / (24 * math.pi ** 2 * math.sqrt(r_min * kappa3)) if kappa3 > 0 else math.inf
    return c1, c2


def e_star(c1, c2, q=1.0):
    """Smallest E with C₁E^{4/3} + C₂E ≥ |Q| (bisection; the left side is increasing)."""
    if math.isinf(c2):
        return 0.0
    lo, hi = 0.0, 1.0
    while c1 * hi ** (4 / 3) + c2 * hi < q:
        hi *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if c1 * mid ** (4 / 3) + c2 * mid < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


rows = []
for label, r_min, kmin, k3 in (("M2' (κ₃ = κ = 2), S₃ point", 2.0, 2.0, 2.0),
                               ("κ₃ = 0.92 (near κ₃*)", 2.0, 2.0, 0.92),
                               ("κ₃ = 0.1", 2.0, 2.0, 0.1),
                               ("M2 (κ₃ = 0)", 2.0, 2.0, 0.0)):
    c1, c2 = constants(r_min, kmin, k3)
    rows.append(dict(model=label, r_min=r_min, kappa_min=kmin, kappa3=k3, C1=c1,
                     C2=(None if math.isinf(c2) else c2),
                     E_lower_Q1=e_star(c1, c2, 1.0), E_lower_Q4=e_star(c1, c2, 4.0),
                     CS_only_bound_Q1=(1 / c1) ** 0.75))
out = dict(S3=S3, rows=rows,
           numerical_reference="embedded Q = 1 Hopfion, ne = 32, a = 3: E = 321.61")
with open(os.path.join(HERE, "m2prime_bound.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1, ensure_ascii=False))
