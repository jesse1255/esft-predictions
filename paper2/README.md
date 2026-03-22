# Paper II — Emergent Gauge Symmetry from Hopf Soliton Topology

**Journal:** Physical Review D (submitted)
**Zenodo:** [10.5281/zenodo.19159255](https://doi.org/10.5281/zenodo.19159255)

## Summary

Three identical Hopf solitons with core scale ϙ produce a rank-3 configuration whose
symmetry group is S₃ = Weyl(A₂), generating SU(3) gauge structure topologically.
The string tension √σ = 459 MeV is derived (not fitted), yielding hadronic predictions
with zero additional free parameters.

## Derivation Chain

```
ϙ = 0.003 fm  (from Paper I)
│
├─→ Λ_core = 65,776 MeV                              [Eq. 10]
├─→ √σ = 459 MeV  (topological string tension)        [Eq. 37]
├─→ ε = √σ/Λ = 0.006979                               [Eq. 12]
│
├─→ α_s = 18π/(11·ln(1/ε²)) = 0.518                   [derived]
├─→ M_Q = (√σ/2π)·ln(1/ε) = 363 MeV                   [Eq. 15]
├─→ m_ω = 2M_Q(1+α_s/2π) = 785 MeV                    [Eq. 20]
├─→ m_N = 3×M_Q = 1088 MeV                             [from M_Q]
├─→ m_Δ−m_N = σ/(2M_Q) = 290 MeV                      [derived]
├─→ ⟨q̄q⟩^(1/3) = (√σ³/2π)^(1/3) = 249 MeV            [Eq. 22]
├─→ m_q = σ/Λ = 3.20 MeV                               [derived]
├─→ m_s = Λε^(4/3) = 87.7 MeV                          [derived]
├─→ f_π = √σ/√(2πNc) = 106 MeV                        [derived]
├─→ Casimir scaling: 6 representations < 3%             [Sec. 5]
├─→ Glueball J^PC spectrum: 10 states, RMS ~5%         [Eq. 35]
└─→ Adams theorem: only SU(3)×SU(2)×U(1) possible     [Sec. 8]
```

## Run

```bash
python derivation_chain.py
```

No dependencies beyond Python 3.x standard library.

## Key Results

| Observable | ESFT | Experiment | Deviation |
|-----------|------|------------|-----------|
| α_s | 0.518 | 0.518 | 0.1% |
| m_ω | 785 MeV | 783 MeV | 0.3% |
| ⟨q̄q⟩^(1/3) | 249 MeV | 250 MeV | 0.5% |
| √σ | 459 MeV | 440 MeV | 4.3% |
| Casimir ratios | exact | lattice | < 3% |
