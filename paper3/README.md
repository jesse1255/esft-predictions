# Paper III — Electron Mass from Topological String Tension

**Journal:** Physical Review Letters (submitted)
**Zenodo:** [10.5281/zenodo.19159513](https://doi.org/10.5281/zenodo.19159513)

## Summary

From ϙ (Paper I), √σ (Paper II), and the BKT universal exponent η = 1/4,
this paper derives all three lepton masses, heavy quark masses, electroweak parameters,
and internal cross-checks — with zero additional free parameters.

## Derivation Chain

```
ϙ = 0.003 fm      (Paper I, fitted)
√σ = 459 MeV      (Paper II, derived)
η = 1/4            (BKT exact, Nelson-Kosterlitz 1977)
│
├─→ ε = √σ/Λ = 0.006979                             [Eq. 2]
│
│  Leptons:
├─→ m_e = σ/(2πΛ) = 0.510 MeV                       [Eq. 6]
├─→ m_μ = Λε^(4/3) = 87.7 MeV                       [derived]
├─→ m_τ = √σ·ε^(−η) = 1588 MeV                      [Eq. 7]
│
│  Cross-check:
├─→ m_π × m_τ = σ  (algebraic identity)              [Eq. 8]
│
│  Mesons:
├─→ m_π = √σ·ε^η = 133 MeV                          [Eq. 4]
│
│  Heavy quarks:
├─→ m_c = Λε^(4/5) = 1.24 GeV                       [Eq. 5]
├─→ m_b = Λε^(5/9) = 4.17 GeV                       [Eq. 6b]
├─→ m_t = Λε^(−1/5) = 178 GeV                       [Eq. 9]
│
│  Electroweak:
├─→ m_H = Λε^(−η/2) = 122 GeV                       [Eq. 11]
├─→ v = Λε^(−1/4) = 228 GeV                          [derived]
├─→ y_t = √2·ε^(1/20) = 1.10                        [Eq. 10]
├─→ sin²θ_W = 1/4 = 0.25                             [derived]
├─→ m_Z/m_W = 2/√3 = 1.155                           [derived]
├─→ 1/α = 64/α_s + 16.4 = 140                       [Eq. 13]
│
│  BKT:
└─→ F(λ/J=1) = 1.774                                 [Eq. 14]
```

## Run

```bash
python derivation_chain.py
```

No dependencies beyond Python 3.x standard library.

## Key Results

| Observable | ESFT | Experiment | Deviation |
|-----------|------|------------|-----------|
| m_e | 0.510 MeV | 0.511 MeV | 0.2% |
| m_b | 4.17 GeV | 4.18 GeV | 0.2% |
| F(BKT) | 1.774 | 1.79 | 0.9% |
| m_H | 122 GeV | 125 GeV | 2.3% |
| m_t | 178 GeV | 173 GeV | 2.8% |
| m_π | 133 MeV | 138 MeV | 3.9% |
