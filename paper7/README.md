# Paper VII — Fibonacci Structure of Fermion Masses

**Journal:** Physical Review D (in preparation)

## Summary

The mass exponent n in m = Λε^n decomposes into an SU(3) color ladder
and an SU(2) weak modulation whose numerators follow a Fibonacci recurrence.
This reproduces all 6 quark masses and 3 charged lepton masses with zero free parameters.

## The Formula

n(l) = (2 − 2l/N_c) + F_{l-2}/(N_c · D_{l-2})

where:
- l = generation index (0–4)
- N_c = 3 (color), N_w = 2 (weak)
- F_k: Fibonacci sequence seeded by (N_w, N_c+N_w) = (2, 5)
- D_k: alternating denominators N_c+N_w (even) and N_c (odd)

## Key Results

| Particle | l | n | ESFT mass | Experiment | Deviation |
|----------|---|---|-----------|------------|-----------|
| u, d | 0 | 2 | 3.2 MeV | 3.45 MeV | 7.2% |
| s | 1 | 4/3 | 88 MeV | 93 MeV | 6.1% |
| c | 2 | 4/5 | 1239 MeV | 1270 MeV | 2.4% |
| b | 3 | 5/9 | 4170 MeV | 4180 MeV | 0.2% |
| t | 4 | −1/5 | 177.5 GeV | 172.7 GeV | 2.8% |

### Charged Leptons

| Lepton | Formula | ESFT mass | Experiment | Deviation |
|--------|---------|-----------|------------|-----------|
| e | σ/(2πΛ) | 0.510 MeV | 0.511 MeV | 0.2% |
| μ | Λε^(4/3) × √(π/2) | 110 MeV | 106 MeV | 4.0% |
| τ | Koide from A₂ geometry | 1841 MeV | 1777 MeV | 3.6% |

### Koide Relation

The Koide relation Q = 2/3 is derived as a geometric identity:
Q = cos²θ where θ is the tetrahedral angle between the A₂ root plane and the (1,1,1) direction.

### Cabibbo Angle

θ_C = arctan(√(m_d/m_s)) = 13.02° (experiment: 13.04°, deviation: 0.15%)

The Cabibbo angle becomes a zero-parameter prediction when using ESFT-derived quark masses.

### Prerequisites

- **Paper I:** Core scale ϙ
- **Paper II:** String tension √σ, ε
- **Paper IV:** BKT mechanism (√(π/2) renormalization)
