# ESFT — Energy String Field Theory

Numerical verification, derivation chains, and simulations for the quantitative
predictions of Energy String Field Theory (ESFT).

![Predicted vs Observed](predictions_vs_experiment.png)

*Left: mass predictions (log-log, MeV). Right: dimensionless predictions.
Dashed line = perfect agreement. Green < 5%, yellow 5–15%, red > 15%.*

---

## Key Results (April 2026)

| Result | ESFT Prediction | Experiment | Status |
|--------|----------------|------------|--------|
| Particle mass spectrum | 152 particles from one formula, zero fitted parameters | PDG 2024 | 98.7% within 10% |
| CKM quark mixing | All 4 parameters derived from A₂ geometry | PDG | θ₁₃: 0.1%, θ₂₃: 1.1%, θ₁₂: 0.15%, δ_CP: 0.7% |
| PMNS neutrino mixing | All 4 parameters derived from A₂ geometry | NuFIT 5.3 | θ₂₃: 0.02%, θ₁₂: 0.02%, θ₁₃: 1.6% |
| Muon g−2 | Δa_μ = 3.4 × 10⁻⁹ (zero parameters) | Fermilab 2.5 × 10⁻⁹ | 36% (zero-parameter, no fit) |
| Pion decay constant | f_π = 93.95 MeV | 92.07 MeV | 2.0% |
| π-π scattering length | a₀⁰ = 0.214 | 0.220 ± 0.005 | 2.8% (improved from 29% at tree level) |
| Newton's constant | Obtained from soliton field equation | G_N | Linearized Einstein equation reproduced |
| Born rule | Obtained from soliton Fokker-Planck dynamics | — | Not postulated |
| Electron mass | m_e = 0.510 MeV | 0.511 MeV | 0.2% |
| Bottom quark mass | m_b = 4.17 GeV | 4.18 GeV | 0.2% |

## Structure of the Predictions

The mass formula m = Λε^n contains no continuously adjustable parameters. The exponent n is determined by the topological winding number, with its numerator and denominator built from the color gauge dimension N_c = 3 and the weak isospin dimension N_w = 2. These integers also define the Standard Model gauge group SU(3) × SU(2) × U(1); within ESFT, they arise from the rank-3 Hopf soliton topology (Paper II).

Deviations from experiment have identifiable structural origins: either (a) a higher-order topological correction not yet included, or (b) a point where ESFT and the Standard Model make different predictions. The two largest deviations (η meson at 16.7% and W boson at 13.9%) are traced to the axial anomaly and Weinberg angle splitting respectively; corrections using only N_c and N_w reduce both to within 3%.

### Fermion Masses — Zero Free Parameters

All predictions below use a single fitted scale ϙ = 0.003 fm. The exponents n are determined entirely by gauge group dimensions (N_c = 3, N_w = 2) and Fibonacci structure.

**Quarks:**

| Particle | Generation | Exponent n | ESFT | Experiment | Deviation |
|----------|-----------|-----------|------|------------|-----------|
| u, d | 1st | 2 | 3.2 MeV | 3.45 MeV | 7.2% |
| s | 2nd | 4/3 | 88 MeV | 93 MeV | 6.1% |
| c | 3rd | 4/5 | 1239 MeV | 1270 MeV | 2.4% |
| b | 3rd | 5/9 | 4170 MeV | 4180 MeV | **0.2%** |
| t | 3rd | −1/5 | 177.5 GeV | 172.7 GeV | 2.8% |

**Charged Leptons:**

| Particle | Formula | ESFT | Experiment | Deviation |
|----------|---------|------|------------|-----------|
| e | σ/(2πΛ) | 0.510 MeV | 0.511 MeV | **0.2%** |
| μ | Λε^(4/3) × √(π/2) | 110 MeV | 106 MeV | 4.0% |
| τ | Koide from A₂ | 1841 MeV | 1777 MeV | 3.6% |

**Mixing Angles (CKM + PMNS):**

| Parameter | ESFT | Experiment | Deviation |
|-----------|------|------------|-----------|
| θ₁₂ (Cabibbo) | 13.02° | 13.04° | 0.15% |
| θ₂₃ (CKM) | 2.41° | 2.38° | 1.1% |
| θ₁₃ (CKM) | 0.201° | 0.201° | **0.1%** |
| δ_CP (CKM) | 68.3° | 68.8° | 0.7% |
| θ₂₃ (PMNS) | 48.99° | 49.0° | **0.02%** |
| θ₁₂ (PMNS) | 33.40° | 33.41° | **0.02%** |
| θ₁₃ (PMNS) | 8.68° | 8.54° | 1.6% |

---

## Testable Predictions

ESFT makes several quantitative predictions that differ from the Standard Model and can be confirmed or ruled out by current or near-future experiments:

| Prediction | Value | Where to Test | SM Expectation |
|-----------|-------|---------------|----------------|
| q₆ resonance | 65.78 GeV, J^P = 1/2⁻, Γ ≈ 3.2 MeV | LHC (ATLAS/CMS) | No such state predicted |
| BKT phase transition in π-π scattering | Discontinuity at √s = 918 MeV | BESIII, COMPASS | Smooth crossover |
| Electron g−2 soliton correction | δ(g/2) = 4.4 × 10⁻¹⁶ | Next-gen g−2 experiments | Zero (point particle) |
| Soliton form factor (electron) | F(q) = (πqa/2)/sinh(πqa/2); 0.38% at 10 GeV, 9.6% at 50 GeV | Belle II, BES III | Point-like (F = 1) |
| Density-dependent G | ΔG/G ~ 5 × 10⁻⁵ at neutron star density | Binary pulsar timing | G = const |
| QNM triplet degeneracy | Ringdown amplitude × √3 from A₂ structure | LIGO/Virgo/KAGRA O5+ | Non-degenerate |
| Coherence floor | Γ ~ 8 × 10⁻³ s⁻¹; testable when qubit T₁ > 10 s | Transmon/SQUID (5–10 yr) | No floor predicted |

If any of these are experimentally excluded, the corresponding sector of ESFT is falsified.

---

## ESFT vs Standard Model

| Aspect | Standard Model | ESFT |
|--------|---------------|------|
| Free parameters | 19+ (masses, couplings, angles fitted to data) | 1 fitted scale (ϙ = 0.003 fm); rest derived |
| Particle masses | Input parameters | Derived: m = Λε^n where n comes from topology |
| CKM/PMNS mixing | 8 input parameters | Derived from A₂ root geometry (N_c = 3, N_w = 2) |
| Gravity | Separate framework (GR) | Linearized Einstein equation obtained from soliton fields |
| Quantum mechanics | Born rule postulated | Born rule obtained from soliton Fokker-Planck dynamics |
| Mass hierarchy | Not derived (m_t/m_e ~ 10⁵ unexplained) | Fibonacci structure from SU(3)×SU(2) modulation |
| Three generations | Not derived | Mapped to three topological phases of the A₂ field |
| Dark matter profiles | Fitted (NFW, etc.) | Obtained from soliton density distribution |

## Papers

### Published Preprints

| Paper | Title | Zenodo DOI |
|-------|-------|------------|
| I | Topological soliton coherence scale | [10.5281/zenodo.19802907](https://doi.org/10.5281/zenodo.19802907) |
| II | Emergent gauge symmetry from Hopf soliton topology | [10.5281/zenodo.19803030](https://doi.org/10.5281/zenodo.19803030) |
| III | Electron mass from topological string tension | [10.5281/zenodo.19803049](https://doi.org/10.5281/zenodo.19803049) |

### In Preparation (IV–XIII)

**Paper IV: BKT Topological Protection.**
Establishes why particles are stable. The BKT vortex binding mechanism arises naturally from the Hopf fibration structure, providing exact topological protection for all soliton states. Extends BKT theory to coupled three-field systems. Derives a density-driven phase transition relevant to astrophysical environments. Explains proton longevity and quark-gluon plasma near-perfect fluidity from the same mechanism.

**Paper V: Hadronic Physics.**
Derives nuclear force parameters from the same framework that produces particle masses. Addresses the pion decay constant, vector meson masses, the nucleon hard-core radius, and the nucleon-delta mass splitting, all from a single starting point with no additional free parameters.

**Paper VI: Emergent Gravity.**
Proposes that gravity is not fundamental but emerges from the progressive alignment of soliton topological directions under increasing matter density. Derives Newton's constant from two independent routes (classical field equation and quantum induced gravity). Shows that the full nonlinear Einstein equation follows from three independent mathematical arguments. Predicts density-dependent gravitational coupling testable by binary pulsar timing.

**Paper VII: Fibonacci Structure of Fermion Masses.**
Resolves why quarks and leptons have the masses they do. The mass exponents decompose into an SU(3) color ladder and SU(2) weak modulation whose numerators follow Fibonacci sequences. Reproduces all six quark masses and three charged lepton masses with zero free parameters. Derives the Koide relation as a geometric identity.

**Paper VIII: Black Holes and Gravitational Phenomenology.**
Develops consequences of emergent gravity for black holes, neutron stars, and gravitational waves. Identifies black holes as regions of complete topological locking with a Type I/II classification. The three-field structure produces distinctive signatures in black hole entropy, Hawking radiation, and gravitational wave ringdown. Predicts a modified neutron star maximum mass accommodating mass-gap objects. Identifies a topological phase boundary near the horizon producing gravitational wave echoes at a characteristic timescale distinguishable from competing models.

**Paper IX: CKM and PMNS Mixing.**
Derives all eight quark and neutrino mixing parameters (four CKM + four PMNS) from root-lattice geometry with zero free parameters. The Standard Model treats these as independent inputs; ESFT derives them from the same algebraic structure that determines particle masses. Reveals a structural quark-lepton complementarity: CKM angles arise from mass-ratio suppression while PMNS angles are direct geometric projections.

**Paper X: Emergent Quantum Mechanics.**
Proposes that quantum mechanics is not fundamental but emerges from classical soliton dynamics near a topological phase transition. The Born rule, superposition, interference, measurement-induced collapse, and Bell nonlocality all emerge from the underlying field theory without additional axioms. The complex structure of Hilbert space follows from the circular topology of the phase field. Derives quantitative decoherence rates from soliton-phonon scattering.

**Paper XI: Scattering Amplitudes.**
Bridges the gap between static mass predictions and dynamical scattering processes. Connects topological soliton interactions to measurable cross sections using sine-Gordon integrability and collective coordinate methods. Achieves significant improvement over tree-level results through topological corrections. Makes several testable predictions including a new narrow resonance as a unique collider search target with no Standard Model counterpart.

**Paper XII: Cosmological Predictions.**
Demonstrates that four fundamental quantities spanning 96 orders of magnitude—dark matter relaxation, vacuum energy, CMB temperature, and neutrino mass—all derive from a single algebraic object through different projection rules. Resolves the cosmological constant problem, predicts the neutrino mass consistent with oscillation data and cosmological bounds, and addresses baryogenesis through critical slowing at a topological phase transition. No adjustable parameters.

**Paper XIII: Unified Master Equation.**
Derives a single equation that simultaneously contains gauge forces and gravity as two projections of the same underlying field. The spacetime metric and gauge connection are both functionals of the phase field rather than independent fields. Verifies that all results from Papers I–XII emerge as limiting cases. Resolves the three open mathematical problems of the unified framework: metric functional form, self-consistent coupling existence, and cross-sector hierarchy stability.

## Repository Structure

```
esft-predictions/
├── README.md                           ← This file
├── predictions_vs_experiment.png       ← 24 predictions vs experiment (45° plot)
│
├── paper1/                             ← Paper I: Soliton coherence scale
│   ├── README.md                       ← Derivation chain diagram + domain table
│   └── derivation_chain.py             ← ϙ → α_EM → Λ_core → 7 domains
│
├── paper2/                             ← Paper II: Emergent gauge symmetry
│   ├── README.md                       ← Chain diagram + key results
│   └── derivation_chain.py             ← ϙ → √σ → α_s → hadron spectrum
│
├── paper3/                             ← Paper III: Electron mass
│   ├── README.md                       ← Chain diagram + key results
│   └── derivation_chain.py             ← ϙ + √σ + η → leptons + EW + quarks
│
├── paper4/                             ← Paper IV: BKT phase transition
│   ├── README.md                       ← Full parameter table + algorithm
│   └── bkt_animation.py                ← Monte Carlo simulation → MP4
│
├── paper5/                             ← Paper V: Hadronic physics
│   └── README.md                       ← f_π, m_ω, nucleon core, N-Δ splitting
│
├── paper6/                             ← Paper VI: Emergent gravity
│   └── README.md                       ← Newton's constant, Sakharov consistency
│
├── paper7/                             ← Paper VII: Fibonacci fermion masses
│   └── README.md                       ← Quark/lepton mass formula, Koide relation
│
├── paper8/                             ← Paper VIII: Black holes & gravitational phenomenology
│   └── README.md                       ← BH entropy, Hawking, QNMs, TOV mass
│
├── paper9/                             ← Paper IX: CKM & PMNS mixing
│   └── README.md                       ← All 8 mixing parameters from A₂ geometry
│
├── paper10/                            ← Paper X: Emergent quantum mechanics
│   └── README.md                       ← Born rule, decoherence, path integral
│
├── paper11/                            ← Paper XI: Scattering amplitudes
│   └── README.md                       ← π-π scattering, form factors, 5 predictions
│
└── scripts/                            ← Cross-paper tools
    ├── verify_predictions.py           ← Original 24 predictions in one run
    ├── verify_all_papers.py            ← Full 152+ prediction verification
    └── plot_predictions.py             ← Generate the comparison plot
```

## Quick Start

```bash
# Verify all 24 predictions (Python 3.x stdlib only)
python scripts/verify_predictions.py

# Full verification across all 11 papers
python scripts/verify_all_papers.py

# Step-by-step derivation for a specific paper
python paper1/derivation_chain.py
python paper2/derivation_chain.py
python paper3/derivation_chain.py

# BKT simulation (requires numpy, matplotlib, ffmpeg)
cd paper4 && pip install numpy matplotlib && python bkt_animation.py

# Regenerate comparison plot (requires numpy, matplotlib)
python scripts/plot_predictions.py
```

## Core Parameters

| Symbol | Value | Source | Role |
|--------|-------|--------|------|
| ϙ (Qoppa) | 0.003 fm | Paper I, Eq. (12) | Fitted to α_EM; single free parameter |
| √σ | 459 MeV | Paper II, Eq. (37) | Derived from ϙ via topological string tension |
| η | 1/4 | Nelson–Kosterlitz (1977) | BKT universal exponent; not fitted |

All 24 predictions follow from these inputs. Zero additional free parameters beyond ϙ.

## Results Summary

Papers I–III established 24 predictions from 2 scales (ϙ, √σ) + BKT η = 1/4.
Papers IV–XIII extend this to 152+ predictions covering the full particle spectrum,
mixing matrices, gravity, quantum mechanics, scattering, cosmology, and unification.

| Scope | Count | Best accuracy |
|-------|-------|---------------|
| Particle masses (Papers III, VII) | 152 | m_e: 0.2%, m_b: 0.2% |
| CKM mixing (Paper IX) | 4 parameters | θ₁₃: 0.1% |
| PMNS mixing (Paper IX) | 4 parameters | θ₂₃: 0.02% |
| Hadronic physics (Paper V) | 5 observables | m_ω: 0.3%, f_π: 2.0% |
| Scattering (Paper XI) | 2 amplitudes + predictions | π-π: 2.8% |
| Gravity (Paper VI) | G_N obtained | Linearized Einstein eq. reproduced |

## Reproducibility

Every script in this repository is deterministic (fixed random seeds where applicable)
and self-contained. To reproduce any result:

1. Clone this repo
2. Run the relevant script
3. Compare output to the values in the corresponding paper

No proprietary software, cloud APIs, or external datasets are required.

## AI Disclosure

AI-assisted tools were used for algebraic verification, numerical cross-checks,
and manuscript preparation during the development of ESFT. The conceptual framework,
hypothesis selection, research direction, iterative branch decisions, and final
interpretation of all results were determined by the author. The author takes sole
responsibility for the content of all papers and code in this repository.

## Call for Collaboration

ESFT is independent, unfunded research. I welcome collaboration from researchers working in:

- **Topological phases of matter** — soliton physics, skyrmions, vortex dynamics
- **BKT transitions** — topological protection, 2D phase transitions in 3+1D systems
- **Emergent symmetry** — gauge structure from topology, condensed matter analogies
- **Beyond-Standard-Model physics** — parameter-free approaches to mass generation
- **Mathematical physics** — A₂ root systems, Hopf fibrations, Fibonacci structures in physics

If you find errors, have questions, or want to collaborate, please reach out:
**Contact:** jass168611@gmail.com

## Author

Jesse C. P. Won Ting — Independent Researcher

Independent researcher. The conceptual direction, model selection, and interpretive
framework are my own; AI-assisted tools were used to accelerate algebraic verification
and numerical cross-checks.

- Email: jass168611@gmail.com
- ORCID: [see Paper I]

## License

MIT
