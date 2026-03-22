# ESFT — Energy String Field Theory

## Numerical Verification

This repository contains verification scripts and simulations for the quantitative
predictions of Energy String Field Theory (ESFT).

### Papers

| Paper | Title | Journal | Zenodo DOI |
|-------|-------|---------|------------|
| I | Topological soliton coherence scale | Found. Phys. (under review) | [10.5281/zenodo.19154442](https://doi.org/10.5281/zenodo.19154442) |
| II | Emergent gauge symmetry from Hopf soliton topology in ESFT | Phys. Rev. D (submitted) | [10.5281/zenodo.19159255](https://doi.org/10.5281/zenodo.19159255) |
| III | Electron mass from topological string tension | Phys. Rev. Lett. (submitted) | [10.5281/zenodo.19159513](https://doi.org/10.5281/zenodo.19159513) |

### Repository Structure

```
├── verify_predictions.py      # All 24 predictions vs experiment (Paper I–III)
├── paper1/
│   └── derivation_chain.py    # ϙ → α_EM → Λ_core → 7 domains (step-by-step)
├── paper2/
│   └── derivation_chain.py    # ϙ → √σ → α_s → hadron spectrum (step-by-step)
├── paper3/
│   └── derivation_chain.py    # ϙ + √σ + η → leptons + EW + heavy quarks
└── simulations/
    ├── README.md              # Simulation details & full parameter tables
    └── bkt_animation.py       # BKT phase transition (Paper IV, Sec. 3)
```

### Quick Start

```bash
# Verify all 24 predictions (no dependencies beyond Python 3.x stdlib)
python verify_predictions.py

# Run BKT simulation (requires numpy, matplotlib, ffmpeg)
cd simulations
pip install numpy matplotlib
python bkt_animation.py
```

### Core Parameters

| Symbol | Value | Source | Role |
|--------|-------|--------|------|
| ϙ (Qoppa) | 0.003 fm | Paper I, Eq. (12) | Fitted to α_EM; single free parameter |
| √σ | 459 MeV | Paper II, Eq. (37) | Derived from ϙ via topological string tension |
| η | 1/4 | Nelson–Kosterlitz (1977) | BKT universal exponent; not fitted |

All 24 predictions follow from these inputs. Zero additional free parameters beyond ϙ.

### Results

![Predicted vs Observed](predictions_vs_experiment.png)

*Left: mass predictions (log-log, MeV). Right: dimensionless predictions. Dashed line = perfect agreement. Green < 5%, yellow 5–15%, red > 15%.*

### Results Summary

| Accuracy | Count | Examples |
|----------|-------|----------|
| < 1% | 6 | α_s, m_b, m_e, m_ω, ⟨q̄q⟩, F |
| 1–5% | 8 | m_τ, Λ_QCD, √σ_fund, ... |
| 5–15% | 7 | m_Δ − m_N, f_π, ... |
| 15–25% | 3 | m_c, ... |

### Reproducibility

Every script in this repository is deterministic (fixed random seeds where applicable)
and self-contained. To reproduce any result:

1. Clone this repo
2. Run the relevant script
3. Compare output to the values in the corresponding paper

No proprietary software, cloud APIs, or external datasets are required.

### AI Disclosure

AI-assisted tools were used for algebraic verification, numerical cross-checks,
and manuscript preparation during the development of ESFT. The conceptual framework,
hypothesis selection, research direction, iterative branch decisions, and final
interpretation of all results were determined by the author. The author takes sole
responsibility for the content of all papers and code in this repository.

### Author

Jesse C. P. Ting — Independent Researcher

I am an independent researcher working without institutional funding, laboratory
infrastructure, or formal academic affiliation. My work is driven by long-developed
physical intuition, iterative theoretical exploration, and AI-assisted computational
support. The conceptual direction, model selection, and interpretive framework are
my own; AI is used as a tool to accelerate checking, scanning, and refinement.

- Email: jass168611@gmail.com
- ORCID: [see Paper I]

### License

MIT
