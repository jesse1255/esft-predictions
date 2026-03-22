# ESFT — Energy String Field Theory

## Numerical Verification

This repository contains the Python verification scripts for the quantitative predictions
of Energy String Field Theory (ESFT).

### Papers

| Paper | Title | Journal | Zenodo DOI |
|-------|-------|---------|------------|
| I | Topological soliton coherence scale | Found. Phys. (under review) | [10.5281/zenodo.19154442](https://doi.org/10.5281/zenodo.19154442) |
| II | Emergent gauge symmetry from Hopf soliton topology in ESFT | Phys. Rev. D (submitted) | [10.5281/zenodo.19159255](https://doi.org/10.5281/zenodo.19159255) |
| III | Electron mass from topological string tension | Phys. Rev. Lett. (submitted) | [10.5281/zenodo.19159513](https://doi.org/10.5281/zenodo.19159513) |

### Quick Start

```bash
python verify_predictions.py
```

Requires only Python 3.x standard library (math module). No external dependencies.

### Core Parameters

| Symbol | Value | Source |
|--------|-------|--------|
| ϙ (Qoppa) | 0.003 fm | Paper I (fitted) |
| √σ | 459 MeV | Paper II (derived) |
| η | 1/4 | BKT exact (Nelson-Kosterlitz 1977) |

### Results Summary

24 predictions from 2 scales + BKT η = 1/4. Zero additional free parameters.

- **< 1%:** 6 predictions (α_s, m_b, m_e, m_ω, ⟨q̄q⟩, F)
- **1-5%:** 8 predictions
- **5-15%:** 7 predictions
- **15-25%:** 3 predictions

### Author

Jesse C. P. Ting — Independent Researcher

### License

MIT
