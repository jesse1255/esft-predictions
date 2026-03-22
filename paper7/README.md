# Paper VII — Born Rule and Coherence Floor

**Journal:** (planned)
**Status:** Theoretical framework complete; experimental prediction identified

## Research Direction

Paper VII derives the Born rule as an emergent property of the ESFT phase field,
and predicts a **coherence floor** — a minimum decoherence rate set by the topological
structure of the vacuum.

### Core Prediction

ESFT visibility function:

```
V(a, T) = exp(−T/T_coh) · exp(−(T/τ_disp)²)
```

- **T_coh ≈ 125 s** from ESFT (derived, not fitted)
- All matter-wave interference experiments: 1−V < 10⁻⁷ (undetectable → Born rule consistent)
- **SQUID/transmon is the only testable window:** requires T₁ > 10 s

### Forward Prediction

Coherence floor: Γ ~ 8×10⁻³ s⁻¹

Current best transmon (Bland 2025, Nature): T₁ = 1.68 ms → ESFT effect is 10⁻⁵ of TLS loss.
**Prediction becomes testable when T₁ > 10 s** (expected within 5–10 years).

### Why This Matters

- The Born rule has never been derived from first principles in standard QM
- ESFT provides a parameter-free prediction for when deviations should appear
- If confirmed, it would be the first experimental evidence for sub-quantum structure

### Prerequisites

- **Paper I:** Core scale ϙ, coherence/decoherence framework (Eq. 17–19)
- **Paper IV:** BKT dynamics underlying the coherence timescale

### Scripts

Coherence calculation scripts will be added upon manuscript completion.
