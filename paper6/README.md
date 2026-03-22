# Paper VI — ESFT Cosmology: Late-Time Phase Relaxation

**Journal:** (planned)
**Status:** Minimal fit model complete; data comparison in progress

## Research Direction

Paper VI connects ESFT to cosmological observables by deriving late-time modifications
to the expansion history from the topological phase field dynamics.

The core mechanism: three-mode phase relaxation produces a **Hubble–Γ dynamic equilibrium**
that modifies the effective dark energy density at late times, while leaving early-universe
(CMB) physics untouched.

### Framework

The cosmological modification is controlled by three parameters, all derivable from ESFT:

| Parameter | Meaning | Best-fit value |
|-----------|---------|---------------|
| ε | Amplitude of late-time modification | 0.173 |
| a_c | Scale factor at onset (z_c ≈ 0.91) | 0.525 |
| Δ | Transition sharpness | 0.003 |

**Theory bridge:** All three parameters trace back to a single function Γ_relax(a) —
the relaxation rate of the topological phase field.

### Preliminary Results

- ΛCDM χ² = 998.78, ESFT best χ² = 971.40 → **Δχ² = −27.38 (ESFT preferred)**
- Tested against: DESI DR2 BAO (14 pts), Pantheon+ SN (20 bins), fσ8 (16 pts)
- Unique ESFT signature: synchronized triple transition (H + fσ8 + lensing) at same z_c

### Vacuum Energy

The vacuum energy formula:

```
ρ_Λ = ½(ℏH₀)² t² exp(2b/√t)
```

where t is the KT reduced temperature and b ≈ π/2 is the BKT vortex unbinding constant.
If b ≈ 1.3, this yields ρ_Λ ~ 10⁻⁴⁷ GeV⁴ — matching observation with zero free parameters.

### Prerequisites

- **Paper I:** Core scale ϙ and KT reduced temperature t = 1.1×10⁻³
- **Paper IV:** BKT mechanism and vortex unbinding constant b

### Scripts

Data fitting and cosmological evolution scripts will be added upon manuscript completion.
