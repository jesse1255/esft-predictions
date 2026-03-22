# Paper I — Topological Soliton Coherence Scale

**Journal:** Foundations of Physics (under review)
**Zenodo:** [10.5281/zenodo.19154442](https://doi.org/10.5281/zenodo.19154442)

## Summary

One parameter — the topological soliton core scale ϙ = 0.003 fm — is fitted to α_EM,
then used to derive predictions across seven physical domains without additional free parameters.

## Derivation Chain

```
ϙ = 0.003 fm  (fitted to α_EM)
│
├─→ Λ_core = ℏc/ϙ = 65,776 MeV ≈ 66 GeV     [Eq. 10]
├─→ α_EM from winding integral I(ϙ) = 0.04675  [Eq. 3–4]
├─→ δ(g/2) = 4.4×10⁻¹⁶  (beyond-SM)           [Eq. 5]
├─→ Λ_core ≈ m_W scale  (electroweak emergence) [Eq. 11]
├─→ 1s-2s hydrogen correction ~ (ϙ/a₀)²        [Eq. 8]
├─→ Lamb shift correction                       [Eq. 9]
├─→ Galactic rotation curves (SPARC comparison)  [Eq. 14–16]
└─→ Quantum coherence / decoherence             [Eq. 17–19]
```

## Run

```bash
python derivation_chain.py
```

No dependencies beyond Python 3.x standard library.

## Domains Covered

| Domain | Predictions |
|--------|------------|
| Atomic spectroscopy | α_EM, g-2 correction |
| Hydrogen precision | 1s-2s, Lamb shift |
| Electroweak scale | Λ_core ≈ m_W |
| Galactic dynamics | Flat rotation curves |
| Quantum coherence | Born rule, decoherence timescale |
