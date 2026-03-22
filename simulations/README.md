# ESFT Simulations

## BKT Topological Phase Transition (`bkt_animation.py`)

2D XY-model Monte Carlo simulation demonstrating the Berezinskii–Kosterlitz–Thouless (BKT) transition — the mechanism underlying topological locking in ESFT.

### What it shows

1. **High temperature (T > T_BKT):** Disordered phase field with ~1800 free vortex–antivortex pairs
2. **Critical region (T ≈ T_BKT ≈ 0.893 J):** Sharp annihilation — vortex count drops from ~1800 to ~70
3. **Low temperature (T < T_BKT):** Topological locking — only 4–8 bound pairs survive

### Run

```bash
pip install numpy matplotlib
python bkt_animation.py
```

Requires `ffmpeg` on PATH for MP4 output.

**Output:** `bkt_evolution.mp4` — 200 frames, ~17s @ 12fps, 128×128 lattice.

### Physics

- Vectorized checkerboard Metropolis algorithm
- Vortex detection via plaquette circulation: q = (1/2π) ∮ dΘ
- Quench protocol: T linearly cooled from 2.5 → 0.3 (T_BKT ≈ 0.893)

The simulation demonstrates that topological defects (vortex–antivortex pairs) spontaneously bind below T_BKT without any external intervention — the same mechanism that, in ESFT, generates stable soliton structures from the phase field Θ.
