# Paper IV — BKT Topological Phase Transition

**Journal:** (in preparation)

## BKT Phase Transition Simulation

**Script:** `bkt_animation.py`
**Supports:** Paper IV, Sec. 3 — Topological locking mechanism

### Purpose

Demonstrates that vortex–antivortex binding in a 2D XY model occurs spontaneously
below T_BKT, without external intervention. In ESFT, this is the mechanism that
generates stable soliton structures from the phase field Θ.

### What it shows

| Phase | Temperature | Vortex count | Physical meaning |
|-------|-------------|-------------|------------------|
| Disordered | T > T_BKT (~1.0) | ~1800 pairs | Free vortices, no confinement |
| Critical | T ≈ T_BKT (0.893 J) | ~70 pairs | Sharp pair annihilation |
| Locked | T < T_BKT (~0.8) | 4–8 pairs | Topological binding; only bound pairs survive |

### Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Lattice size | 128 × 128 | Periodic boundary conditions |
| Coupling J | 1.0 | Sets energy scale |
| T_start | 2.5 | Well above T_BKT |
| T_end | 0.3 | Well below T_BKT |
| T_BKT | 0.893 J | Kosterlitz–Thouless critical temperature |
| Cooling protocol | Linear quench, 200 steps | T_start → T_end over N_FRAMES |
| MC sweeps/frame | 20 | Vectorized checkerboard Metropolis |
| Random seed | 42 | Fixed for reproducibility |
| Output frames | 200 | 12 fps → ~17s video |
| Output resolution | 120 dpi | 14 × 6.5 inch figure |

### Algorithm

- **Update:** Vectorized checkerboard Metropolis (even/odd sublattice alternation)
- **Vortex detection:** Plaquette circulation q = (1/2π) ∮ dΘ, computed via `np.angle(exp(iΔΘ))` to handle branch cuts
- **Visualization:** Left panel = phase field Θ(x,y) with HSV colormap; Right panel = vortex charge map (+1 cyan, −1 red)

### Run

```bash
pip install numpy matplotlib
python bkt_animation.py
```

Requires `ffmpeg` on PATH.

**Output:** `bkt_evolution.mp4` (200 frames, ~17s @ 12fps, ~31 MB)

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| NumPy | any | Array operations, MC updates |
| Matplotlib | any | Rendering frames |
| ffmpeg | any | MP4 encoding |
