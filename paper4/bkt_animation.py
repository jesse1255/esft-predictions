"""
ESFT Paper 4 — BKT Topological Phase Transition Animation
==========================================================
2D XY model quench: random phase → vortex-antivortex pairs → topological locking.
Output: bkt_evolution.mp4

Uses vectorized checkerboard Metropolis for speed.

Author: jeje + sisi | 2026-03-22
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, sys, time

# ── Parameters ──────────────────────────────────────────
N        = 128          # lattice size
J        = 1.0          # coupling
T_START  = 2.5          # well above T_BKT ≈ 0.893 J
T_END    = 0.3          # well below T_BKT
N_FRAMES = 200          # total animation frames
SWEEPS_PER_FRAME = 20   # checkerboard sweeps between frames
SEED     = 42

T_BKT = 0.893 * J
temps = np.linspace(T_START, T_END, N_FRAMES)

np.random.seed(SEED)
theta = np.random.uniform(0, 2*np.pi, (N, N))

# ── Vectorized checkerboard Metropolis ──────────────────
def neighbor_sum_cos(theta, proposed):
    """Compute sum of cos(proposed - neighbor) for all sites."""
    return (np.cos(proposed - np.roll(theta, 1, axis=0)) +
            np.cos(proposed - np.roll(theta, -1, axis=0)) +
            np.cos(proposed - np.roll(theta, 1, axis=1)) +
            np.cos(proposed - np.roll(theta, -1, axis=1)))

def mc_sweep_vectorized(theta, T, n_sweeps=1):
    """Checkerboard vectorized Metropolis on XY model."""
    for _ in range(n_sweeps):
        for parity in [0, 1]:
            # checkerboard mask
            ii, jj = np.meshgrid(range(N), range(N), indexing='ij')
            mask = ((ii + jj) % 2 == parity)

            # propose new angles
            delta = np.random.uniform(-np.pi, np.pi, (N, N))
            proposed = (theta + delta) % (2 * np.pi)

            # energy difference
            s_new = neighbor_sum_cos(theta, proposed)
            s_old = neighbor_sum_cos(theta, theta)
            dE = -J * (s_new - s_old)

            # accept/reject
            accept = (dE < 0) | (np.random.rand(N, N) < np.exp(-dE / max(T, 1e-10)))
            accept &= mask

            theta = np.where(accept, proposed, theta)
    return theta

# ── Vortex detection ────────────────────────────────────
def detect_vortices(theta):
    dtheta_x = np.angle(np.exp(1j * (np.roll(theta, -1, axis=1) - theta)))
    dtheta_y = np.angle(np.exp(1j * (np.roll(theta, -1, axis=0) - theta)))
    curl = (dtheta_x + np.roll(dtheta_y, -1, axis=1)
            - np.roll(dtheta_x, -1, axis=0) - dtheta_y)
    charge = np.round(curl / (2 * np.pi)).astype(int)
    pos = np.argwhere(charge == 1)
    neg = np.argwhere(charge == -1)
    return pos, neg

# ── Simulate ────────────────────────────────────────────
print(f"BKT quench: {N}x{N}, {N_FRAMES} frames, {SWEEPS_PER_FRAME} sweeps/frame")
print(f"T: {T_START:.2f} -> {T_END:.2f}  (T_BKT ~ {T_BKT:.3f})")

t0 = time.time()
frames_data = []

for idx, T in enumerate(temps):
    theta = mc_sweep_vectorized(theta, T, SWEEPS_PER_FRAME)
    pos, neg = detect_vortices(theta)
    frames_data.append({
        'theta': theta.copy(),
        'pos': pos, 'neg': neg,
        'T': T, 'n_pos': len(pos), 'n_neg': len(neg),
    })
    if (idx + 1) % 10 == 0 or idx == 0:
        elapsed = time.time() - t0
        eta = elapsed / (idx + 1) * (N_FRAMES - idx - 1)
        phase = "HIGH-T" if T > T_BKT*1.1 else ("CRITICAL" if T > T_BKT*0.9 else "LOCKED")
        print(f"  [{idx+1:3d}/{N_FRAMES}] T={T:.3f}  v+={len(pos):4d} v-={len(neg):4d}  "
              f"[{phase}]  {elapsed:.0f}s elapsed, ~{eta:.0f}s left")

print(f"\nSimulation done in {time.time()-t0:.1f}s. Rendering...")

# ── Render ──────────────────────────────────────────────
fig, (ax_phase, ax_vortex) = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='#0a0a0a')
fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.05, wspace=0.12)

im = ax_phase.imshow(frames_data[0]['theta'], cmap='hsv', vmin=0, vmax=2*np.pi,
                     interpolation='bilinear', origin='lower')
scat_pos = ax_vortex.scatter([], [], c='#00ffff', s=10, marker='^', alpha=0.85)
scat_neg = ax_vortex.scatter([], [], c='#ff4444', s=10, marker='v', alpha=0.85)

ax_phase.set_title('Phase field  Θ(x,y)', color='white', fontsize=13, pad=8)
ax_phase.axis('off')
ax_vortex.set_xlim(-0.5, N-0.5)
ax_vortex.set_ylim(-0.5, N-0.5)
ax_vortex.set_aspect('equal')
ax_vortex.axis('off')
title_obj = fig.suptitle('', fontsize=14, fontweight='bold', y=0.96)

def render_frame(idx):
    d = frames_data[idx]
    im.set_data(d['theta'])

    pos_xy = d['pos'][:, ::-1] if len(d['pos']) > 0 else np.empty((0, 2))
    neg_xy = d['neg'][:, ::-1] if len(d['neg']) > 0 else np.empty((0, 2))
    scat_pos.set_offsets(pos_xy)
    scat_neg.set_offsets(neg_xy)

    T = d['T']
    if T > T_BKT * 1.1:
        label, color = "DISORDERED  (T > T_BKT)", '#ff6666'
    elif T > T_BKT * 0.9:
        label, color = "≈ CRITICAL  (T ≈ T_BKT)", '#ffcc00'
    else:
        label, color = "TOPOLOGICALLY LOCKED  (T < T_BKT)", '#00ff88'

    title_obj.set_text(
        f"ESFT  BKT Topological Phase Transition\n"
        f"T = {T:.3f}    T_BKT = {T_BKT:.3f}    "
        f"+{d['n_pos']}  −{d['n_neg']} vortices    {label}"
    )
    title_obj.set_color(color)

    ax_vortex.set_title(f'Vortex charges  (+{d["n_pos"]} / −{d["n_neg"]})',
                        color='white', fontsize=13, pad=8)
    return im, scat_pos, scat_neg

ani = animation.FuncAnimation(fig, render_frame, frames=N_FRAMES, interval=80, blit=False)

out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'bkt_evolution.mp4')
ani.save(out_path, writer='ffmpeg', fps=12, dpi=120,
         savefig_kwargs={'facecolor': '#0a0a0a'})
print(f"\n✅ Saved: {out_path}")
print(f"   {N_FRAMES} frames, {N_FRAMES/12:.1f}s @ 12fps, {N}x{N} lattice")
plt.close()
