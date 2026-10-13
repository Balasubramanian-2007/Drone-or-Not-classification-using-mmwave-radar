import numpy as np
import matplotlib.pyplot as plt

# After compute_range_doppler() gives rd_sequence (184, 128, 128)
# Plot first 6 frames to see consistency

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, frame_idx in enumerate([0, 20, 40, 80, 120, 183]):
    rd_frame = np.abs(rd_sequence[frame_idx])
    axes[i].imshow(np.log1p(rd_frame).T,
                   aspect='auto', cmap='jet', origin='lower')
    axes[i].axvline(x=42, color='white', linestyle='--',
                    alpha=0.7, label='Target bin')
    axes[i].axhline(y=64, color='red', linestyle='--',
                    alpha=0.7, label='Zero Doppler')
    axes[i].set_title(f'Frame {frame_idx}')
    axes[i].set_xlabel('Range Bin')
    axes[i].set_ylabel('Doppler Bin')

plt.suptitle('Range-Doppler Maps — Drone Capture\n'
             'Consistent bright spot = real target',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('frame_by_frame_verification.png', dpi=120)
plt.show()