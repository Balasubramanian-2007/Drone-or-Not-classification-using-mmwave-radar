# extract_my_spectrograms.py
# Converts YOUR .bin files to 64x64 spectrograms
# Same pipeline as raw_adc_to_prediction.py
# These become your fine-tuning dataset

import numpy as np
import torch
import os
import glob
import pickle
from scipy.signal import stft
from scipy.ndimage import zoom

RADAR_CONFIG = {
    'num_adc_samples': 256,
    'num_chirps'     : 128,
    'num_rx'         : 4,
}

def bin_to_spectrogram(bin_path, img_size=64):
    """
    Convert one .bin file to one 64×64 spectrogram.
    Exact same pipeline as predict_from_bin().
    """
    # Parse
    with open(bin_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.int16)
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    IQ_flat = I + 1j * Q

    spc    = RADAR_CONFIG['num_adc_samples'] * \
             RADAR_CONFIG['num_rx']
    chirps = len(IQ_flat) // spc
    IQ     = IQ_flat[:chirps*spc].reshape(
             chirps,
             RADAR_CONFIG['num_adc_samples'],
             RADAR_CONFIG['num_rx'])

    # DC removal + Range FFT
    IQ_comb = np.sum(IQ, axis=2)
    IQ_comb = IQ_comb - np.mean(IQ_comb,
                                  axis=0, keepdims=True)
    hann    = np.hanning(RADAR_CONFIG['num_adc_samples'])
    rf      = np.fft.fft(IQ_comb * hann[np.newaxis,:],
                          axis=1)
    rf      = rf[:, :RADAR_CONFIG['num_adc_samples']//2]

    # Doppler FFT → frames
    chirps_per_frame = RADAR_CONFIG['num_chirps']
    num_frames       = chirps // chirps_per_frame
    hann_slow        = np.hanning(chirps_per_frame)
    rd_sequence      = []

    for i in range(num_frames):
        s   = i * chirps_per_frame
        e   = s + chirps_per_frame
        frm = rf[s:e, :]
        frm = frm * hann_slow[:, np.newaxis]
        dft = np.fft.fft(frm, axis=0)
        dft = np.fft.fftshift(dft, axes=0)
        rd_sequence.append(dft.T)

    rd_sequence = np.array(rd_sequence)

    # Smart crop with DC suppression
    mean_power          = np.mean(np.abs(rd_sequence)**2,
                                   axis=0)
    mean_power[:15, :]  = 0
    mean_power[-5:, :]  = 0
    mean_power[:, 60:68]= 0

    peak     = np.unravel_index(np.argmax(mean_power),
                                 mean_power.shape)
    peak_r   = peak[0]
    peak_d   = peak[1]

    rs = max(0,  peak_r - 10)
    re = min(rd_sequence.shape[1], rs + 20)
    ds = max(0,  peak_d - 15)
    de = min(rd_sequence.shape[2], ds + 30)

    if re - rs < 20: rs = max(0, re - 20)
    if de - ds < 30: ds = max(0, de - 30)

    rd_crop = rd_sequence[:, rs:re, ds:de]

    # SNR check
    peak_pwr  = float(np.max(mean_power))
    noise_flr = float(np.median(
        mean_power[mean_power > 0]))
    snr_db    = 10 * np.log10(
        peak_pwr / (noise_flr + 1e-10))

    # Spectrogram via STFT
    mp        = np.mean(np.abs(rd_crop)**2, axis=(0,2))
    best_gate = int(np.argmax(mp))
    sig       = rd_crop[:, best_gate, :]
    sig       = np.sum(sig, axis=1)
    sig       = sig - np.mean(sig)

    N        = len(sig)
    nperseg  = min(32, N-1)
    noverlap = int(nperseg * 0.75)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    _, _, Zxx = stft(sig, fs=1.0, window='hann',
                     nperseg=nperseg, noverlap=noverlap,
                     return_onesided=False)
    Zxx  = np.fft.fftshift(Zxx, axes=0)
    spec = np.abs(Zxx)**2
    spec = 10 * np.log10(spec + 1e-10)
    spec = (spec - spec.min()) / \
           (spec.max() - spec.min() + 1e-10)

    zr   = img_size / spec.shape[0]
    zc   = img_size / spec.shape[1]
    spec = zoom(spec, (zr, zc), order=1)

    return spec.astype(np.float32), snr_db, num_frames


# ── Process your files ────────────────────────────────────
MY_FILES = {
    # Drone captures → label 1
    r'd:\Mistral_BirdvsDrone\new\adc_data.bin': 1,

    # Not-drone captures → label 0
    r'd:\Mistral_BirdvsDrone\new\adc_data_plain.bin': 0,
    r'd:\Mistral_BirdvsDrone\new\adc_data_drone_static.bin': 0,
    r'd:\Mistral_BirdvsDrone\new\adc_data_original_drone.bin': 0,
    r'd:\Mistral_BirdvsDrone\new\adc_data_bala1.bin': 0,
    r'd:\Mistral_BirdvsDrone\new\adc_data_bala2.bin': 0,
}

spectrograms = []
labels       = []
filenames    = []
snrs         = []

print("── Generating Spectrograms from Your Radar ──")
for fpath, label in MY_FILES.items():
    if not os.path.exists(fpath):
        print(f"  MISSING: {fpath}")
        continue
    try:
        spec, snr, frames = bin_to_spectrogram(fpath)
        spectrograms.append(spec)
        labels.append(label)
        filenames.append(os.path.basename(fpath))
        snrs.append(snr)
        print(f"  ✅ {os.path.basename(fpath):<40} "
              f"SNR={snr:.1f}dB  frames={frames} "
              f"label={'DRONE' if label==1 else 'NOT-DRONE'}")
    except Exception as e:
        print(f"  ❌ {os.path.basename(fpath)}: {e}")

X = np.array(spectrograms)
y = np.array(labels)

print(f"\nTotal spectrograms : {len(X)}")
print(f"Drone   (1)        : {sum(y==1)}")
print(f"Not-drone (0)      : {sum(y==0)}")
print(f"Shape              : {X.shape}")

# Save for fine-tuning
np.save('my_spectrograms.npy', X)
np.save('my_labels.npy', y)
np.save('my_snrs.npy', np.array(snrs))
print("\n✅ Saved: my_spectrograms.npy, my_labels.npy")

# Visualize one of each
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

drone_idx    = next(i for i,l in enumerate(labels) if l==1)
notdrone_idx = next(i for i,l in enumerate(labels) if l==0)

axes[0].imshow(spectrograms[drone_idx],
               aspect='auto', cmap='jet', origin='lower')
axes[0].set_title(f'DRONE Spectrogram\n'
                   f'{filenames[drone_idx]} '
                   f'(SNR={snrs[drone_idx]:.1f}dB)',
                   fontweight='bold')
axes[0].set_xlabel('Time Frames')
axes[0].set_ylabel('Doppler Frequency')

axes[1].imshow(spectrograms[notdrone_idx],
               aspect='auto', cmap='jet', origin='lower')
axes[1].set_title(f'NOT-DRONE Spectrogram\n'
                   f'{filenames[notdrone_idx]} '
                   f'(SNR={snrs[notdrone_idx]:.1f}dB)',
                   fontweight='bold')
axes[1].set_xlabel('Time Frames')
axes[1].set_ylabel('Doppler Frequency')

plt.tight_layout()
plt.savefig('my_radar_spectrograms.png', dpi=150)
plt.show()