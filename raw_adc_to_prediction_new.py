import numpy as np
import torch
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.ndimage import zoom

# ── RADAR CONFIGURATION ───────────────────────────────────
RADAR_CONFIG = {
    'num_adc_samples'  : 256,
    'num_chirps'       : 128,
    'num_rx'           : 4,
    'num_tx'           : 3,
    'sample_rate_ksps' : 10000,
    'freq_slope_mhz_us': 29.982,
    'idle_time_us'     : 100,
    'ramp_end_time_us' : 60,
    'start_freq_ghz'   : 60,
    'frame_period_ms'  : 50,
}

RADAR_CONFIG['bandwidth_mhz'] = (
    RADAR_CONFIG['freq_slope_mhz_us'] *
    RADAR_CONFIG['num_adc_samples'] /
    RADAR_CONFIG['sample_rate_ksps'] * 1000
)
RADAR_CONFIG['range_res_m'] = (
    3e8 / (2 * RADAR_CONFIG['bandwidth_mhz'] * 1e6)
)
RADAR_CONFIG['max_range_m'] = (
    RADAR_CONFIG['range_res_m'] *
    RADAR_CONFIG['num_adc_samples']
)

print("── Radar Configuration ──")
for k, v in RADAR_CONFIG.items():
    print(f"  {k:<25}: {v}")


# ═══════════════════════════════════════════════════════════
# PART 1: PARSE — FIXED VERSION
# Key fix: do NOT use num_tx to calculate expected size
# DCA1000 stores all TX chirps sequentially, not interleaved
# Calculate actual chirps directly from file size
# ═══════════════════════════════════════════════════════════

def parse_dca1000_bin(bin_path, config):
    num_samples = config['num_adc_samples']
    num_rx      = config['num_rx']

    with open(bin_path, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.int16)

    print(f"\n── DCA1000 Parse ──")
    print(f"  File              : {os.path.basename(bin_path)}")
    print(f"  Raw int16 samples : {len(raw)}")

    # Separate I and Q
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    IQ_flat = I + 1j * Q

    total_complex     = len(IQ_flat)
    samples_per_chirp = num_samples * num_rx

    # Calculate actual chirps from real file size
    actual_chirps = total_complex // samples_per_chirp
    usable        = actual_chirps * samples_per_chirp

    print(f"  Total complex     : {total_complex}")
    print(f"  Samples per chirp : {samples_per_chirp} "
          f"({num_samples} ADC × {num_rx} RX)")
    print(f"  Actual chirps     : {actual_chirps}")
    print(f"  Discarded samples : {total_complex - usable}")

    IQ = IQ_flat[:usable].reshape(actual_chirps, num_samples, num_rx)
    print(f"  IQ shape          : {IQ.shape} "
          f"(chirps × samples × rx)")
    return IQ


# ═══════════════════════════════════════════════════════════
# PART 2: RANGE-DOPPLER COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_range_doppler(IQ, config):
    num_chirps_total = IQ.shape[0]
    num_samples      = IQ.shape[1]
    chirps_per_frame = config['num_chirps']

    # Sum across RX for better SNR
    IQ_combined = np.sum(IQ, axis=2)

    # ── ADD THIS: Remove DC offset per chirp ─────────────
    # Subtracting mean removes static clutter contribution
    IQ_combined = IQ_combined - \
                  np.mean(IQ_combined, axis=0, keepdims=True)
    # ─────────────────────────────────────────────────────

    # Hann window along fast-time
    hann_fast   = np.hanning(num_samples)
    IQ_windowed = IQ_combined * hann_fast[np.newaxis, :]

    # Range FFT
    range_fft = np.fft.fft(IQ_windowed, axis=1)
    range_fft = range_fft[:, :num_samples // 2]

    # Split into frames
    num_frames = num_chirps_total // chirps_per_frame

    print(f"\n── Range-Doppler Computation ──")
    print(f"  Total chirps      : {num_chirps_total}")
    print(f"  Chirps per frame  : {chirps_per_frame}")
    print(f"  Frames available  : {num_frames}")

    hann_slow   = np.hanning(chirps_per_frame)
    rd_sequence = []

    for frame_idx in range(num_frames):
        start        = frame_idx * chirps_per_frame
        end          = start + chirps_per_frame
        frame_range  = range_fft[start:end, :]
        frame_win    = frame_range * hann_slow[:, np.newaxis]
        doppler_fft  = np.fft.fft(frame_win, axis=0)
        doppler_fft  = np.fft.fftshift(doppler_fft, axes=0)
        rd_sequence.append(doppler_fft.T)

    rd_sequence = np.array(rd_sequence)
    print(f"  Frames computed   : {len(rd_sequence)}")
    print(f"  RD map shape      : {rd_sequence.shape}")
    return rd_sequence


# ═══════════════════════════════════════════════════════════
# PART 3: SMART CROP
# ═══════════════════════════════════════════════════════════

def smart_crop(data, range_size=20, doppler_size=30):
    mean_power = np.mean(np.abs(data)**2, axis=0)

    # Suppress DC clutter — first 15 range bins are
    # near-field/DC contamination for this radar config
    mean_power[:15, :] = 0      # suppress near bins
    mean_power[-5:, :] = 0      # suppress aliased bins

    # Also suppress zero-Doppler clutter line
    # bin 64 = zero velocity = static clutter
    mean_power[:, 60:68] = 0    # suppress zero-Doppler

    peak_loc     = np.unravel_index(np.argmax(mean_power),
                                     mean_power.shape)
    peak_range   = peak_loc[0]
    peak_doppler = peak_loc[1]

    r_start = max(0, peak_range   - range_size  // 2)
    r_end   = min(data.shape[1],  r_start + range_size)
    d_start = max(0, peak_doppler - doppler_size // 2)
    d_end   = min(data.shape[2],  d_start + doppler_size)

    if r_end - r_start < range_size:
        r_start = max(0, r_end - range_size)
    if d_end - d_start < doppler_size:
        d_start = max(0, d_end - doppler_size)

    cropped = data[:, r_start:r_end, d_start:d_end]

    print(f"\n── Smart Crop ──")
    print(f"  Peak range bin    : {peak_range} "
          f"(~{peak_range * RADAR_CONFIG['range_res_m']:.2f}m)")
    print(f"  Peak doppler bin  : {peak_doppler}")
    print(f"  Crop range        : [{r_start}:{r_end}]")
    print(f"  Crop doppler      : [{d_start}:{d_end}]")
    print(f"  Cropped shape     : {cropped.shape}")
    return cropped, peak_range, peak_doppler

# ═══════════════════════════════════════════════════════════
# PART 4: SPECTROGRAM GENERATION
# ═══════════════════════════════════════════════════════════

def range_doppler_to_spectrogram(rd_sequence, img_size=64):
    mean_power    = np.mean(np.abs(rd_sequence)**2, axis=(0, 2))
    best_gate     = np.argmax(mean_power)
    doppler_seq   = rd_sequence[:, best_gate, :]
    slow_time_sig = np.sum(doppler_seq, axis=1)
    slow_time_sig = slow_time_sig - np.mean(slow_time_sig)

    N        = len(slow_time_sig)
    nperseg  = min(32, N - 1)
    noverlap = int(nperseg * 0.75)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    print(f"\n── Spectrogram ──")
    print(f"  Frames (N)        : {N}")
    print(f"  Best range gate   : {best_gate}")
    print(f"  nperseg           : {nperseg}")
    print(f"  noverlap          : {noverlap}")

    _, _, Zxx = stft(slow_time_sig, fs=1.0,
                     window='hann', nperseg=nperseg,
                     noverlap=noverlap, return_onesided=False)

    Zxx  = np.fft.fftshift(Zxx, axes=0)
    spec = np.abs(Zxx)**2
    spec = 10 * np.log10(spec + 1e-10)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-10)

    zr   = img_size / spec.shape[0]
    zc   = img_size / spec.shape[1]
    spec = zoom(spec, (zr, zc), order=1)
    print(f"  Spectrogram shape : {spec.shape}")
    return spec.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# PART 5: MODEL
# ═══════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)

class ResBlock(nn.Module):
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se   = SEBlock(channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.block(x) * self.se(x) + x)

class MicroDopplerCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.stage1 = nn.Sequential(
            ResBlock(32, 0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 0.2), ResBlock(64, 0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 0.3),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, 64),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(
            self.stage3(self.stage2(self.stage1(self.stem(x)))))


# ═══════════════════════════════════════════════════════════
# PART 6: LOAD MODEL
# ═══════════════════════════════════════════════════════════

def load_best_model(checkpoint_path, device):
    model = MicroDopplerCNN(num_classes=2, dropout=0.3).to(device)
    state = torch.load(checkpoint_path,
                       map_location=device,
                       weights_only=True)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"\n✅ Model loaded from {checkpoint_path}")
    return model


# ═══════════════════════════════════════════════════════════
# PART 7: FULL PREDICTION PIPELINE
# ═══════════════════════════════════════════════════════════

def cfar_detection(rd_sequence, guard=2, train=8,
                   threshold_factor=15):
    """
    Simple CFAR-like detection to check if a
    real target exists before classification.

    Returns:
        detected: bool — True if real target found
        peak_power: float — power at detected bin
        noise_floor: float — estimated noise floor
        snr_db: float — signal to noise ratio
    """
    # Average power map across all frames
    power_map   = np.mean(np.abs(rd_sequence)**2, axis=0)

    # Suppress DC
    power_map[:15, :] = 0
    power_map[-5:, :] = 0
    power_map[:, 60:68] = 0    # zero Doppler

    # Find peak
    peak_power  = float(np.max(power_map))
    noise_floor = float(np.median(power_map[power_map > 0]))

    snr         = peak_power / (noise_floor + 1e-10)
    snr_db      = 10 * np.log10(snr)

    # Detection threshold — tune this value
    # Higher = fewer false alarms, more misses
    # Lower  = more detections, more false alarms
    THRESHOLD_DB = 13.0   # 13 dB above noise floor

    detected = snr_db > THRESHOLD_DB

    print(f"\n── CFAR Detection ──")
    print(f"  Peak power    : {peak_power:.2e}")
    print(f"  Noise floor   : {noise_floor:.2e}")
    print(f"  SNR           : {snr_db:.1f} dB")
    print(f"  Threshold     : {THRESHOLD_DB} dB")
    print(f"  Target found  : {'✅ YES' if detected else '❌ NO'}")

    return detected, peak_power, noise_floor, snr_db

def predict_from_bin(bin_path, model, device, visualize=True):
    CLASS_NAMES = ['Bird', 'Drone']

    print(f"\n{'═'*55}")
    print(f"  Processing: {os.path.basename(bin_path)}")
    print(f"{'═'*55}")

    # Parse
    IQ = parse_dca1000_bin(bin_path, RADAR_CONFIG)

    # Range-Doppler
    rd_sequence = compute_range_doppler(IQ, RADAR_CONFIG)

    detected, peak_pwr, noise_flr, snr_db = cfar_detection(
        rd_sequence)

    if not detected:
        print(f"\n{'═'*55}")
        print(f"  RESULT       : NO TARGET DETECTED")
        print(f"  SNR          : {snr_db:.1f} dB (below threshold)")
        print(f"  Classification skipped — no real target")
        print(f"{'═'*55}")
        return {
            'prediction' : 'No Target',
            'confidence' : 0.0,
            'prob_bird'  : 0.0,
            'prob_drone' : 0.0,
            'snr_db'     : snr_db,
            'detected'   : False
        }

    if rd_sequence.shape[0] < 5:
        print(f"\n⚠️  Only {rd_sequence.shape[0]} frames.")
        print("   Increase num_frames in mmWave Studio.")
        print("   Minimum 300 frames recommended.")

    # Smart crop
    rd_cropped, peak_r, peak_d = smart_crop(rd_sequence)

    # Spectrogram
    spec = range_doppler_to_spectrogram(rd_cropped, img_size=64)

    # Visualize
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        range_profile = np.mean(np.abs(rd_sequence)**2, axis=(0, 2))
        axes[0].plot(range_profile)
        axes[0].axvline(x=peak_r, color='red', linestyle='--',
                        label=f'Target bin {peak_r}')
        axes[0].set_title('Range Profile')
        axes[0].set_xlabel('Range Bin')
        axes[0].set_ylabel('Power')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].imshow(np.log1p(np.abs(rd_sequence[0])).T,
                       aspect='auto', cmap='jet', origin='lower')
        axes[1].set_title(f'Range-Doppler Map (Frame 1)\n'
                           f'Total frames: {rd_sequence.shape[0]}')
        axes[1].set_xlabel('Range Bin')
        axes[1].set_ylabel('Doppler Bin')

        axes[2].imshow(spec, aspect='auto',
                       cmap='jet', origin='lower')
        axes[2].set_title('Micro-Doppler Spectrogram\n(Model Input)')
        axes[2].set_xlabel('Time Frames')
        axes[2].set_ylabel('Doppler Frequency')

        plt.suptitle(f'Processing: {os.path.basename(bin_path)}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('prediction_pipeline.png', dpi=120)
        plt.show()

    # Predict
    tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred   = int(probs.argmax())

    result = {
        'prediction' : CLASS_NAMES[pred],
        'confidence' : float(probs[pred]) * 100,
        'prob_bird'  : float(probs[0]) * 100,
        'prob_drone' : float(probs[1]) * 100,
        'range_m'    : peak_r * RADAR_CONFIG['range_res_m'],
        'num_frames' : rd_sequence.shape[0],
    }

    print(f"\n{'═'*55}")
    print(f"  PREDICTION   : {result['prediction']} ")
    print(f"  Confidence   : {result['confidence']:.1f}%")
    print(f"  P(Bird)      : {result['prob_bird']:.1f}%")
    print(f"  P(Drone)     : {result['prob_drone']:.1f}%")
    print(f"  Target range : {result['range_m']:.1f}m")
    print(f"  Frames used  : {result['num_frames']}")
    print(f"{'═'*55}")
    # return result
    result['rd_sequence'] = rd_sequence   # expose for verification
    return result


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_best_model(r"d:\Mistral_BirdvsDrone\new\best_model_fold2.pth", device)
    result = predict_from_bin(r"d:\Mistral_BirdvsDrone\new\adc_data_original_drone.bin", model, device,visualize=True)