# ═══════════════════════════════════════════════════════════
# CELL 1 — Mount Drive and Install Libraries
# ═══════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

!pip install torch torchvision tqdm scikit-learn scipy \
             matplotlib seaborn numpy -q

print("✅ Libraries ready")

# ═══════════════════════════════════════════════════════════
# CELL 2 — Extract Dataset
# ═══════════════════════════════════════════════════════════
import zipfile, os

ZIP_PATH     = '/content/drive/MyDrive/radar_project/Data.zip'
EXTRACT_PATH = '/content/ieee_data/'

os.makedirs(EXTRACT_PATH, exist_ok=True)

if not os.listdir(EXTRACT_PATH):
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(EXTRACT_PATH)
    print("Done.")

# Show what was extracted
for root, dirs, files in os.walk(EXTRACT_PATH):
    for f in files:
        if f.endswith('.pkl'):
            path = os.path.join(root, f)
            size = os.path.getsize(path) / (1024**2)
            print(f"{path}  ({size:.1f} MB)")

# ═══════════════════════════════════════════════════════════
# CELL 3 — Inspect Raw Data Structure
# ═══════════════════════════════════════════════════════════
import pickle
import numpy as np
import glob

pkl_files = glob.glob(EXTRACT_PATH + '**/*.pkl', recursive=True)
print(f"Total .pkl files found: {len(pkl_files)}")

# Inspect first file
with open(pkl_files[0], 'rb') as f:
    sample = pickle.load(f)

arr = np.array(sample)
print(f"\nFile: {pkl_files[0]}")
print(f"Shape : {arr.shape}")
print(f"Dtype : {arr.dtype}")
print(f"Is complex: {np.iscomplexobj(arr)}")
print(f"Max amplitude: {np.max(np.abs(arr)):.4f}")
print(f"Min amplitude: {np.min(np.abs(arr)):.4f}")


# ═══════════════════════════════════════════════════════════
# CELL 4 — Build Micro-Doppler Spectrograms From Raw Data
# Time-frequency spectrograms
# ═══════════════════════════════════════════════════════════
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

def range_doppler_to_spectrogram(rd_sequence, img_size=64):
    """
    Convert a sequence of Range-Doppler frames to a
    micro-Doppler spectrogram.

    Their code: uses raw complex RD map per frame directly
    Your code:  extracts slow-time Doppler signal from the
                dominant range gate and applies STFT to get
                the time-frequency micro-Doppler signature

    Args:
        rd_sequence: complex array (N_frames × range_bins × doppler_bins)
        img_size: output spectrogram size (img_size × img_size)
    Returns:
        spectrogram: float32 array (img_size × img_size)
    """
    # Step 1: Find dominant range gate across all frames
    mean_power    = np.mean(np.abs(rd_sequence)**2, axis=(0, 2))
    best_gate     = np.argmax(mean_power)

    # Step 2: Extract slow-time Doppler signal at that gate
    # Shape: (N_frames × doppler_bins) → flatten to 1D time series
    doppler_seq   = rd_sequence[:, best_gate, :]        # (N_frames, doppler_bins)

    # Step 3: Sum across Doppler bins (coherent integration)
    # This gives one complex value per frame — the slow-time signal
    slow_time_sig = np.sum(doppler_seq, axis=1)         # (N_frames,)

    # Step 4: Remove DC clutter
    slow_time_sig = slow_time_sig - np.mean(slow_time_sig)

    # Step 5: STFT → micro-Doppler spectrogram
    N = len(slow_time_sig)
    nperseg  = min(32, N - 1)
    noverlap = int(nperseg * 0.75)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    _, _, Zxx = stft(
        slow_time_sig,
        fs=1.0,                     # normalized frequency
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False
    )

    # Step 6: Power spectrogram in dB
    Zxx  = np.fft.fftshift(Zxx, axes=0)
    spec = np.abs(Zxx)**2
    spec = 10 * np.log10(spec + 1e-10)

    # Step 7: Normalize to [0, 1]
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-10)

    # Step 8: Resize to fixed img_size × img_size
    from scipy.ndimage import zoom
    zr   = img_size / spec.shape[0]
    zc   = img_size / spec.shape[1]
    spec = zoom(spec, (zr, zc), order=1)

    return spec.astype(np.float32)


def visualize_spectrograms(drone_spec, bird_spec):
    """Plot drone vs bird micro-Doppler spectrogram side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axes[0].imshow(drone_spec, aspect='auto',
                          cmap='jet', origin='lower')
    axes[0].set_title('DRONE — Micro-Doppler Spectrogram',
                       fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Time frames')
    axes[0].set_ylabel('Doppler Frequency')
    plt.colorbar(im1, ax=axes[0], label='Normalized Power')

    im2 = axes[1].imshow(bird_spec, aspect='auto',
                          cmap='jet', origin='lower')
    axes[1].set_title('BIRD — Micro-Doppler Spectrogram',
                       fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Time frames')
    axes[1].set_ylabel('Doppler Frequency')
    plt.colorbar(im2, ax=axes[1], label='Normalized Power')

    plt.tight_layout()
    plt.savefig('drone_vs_bird_spectrogram.png', dpi=150)
    plt.show()

print("✅ Spectrogram functions defined")


# ═══════════════════════════════════════════════════════════
# CELL 5 — Load All Data and Generate Spectrograms
# ═══════════════════════════════════════════════════════════
from tqdm import tqdm

IMG_SIZE   = 64     # spectrogram resolution
CROP_R_S   = 60     # range start  (matches their crop logic)
CROP_R_E   = 80     # range end    → 20 range bins
CROP_D_S   = 60     # doppler start
CROP_D_E   = 90     # doppler end  → 30 doppler bins

spectrograms = []
labels       = []
filenames    = []

drone_specs_for_viz = None
bird_specs_for_viz  = None

print("Generating micro-Doppler spectrograms...")
for fpath in tqdm(pkl_files):
    fname = os.path.basename(fpath).lower()

    # Determine label from filename (same logic as their cut_drone_bird.py)
    if 'bird' in fname:
        label = 0
    elif 'mavik' in fname or 'p3p' in fname or 'drone' in fname:
        label = 1
    else:
        continue   # skip unknown files

    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    data = np.array(data)

    # Apply same crop as original code
    if data.ndim == 3 and data.shape[1] >= CROP_R_E \
                       and data.shape[2] >= CROP_D_E:
        data = data[:, CROP_R_S:CROP_R_E, CROP_D_S:CROP_D_E]
    elif data.ndim == 2:
        # single frame — add batch dim
        data = data[np.newaxis, :, :]

    # Generate spectrogram from the sequence
    try:
        spec = range_doppler_to_spectrogram(data, img_size=IMG_SIZE)
    except Exception as e:
        print(f"  Skipped {fname}: {e}")
        continue

    spectrograms.append(spec)
    labels.append(label)
    filenames.append(fname)

    # Save one of each for visualization
    if label == 1 and drone_specs_for_viz is None:
        drone_specs_for_viz = spec
    if label == 0 and bird_specs_for_viz is None:
        bird_specs_for_viz  = spec

# Convert to arrays
X = np.array(spectrograms)[:, np.newaxis, :, :]  # (N, 1, 64, 64)
y = np.array(labels)

print(f"\n✅ Dataset ready")
print(f"Total samples   : {len(X)}")
print(f"Drones (label=1): {sum(y==1)}")
print(f"Birds  (label=0): {sum(y==0)}")
print(f"Spectrogram shape: {X.shape}")

# Visualize
if drone_specs_for_viz is not None and bird_specs_for_viz is not None:
    visualize_spectrograms(drone_specs_for_viz, bird_specs_for_viz)


# ═══════════════════════════════════════════════════════════
# CELL 6 — Define YOUR CNN Model (ResNet-style)
# Complete architecture:
# - Squeeze-and-Excitation attention blocks
# - Global Average Pooling instead of Flatten (more robust)
# - Deeper residual connections
# - Dropout for regularization
# ═══════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Recalibrates channel-wise features adaptively.
    NOT present in the original repository code.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResBlock(nn.Module):
    """Residual block with SE attention."""
    def __init__(self, channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se      = SEBlock(channels)
        self.relu    = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) * self.se(x) + x)


class MicroDopplerCNN(nn.Module):
    """
    CNN for micro-Doppler spectrogram classification.

    KEY DIFFERENCES from original Res_Convnet:
    1. Input: 64×64 spectrogram vs their 20×30 raw RD map
    2. SE attention blocks for channel recalibration
    3. Global Average Pooling (no fixed flatten size)
    4. Dropout regularization throughout
    5. Trained on STFT spectrograms, not raw complex values
    """
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)                    # 64→32
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            ResBlock(32, dropout=0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()                          # 32→16
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ResBlock(64, dropout=0.2),
            ResBlock(64, dropout=0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()                          # 16→8
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ResBlock(128, dropout=0.3),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()                          # 8→4
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # Global Average Pool → (N,256,1,1)
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.classifier(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Test model shape
model_test = MicroDopplerCNN().to(device)
dummy      = torch.randn(4, 1, IMG_SIZE, IMG_SIZE).to(device)
out        = model_test(dummy)
print(f"Input shape  : {dummy.shape}")
print(f"Output shape : {out.shape}  ← (batch, 2 classes)")
del model_test, dummy, out
print("✅ Model architecture verified")


# ═══════════════════════════════════════════════════════════
# CELL 7 — Training Setup
# ═══════════════════════════════════════════════════════════
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, f1_score)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SpectrogramDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X       = torch.FloatTensor(X)
        self.y       = torch.LongTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            # Time flip (horizontal mirror)
            if torch.rand(1) > 0.5:
                x = torch.flip(x, dims=[2])
            # Add Gaussian noise
            x = x + torch.randn_like(x) * 0.02
            # Random brightness shift
            x = x + (torch.rand(1) - 0.5) * 0.1
            x = torch.clamp(x, 0, 1)
        return x, self.y[idx]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (pred.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs = []
    all_preds = []
    all_labels= []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred   = model(xb)
        loss   = criterion(pred, yb)
        probs  = torch.softmax(pred, dim=1)[:, 1]
        total_loss += loss.item() * len(yb)
        correct    += (pred.argmax(1) == yb).sum().item()
        total      += len(yb)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(pred.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    return (total_loss/total, correct/total,
            np.array(all_preds), np.array(all_probs),
            np.array(all_labels))

print("✅ Training utilities defined")

# ═══════════════════════════════════════════════════════════
# CELL 8 IMPROVED — Fixed Training for Small Dataset
# Key changes:
# 1. 3-fold CV instead of 5-fold (larger training set per fold)
# 2. Smaller batch size (8) → more gradient updates per epoch
# 3. More epochs (80) with longer patience (15)
# 4. Stronger augmentation to compensate for small dataset
# 5. Lower LR with warmup
# ═══════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, f1_score)

EPOCHS    = 80     # increased from 40
LR        = 3e-4   # slightly higher
BATCH     = 8      # reduced from 32 → more updates per epoch
PATIENCE  = 15     # increased from 8
N_FOLDS   = 3      # reduced from 5 → larger training set per fold

cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
    print(f"\n{'─'*55}")
    print(f"  FOLD {fold+1}/{N_FOLDS}  |  "
          f"Train={len(tr_idx)}  Val={len(val_idx)}")
    print(f"{'─'*55}")

    # Datasets with stronger augmentation
    tr_ds  = SpectrogramDataset(X[tr_idx], y[tr_idx], augment=True)
    val_ds = SpectrogramDataset(X[val_idx], y[val_idx], augment=False)

    # Weighted sampler for class balance
    counts  = np.bincount(y[tr_idx])
    weights = 1.0 / counts[y[tr_idx]]
    sampler = WeightedRandomSampler(weights, len(weights)*3,
                                    replacement=True)  # oversample 3x

    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH,
                        sampler=sampler, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH,
                        shuffle=False, num_workers=2)

    print(f"  Batches per epoch: {len(tr_dl)}")

    # Model
    model     = MicroDopplerCNN(num_classes=2, dropout=0.4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=1e-3)

    # Cosine annealing with warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=20, T_mult=2)

    # Class-weighted loss for imbalance
    class_counts = np.bincount(y[tr_idx])
    class_weights = torch.FloatTensor(
        [1.0/class_counts[0], 1.0/class_counts[1]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1  = 0      # use F1 as best metric, not accuracy
    best_val_acc = 0
    best_state   = None
    patience_ctr = 0
    history      = {'tr_loss':[], 'tr_acc':[],
                    'val_loss':[], 'val_acc':[], 'val_f1':[]}

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_one_epoch(
            model, tr_dl, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_probs, val_labels = evaluate(
            model, val_dl, criterion, device)
        scheduler.step()

        # Compute F1 for this epoch
        try:
            ep_f1 = f1_score(val_labels, val_preds, zero_division=0)
        except:
            ep_f1 = 0.0

        history['tr_loss'].append(tr_loss)
        history['tr_acc'].append(tr_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(ep_f1)

        # Save best by F1 (better metric for imbalanced data)
        if ep_f1 > best_val_f1 or \
           (ep_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1  = ep_f1
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            best_preds   = val_preds.copy()
            best_probs   = val_probs.copy()
            best_labels  = val_labels.copy()
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch+1) % 10 == 0:
            print(f"  Ep {epoch+1:02d} | "
                  f"tr_loss={tr_loss:.3f} tr_acc={tr_acc:.3f} | "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} "
                  f"val_f1={ep_f1:.3f}")

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

    torch.save(best_state, f'best_model_fold{fold+1}.pth')

    try:
        auc = roc_auc_score(best_labels, best_probs)
    except:
        auc = 0.0

    fold_results.append({
        'fold'    : fold+1,
        'accuracy': best_val_acc,
        'f1'      : best_val_f1,
        'auc'     : auc,
        'preds'   : best_preds,
        'probs'   : best_probs,
        'labels'  : best_labels,
        'history' : history
    })

    print(f"\n  ✅ Fold {fold+1} Best → "
          f"Acc={best_val_acc:.3f}  "
          f"F1={best_val_f1:.3f}  "
          f"AUC={auc:.3f}")

    # Print what the model predicted this fold
    print(f"  Pred distribution: "
          f"Bird={sum(best_preds==0)}  "
          f"Drone={sum(best_preds==1)}")

print("\n✅ Training complete")

# ═══════════════════════════════════════════════════════════
# CELL 9 — Final Results and Plots
# ═══════════════════════════════════════════════════════════
accs = [r['accuracy'] for r in fold_results]
f1s  = [r['f1']       for r in fold_results]
aucs = [r['auc']      for r in fold_results]

print("\n" + "═"*55)
print("  MICRO-DOPPLER CNN — FINAL RESULTS")
print("═"*55)
print(f"  Dataset    : IEEE 60GHz FMCW Radar")
print(f"  Model      : MicroDopplerCNN (SE-ResNet)")
print(f"  Input      : 64×64 STFT Spectrogram")
print(f"  CV         : Stratified 5-Fold")
print(f"  ─────────────────────────────────────────")
for r in fold_results:
    print(f"  Fold {r['fold']}: "
          f"Acc={r['accuracy']:.3f}  "
          f"F1={r['f1']:.3f}  "
          f"AUC={r['auc']:.3f}")
print(f"  ─────────────────────────────────────────")
print(f"  Mean Acc : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
print(f"  Mean F1  : {np.mean(f1s):.3f}  ± {np.std(f1s):.3f}")
print(f"  Mean AUC : {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print("═"*55)

# ── Plot 1: Training curves for each fold ─────────────────
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, r in enumerate(fold_results):
    h = r['history']
    axes[0][i].plot(h['tr_loss'],  label='Train', color='steelblue')
    axes[0][i].plot(h['val_loss'], label='Val',   color='darkorange')
    axes[0][i].set_title(f"Fold {i+1} — Loss")
    axes[0][i].legend(fontsize=8)
    axes[0][i].grid(True, alpha=0.3)

    axes[1][i].plot(h['tr_acc'],  color='steelblue')
    axes[1][i].plot(h['val_acc'], color='darkorange')
    axes[1][i].set_title(f"Fold {i+1} — Accuracy")
    axes[1][i].set_ylim(0, 1.05)
    axes[1][i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=120)
plt.show()

# ── Plot 2: Best fold — Confusion Matrix + ROC ────────────
best_fold = fold_results[np.argmax(accs)]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm = confusion_matrix(best_fold['labels'], best_fold['preds'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bird', 'Drone'],
            yticklabels=['Bird', 'Drone'],
            ax=axes[0], linewidths=2, annot_kws={'size':14})
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True',      fontsize=12)
axes[0].set_title(f'Confusion Matrix — Fold {best_fold["fold"]}',
                  fontsize=13, fontweight='bold')

fpr, tpr, _ = roc_curve(best_fold['labels'], best_fold['probs'])
axes[1].plot(fpr, tpr, color='darkorange', lw=2,
             label=f'AUC = {best_fold["auc"]:.3f}')
axes[1].plot([0,1],[0,1], 'navy', lw=1, linestyle='--')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='orange')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate',  fontsize=12)
axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confusion_roc.png', dpi=150)
plt.show()

# ── Plot 3: Per-fold bar chart ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x    = np.arange(len(fold_results))
w    = 0.25
ax.bar(x-w, accs, w, label='Accuracy', color='steelblue',    alpha=0.85)
ax.bar(x,   f1s,  w, label='F1 Score', color='darkorange',   alpha=0.85)
ax.bar(x+w, aucs, w, label='ROC-AUC',  color='forestgreen',  alpha=0.85)
ax.axhline(np.mean(accs), color='steelblue',   linestyle='--', alpha=0.6)
ax.axhline(np.mean(f1s),  color='darkorange',  linestyle='--', alpha=0.6)
ax.axhline(np.mean(aucs), color='forestgreen', linestyle='--', alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(len(fold_results))])
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Fold Performance — Micro-Doppler CNN',
             fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('per_fold_results.png', dpi=150)
plt.show()

print("\n✅ All plots saved")
print(f"Classification Report (best fold):")
print(classification_report(best_fold['labels'], best_fold['preds'],
      target_names=['Bird', 'Drone']))