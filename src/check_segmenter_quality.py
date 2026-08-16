"""Verification visuelle du segmenteur (train_segmenter.py) : image brute,
carte de probabilite predite, et instances apres watershed, cote a cote.
Equivalent de check_detector_quality.py pour le pipeline segmentation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from real_io import open_zarr_volume
from real_detect_seg import load_model, segment_frame, _pad_to_multiple
from config import ROOT, TRAIN_DIR, WEIGHTS_DIR

DATASET_PATH = TRAIN_DIR / "44b6_0113de3b.zarr"
FRAME_T = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(device)
ckpt = torch.load(WEIGHTS_DIR / "seg_unet.pt", map_location=device, weights_only=False)
print(f"Checkpoint charge: epoque {ckpt['epoch']}, val_loss={ckpt.get('val_loss')}")

arr, scale, quantiles = open_zarr_volume(DATASET_PATH)
frame = arr[FRAME_T][:].astype(np.float32)
q_lo, q_hi = quantiles["0.1"], quantiles["0.999"]
norm = np.clip((frame - q_lo) / (q_hi - q_lo), 0.0, 1.0)
padded, pads = _pad_to_multiple(norm)

with torch.no_grad():
    x = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
    prob_map = model(x)[0, 0].cpu().numpy()
sl = tuple(slice(0, dim) for dim in norm.shape)
prob_map = prob_map[sl]

labels, n_instances = segment_frame(prob_map, prob_threshold=0.5, min_distance=6)
print(f"{n_instances} instances detectees (seuil 0.5)")

proj_raw = frame.max(axis=0)
proj_prob = prob_map.max(axis=0)
proj_labels = labels.max(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(proj_raw, cmap="gray", vmin=q_lo, vmax=q_hi)
axes[0].set_title(f"Image brute (t={FRAME_T})")
axes[0].axis("off")

axes[1].imshow(proj_prob, cmap="inferno", vmin=0, vmax=1)
axes[1].set_title("Probabilite de premier plan predite")
axes[1].axis("off")

axes[2].imshow(proj_raw, cmap="gray", vmin=q_lo, vmax=q_hi)
axes[2].imshow(np.ma.masked_where(proj_labels == 0, proj_labels), cmap="nipy_spectral", alpha=0.5)
axes[2].set_title(f"Instances apres watershed ({n_instances})")
axes[2].axis("off")

plt.tight_layout()
out_path = ROOT / "data" / "seg_detector_check.png"
plt.savefig(out_path, dpi=110)
print(f"Image sauvegardee: {out_path}")
