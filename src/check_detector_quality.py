"""Verification visuelle : applique le detecteur entraine sur une frame reelle
complete et affiche l'image brute + la heatmap predite + les detections
resultantes, cote a cote."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from real_io import open_zarr_volume
from dl_model import HeatmapUNet3D, DETECTOR_LEVELS
from config import ROOT, TRAIN_DIR, WEIGHTS_PATH

DATASET_PATH = TRAIN_DIR / "44b6_0113de3b.zarr"
FRAME_T = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HeatmapUNet3D(levels=DETECTOR_LEVELS).to(device)
ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Checkpoint charge: epoque {ckpt['epoch']}, val_loss={ckpt.get('val_loss')}")

arr, scale, quantiles = open_zarr_volume(DATASET_PATH)
frame = arr[FRAME_T][:].astype(np.float32)
q_lo, q_hi = quantiles["0.1"], quantiles["0.999"]
norm = np.clip((frame - q_lo) / (q_hi - q_lo), 0.0, 1.0)

with torch.no_grad():
    x = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
    heatmap = model(x)[0, 0].cpu().numpy()

coords = peak_local_max(heatmap, min_distance=4, threshold_abs=0.3)
print(f"{len(coords)} detections (seuil heatmap 0.3)")

proj_raw = frame.max(axis=0)
proj_heat = heatmap.max(axis=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(proj_raw, cmap="gray", vmin=q_lo, vmax=q_hi)
axes[0].set_title(f"Image brute (t={FRAME_T})")
axes[0].axis("off")

axes[1].imshow(proj_heat, cmap="inferno", vmin=0, vmax=1)
axes[1].set_title("Heatmap predite par le modele")
axes[1].axis("off")

axes[2].imshow(proj_raw, cmap="gray", vmin=q_lo, vmax=q_hi)
axes[2].scatter(coords[:, 2], coords[:, 1], s=12, c="cyan", marker="+")
axes[2].set_title(f"Detections ({len(coords)})")
axes[2].axis("off")

plt.tight_layout()
out_path = ROOT / "data" / "dl_detector_check.png"
plt.savefig(out_path, dpi=110)
print(f"Image sauvegardee: {out_path}")
