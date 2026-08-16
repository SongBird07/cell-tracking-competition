"""Script d'exploration rapide : teste quelques jeux de parametres de detection
sur UNE frame reelle et sauvegarde une image avec les points detectes en
surimpression, pour choisir des parametres raisonnables avant de lancer sur
tout le dataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from real_io import open_zarr_volume
from config import ROOT, TRAIN_DIR

DATASET_PATH = TRAIN_DIR / "44b6_0113de3b.zarr"

arr, scale, quantiles = open_zarr_volume(DATASET_PATH)
frame = arr[0][:].astype(np.float32)  # (Z, Y, X)
proj = frame.max(axis=0)

configs = [
    dict(sigma=(1.0, 1.8, 1.8), min_distance=3, threshold_abs=quantiles["0.9"]),
    dict(sigma=(1.0, 1.8, 1.8), min_distance=4, threshold_abs=quantiles["0.99"]),
    dict(sigma=(1.5, 2.5, 2.5), min_distance=5, threshold_abs=quantiles["0.9"]),
]

fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 6))
for ax, cfg in zip(axes, configs):
    smoothed = gaussian(frame, sigma=cfg["sigma"], preserve_range=True)
    coords = peak_local_max(smoothed, min_distance=cfg["min_distance"], threshold_abs=cfg["threshold_abs"])
    ax.imshow(proj, cmap="gray", vmin=quantiles["0.1"], vmax=quantiles["0.999"])
    ax.scatter(coords[:, 2], coords[:, 1], s=12, c="red", marker="+")
    ax.set_title(f"sigma={cfg['sigma']} min_dist={cfg['min_distance']}\nthr={cfg['threshold_abs']:.0f} -> {len(coords)} detections")
    ax.axis("off")

plt.tight_layout()
out_path = ROOT / "data" / "real_detect_tuning.png"
plt.savefig(out_path, dpi=110)
print(f"Image sauvegardee: {out_path}")
