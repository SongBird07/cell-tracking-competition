"""Projection en intensite max de quelques frames d'un vrai dataset, pour
avoir une idee visuelle avant de choisir les parametres de detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from real_io import open_zarr_volume
from config import ROOT, TRAIN_DIR

DATASET_PATH = TRAIN_DIR / "44b6_0113de3b.zarr"
FRAMES_TO_SHOW = [0, 25, 50, 75, 99]

arr, scale, quantiles = open_zarr_volume(DATASET_PATH)
print("scale:", scale, "quantiles:", quantiles)

vmax = quantiles.get("0.999", 2000)

fig, axes = plt.subplots(1, len(FRAMES_TO_SHOW), figsize=(3.5 * len(FRAMES_TO_SHOW), 3.5))
for ax, t in zip(axes, FRAMES_TO_SHOW):
    frame = arr[t][:]  # (Z, Y, X) -- seule cette frame est chargee en RAM
    proj = frame.max(axis=0)
    ax.imshow(proj, cmap="gray", vmin=quantiles.get("0.1", 0), vmax=vmax)
    ax.set_title(f"t={t}")
    ax.axis("off")

plt.tight_layout()
out_path = ROOT / "data" / "real_sanity_check_projections.png"
plt.savefig(out_path, dpi=110)
print(f"Image sauvegardee: {out_path}")
