"""Sauvegarde une projection en intensite max (par-dessus l'axe Z) pour quelques
timepoints, histoire de verifier visuellement que les donnees synthetiques
ressemblent a des blobs de cellules qui bougent et se divisent."""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
group = zarr.open_group(str(ROOT / "data" / "synth1.zarr"), mode="r")
volume = group["0"][:]  # (T, Z, Y, X)

T = volume.shape[0]
fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))
for t in range(T):
    proj = volume[t].max(axis=0)  # max projection sur Z -> (Y, X)
    axes[t].imshow(proj, cmap="gray", vmin=0, vmax=255)
    axes[t].set_title(f"t={t}")
    axes[t].axis("off")

plt.tight_layout()
out_path = ROOT / "data" / "sanity_check_projections.png"
plt.savefig(out_path, dpi=110)
print(f"Image sauvegardee: {out_path}")
