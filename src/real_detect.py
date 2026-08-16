"""
Detection sur un VRAI dataset (equivalent de detect.py, adapte a l'OME-Zarr et
a des volumes beaucoup plus gros/bruites/denses que les donnees synthetiques).

Parametres choisis apres inspection visuelle (voir real_detect_tune.py) :
  - sigma plus large que sur les donnees synthetiques : les noyaux reels font
    plusieurs dizaines de voxels, pas quelques voxels
  - threshold_abs derive du quantile 0.9 d'intensite DU DATASET LUI-MEME
    (fourni dans les metadonnees zarr), plutot qu'une constante globale --
    chaque video a un niveau d'exposition/bruit different

LIMITE CONNUE : un seuil global ne capture pas bien les cellules dans les
zones sombres de l'image (ex: bord gauche attenue en profondeur). Un vrai
pipeline de competition utiliserait une segmentation par deep learning
(StarDist3D/Cellpose) plutot que ce seuillage global, justement pour ce genre
de cas.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from real_io import open_zarr_volume
from config import ROOT, TRAIN_DIR


def detect_dataset(zarr_path, sigma=(1.5, 2.5, 2.5), min_distance=5, threshold_quantile="0.9", verbose=True):
    arr, scale_zyx, quantiles = open_zarr_volume(zarr_path)
    if threshold_quantile not in quantiles:
        raise KeyError(
            f"Quantile '{threshold_quantile}' absent des metadonnees de {zarr_path} "
            f"(quantiles disponibles: {list(quantiles.keys())})"
        )
    threshold_abs = quantiles[threshold_quantile]
    T = arr.shape[0]

    rows = []
    for t in range(T):
        frame = arr[t][:].astype(np.float32)  # charge UNE SEULE frame en RAM
        smoothed = gaussian(frame, sigma=sigma, preserve_range=True)
        coords = peak_local_max(smoothed, min_distance=min_distance, threshold_abs=threshold_abs)
        for (z, y, x) in coords:
            rows.append({"frame": t, "z": int(z), "y": int(y), "x": int(x)})
        if verbose and (t % 20 == 0 or t == T - 1):
            print(f"  t={t}/{T - 1}: {len(coords)} detections (cumulatif: {len(rows)})")

    df = pd.DataFrame(rows, columns=["frame", "z", "y", "x"])
    return df, scale_zyx


if __name__ == "__main__":
    dataset_name = "44b6_0113de3b"
    zarr_path = TRAIN_DIR / f"{dataset_name}.zarr"

    print(f"Detection sur '{dataset_name}'...")
    df, scale = detect_dataset(zarr_path)

    out_path = ROOT / "data" / f"{dataset_name}_real_detections.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} detections au total ecrites dans {out_path}")
    print(f"Echelle physique (z,y,x) um/voxel: {scale}")
