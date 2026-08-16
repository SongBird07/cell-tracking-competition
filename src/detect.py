"""
ETAPE 1 : Detection des cellules par frame.

Pour chaque timepoint t du volume (T, Z, Y, X), on :
  1. lisse le volume avec un filtre gaussien (reduit le bruit, evite de detecter
     plusieurs pics dans une meme cellule)
  2. cherche les maxima locaux au-dessus d'un seuil relatif -> ce sont nos
     candidats de centroides de cellules

Sortie : data/synth1_detections.csv avec colonnes [frame, z, y, x]
(coordonnees en VOXELS, comme attendu dans le submission.csv final)
"""

import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from skimage.filters import gaussian
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "synth1"


def detect_cells_blob(volume_2d_or_3d, sigma=1.2, min_distance=3, threshold_abs=45):
    """Detecte les centroides de blobs dans un volume 3D (Z, Y, X).

    On utilise un seuil ABSOLU plutot que relatif au maximum du volume : en zone
    dense (ex: juste apres une division), des cellules qui se chevauchent peuvent
    saturer localement l'intensite, ce qui rend un seuil relatif (threshold_rel)
    trop severe pour les cellules normales ailleurs dans le volume et fait
    disparaitre des detections valides.
    """
    smoothed = gaussian(volume_2d_or_3d.astype(np.float32), sigma=sigma, preserve_range=True)
    coords = peak_local_max(
        smoothed, min_distance=min_distance, threshold_abs=threshold_abs
    )
    return coords  # (N, 3) -> z, y, x


def run_detection(dataset_name):
    store_path = ROOT / "data" / f"{dataset_name}.zarr"
    group = zarr.open_group(str(store_path), mode="r")
    volume = group["0"][:]  # (T, Z, Y, X)
    T = volume.shape[0]

    rows = []
    for t in range(T):
        coords = detect_cells_blob(volume[t])
        for (z, y, x) in coords:
            rows.append({"frame": t, "z": int(z), "y": int(y), "x": int(x)})
        print(f"  t={t}: {len(coords)} cellules detectees")

    df = pd.DataFrame(rows, columns=["frame", "z", "y", "x"])
    out_path = ROOT / "data" / f"{dataset_name}_detections.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetections ecrites: {out_path}  ({len(df)} detections au total)")
    return df


def sanity_compare_to_ground_truth(dataset_name):
    """Compare grossierement le nombre de detections par frame au nombre de
    cellules reellement presentes dans la verite terrain (juste pour un check
    rapide, pas la vraie metrique -- ca vient a l'etape 4)."""
    gt = pd.read_csv(ROOT / "data" / f"{dataset_name}_ground_truth.csv")
    gt_nodes = gt[gt.row_type == "node"]
    counts_gt = gt_nodes.groupby("t").size()

    det = pd.read_csv(ROOT / "data" / f"{dataset_name}_detections.csv")
    counts_det = det.groupby("frame").size()

    print("\nComparaison nb de cellules par frame (GT vs detectees) :")
    print(f"{'t':>3} {'ground_truth':>13} {'detectees':>10}")
    for t in sorted(set(counts_gt.index) | set(counts_det.index)):
        print(f"{t:>3} {counts_gt.get(t, 0):>13} {counts_det.get(t, 0):>10}")


if __name__ == "__main__":
    print(f"Detection sur le dataset '{DATASET_NAME}'...")
    run_detection(DATASET_NAME)
    sanity_compare_to_ground_truth(DATASET_NAME)
