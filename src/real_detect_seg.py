"""
Detection par SEGMENTATION D'INSTANCES (modele entraine par train_segmenter.py),
alternative a real_detect_dl.py (heatmap + pic).

Pipeline classique de segmentation d'instances :
  1. le modele predit une carte de probabilite de premier plan (masque)
  2. seuillage -> masque binaire
  3. transformee de distance + watershed pour SEPARER les cellules qui se
     touchent (une simple composante connexe fusionnerait deux noyaux
     adjacents en un seul objet)
  4. le centroide de chaque instance watershed = la detection finale

Meme interface (detect_dataset -> DataFrame [frame,z,y,x] + scale) que
real_detect.py / real_detect_dl.py, pour brancher directement sur le reste
du pipeline (real_track.py, real_build_submission.py, real_evaluate.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from real_io import open_zarr_volume
from dl_model import HeatmapUNet3D
from real_detect_dl import _pad_to_multiple, _run_model  # reutilise la logique deja auditee
from config import ROOT, TRAIN_DIR, WEIGHTS_DIR

SEG_WEIGHTS_PATH = WEIGHTS_DIR / "seg_unet.pt"
_model_cache = {}


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in _model_cache:
        if not SEG_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Aucun checkpoint de segmentation trouve a {SEG_WEIGHTS_PATH}.\n"
                f"  -> lance d'abord: python src/train_segmenter.py"
            )
        model = HeatmapUNet3D().to(device)
        ckpt = torch.load(SEG_WEIGHTS_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[device] = model
    return _model_cache[device]


def segment_frame(prob_map, prob_threshold=0.5, min_distance=10, min_size=100):
    """Seuillage + watershed sur transformee de distance -> labels d'instances.

    min_size : les instances plus petites que ce nombre de voxels sont
    rejetees. Sans ce filtre, du bruit de fond (petites zones isolees ou
    speckles ou prob_map depasse legerement le seuil) produit une foule de
    "cellules" fantomes de quelques voxels -- observe empiriquement comme la
    cause principale du sur-decoupage (beaucoup de faux positifs edges et
    divisions lors de la premiere evaluation multi-echantillons).
    """
    binary = prob_map > prob_threshold
    if not binary.any():
        return np.zeros_like(binary, dtype=np.int32), 0

    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    if len(coords) == 0:
        return np.zeros_like(binary, dtype=np.int32), 0

    markers = np.zeros(binary.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = watershed(-distance, markers, mask=binary)

    if min_size and min_size > 0:
        sizes = np.bincount(labels.ravel())
        too_small = np.where(sizes < min_size)[0]
        too_small = too_small[too_small != 0]  # 0 = fond, a ignorer
        if len(too_small):
            labels[np.isin(labels, too_small)] = 0
            # renumerote pour rester consecutif (1..n)
            remaining = np.unique(labels)
            remaining = remaining[remaining != 0]
            relabel = np.zeros(labels.max() + 1, dtype=np.int32)
            relabel[remaining] = np.arange(1, len(remaining) + 1)
            labels = relabel[labels]

    n_instances = int(labels.max())
    return labels, n_instances


def detect_dataset(zarr_path, prob_threshold=0.5, min_distance=10, min_size=100, device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    arr, scale_zyx, quantiles = open_zarr_volume(zarr_path)
    q_lo, q_hi = quantiles.get("0.1", 0.0), quantiles.get("0.999", 2000.0)
    T = arr.shape[0]

    rows = []
    for t in range(T):
        frame = arr[t][:].astype(np.float32)
        norm = np.clip((frame - q_lo) / max(q_hi - q_lo, 1e-6), 0.0, 1.0)
        padded, pads = _pad_to_multiple(norm)

        x = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        prob_map = _run_model(model, x, device)

        sl = tuple(slice(0, dim) for dim in norm.shape)
        prob_map = prob_map[sl]

        labels, n_instances = segment_frame(prob_map, prob_threshold, min_distance, min_size)
        if n_instances:
            centroids = ndi.center_of_mass(labels > 0, labels, range(1, n_instances + 1))
            for (z, y, x_) in centroids:
                rows.append({"frame": t, "z": int(round(z)), "y": int(round(y)), "x": int(round(x_))})

        if verbose and (t % 20 == 0 or t == T - 1):
            print(f"  t={t}/{T - 1}: {n_instances} instances (cumulatif: {len(rows)})")

    df = pd.DataFrame(rows, columns=["frame", "z", "y", "x"])
    return df, scale_zyx


if __name__ == "__main__":
    dataset_name = "44b6_0113de3b"
    zarr_path = TRAIN_DIR / f"{dataset_name}.zarr"

    print(f"Detection (segmentation+watershed) sur '{dataset_name}'...")
    df, scale = detect_dataset(zarr_path)

    out_path = ROOT / "data" / f"{dataset_name}_seg_detections.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} detections au total ecrites dans {out_path}")
