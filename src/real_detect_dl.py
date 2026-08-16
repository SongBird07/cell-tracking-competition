"""
Detection par le modele entraine (HeatmapUNet3D), remplacant real_detect.py.

Meme interface (detect_dataset -> DataFrame [frame,z,y,x] + scale) pour
brancher directement sur real_track.py / real_build_submission.py /
real_evaluate.py sans rien changer d'autre dans le pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from skimage.feature import peak_local_max
from real_io import open_zarr_volume
from dl_model import HeatmapUNet3D, DETECTOR_LEVELS
from config import ROOT, TRAIN_DIR, WEIGHTS_PATH

_model_cache = {}


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device not in _model_cache:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Aucun checkpoint entraine trouve a {WEIGHTS_PATH}.\n"
                f"  -> lance d'abord: python src/train_detector.py"
            )
        model = HeatmapUNet3D(levels=DETECTOR_LEVELS).to(device)
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache[device] = model
    return _model_cache[device]


def _pad_to_multiple(volume, multiple=8):
    """Le U-Net a 3 niveaux de pooling -> chaque dimension doit etre divisible
    par 8. On complete au besoin par du zero-padding (retire apres inference)."""
    pads = [(0, (-dim) % multiple) for dim in volume.shape]
    if all(p == (0, 0) for p in pads):
        return volume, pads
    padded = np.pad(volume, pads, mode="constant")
    return padded, pads


def _run_model(model, x, device):
    """Inference avec repli automatique sur CPU si le GPU manque de VRAM
    (utile pour des volumes de test plus gros que ceux vus a l'entrainement,
    sans faire planter tout le pipeline pour une seule frame difficile)."""
    try:
        with torch.no_grad():
            return model(x)[0, 0].cpu().numpy()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("    (VRAM insuffisante pour cette frame, repli CPU)")
        cpu_model = model.to("cpu")
        with torch.no_grad():
            out = cpu_model(x.to("cpu"))[0, 0].numpy()
        model.to(device)
        return out


def refine_peak_subvoxel(heatmap, peak_zyx, window=2):
    """Centroide pondere local autour d'un pic entier, pour une precision
    sous-voxel -- peak_local_max ne peut renvoyer que des coordonnees
    entieres (le sommet exact du pic tombe souvent ENTRE deux voxels).
    Un raffinement meme petit (< 1 voxel) peut faire basculer une paire
    predite/verite-terrain du mauvais cote du seuil de matching (7um), donc
    ce gain est "gratuit" par rapport a un reentrainement.
    """
    z, y, x = peak_zyx
    Z, Y, X = heatmap.shape
    z0, z1 = max(z - window, 0), min(z + window + 1, Z)
    y0, y1 = max(y - window, 0), min(y + window + 1, Y)
    x0, x1 = max(x - window, 0), min(x + window + 1, X)

    patch = heatmap[z0:z1, y0:y1, x0:x1]
    weights = np.clip(patch, 0, None)
    total = weights.sum()
    if total <= 0:
        return float(z), float(y), float(x)

    zz, yy, xx = np.meshgrid(
        np.arange(z0, z1), np.arange(y0, y1), np.arange(x0, x1), indexing="ij"
    )
    cz = (zz * weights).sum() / total
    cy = (yy * weights).sum() / total
    cx = (xx * weights).sum() / total
    return float(cz), float(cy), float(cx)


def detect_dataset(zarr_path, min_distance=4, threshold_abs=0.3, subvoxel_refine=True, device=None, verbose=True):
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
        heatmap = _run_model(model, x, device)

        # retire le padding avant la detection de pics
        sl = tuple(slice(0, dim) for dim in norm.shape)
        heatmap = heatmap[sl]

        coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold_abs)
        for (z, y, x_) in coords:
            if subvoxel_refine:
                z, y, x_ = refine_peak_subvoxel(heatmap, (z, y, x_))
            rows.append({"frame": t, "z": int(round(z)), "y": int(round(y)), "x": int(round(x_))})

        if verbose and (t % 20 == 0 or t == T - 1):
            print(f"  t={t}/{T - 1}: {len(coords)} detections (cumulatif: {len(rows)})")

    df = pd.DataFrame(rows, columns=["frame", "z", "y", "x"])
    return df, scale_zyx


if __name__ == "__main__":
    dataset_name = "44b6_0113de3b"
    zarr_path = TRAIN_DIR / f"{dataset_name}.zarr"

    print(f"Detection (deep learning) sur '{dataset_name}'...")
    df, scale = detect_dataset(zarr_path)

    out_path = ROOT / "data" / f"{dataset_name}_dl_detections.csv"
    df.to_csv(out_path, index=False)
    print(f"\n{len(df)} detections au total ecrites dans {out_path}")
