"""
Entrainement AUTOMATIQUE d'un SEGMENTEUR d'instances (plutot qu'un simple
detecteur par heatmap, voir train_detector.py).

Pourquoi : la heatmap (un pic gaussien par cellule) donne une localisation
imprecise en zone dense, car peak_local_max n'utilise qu'un seul voxel. Ici,
le modele apprend a predire un MASQUE BINAIRE (0/1) approchant la vraie
silhouette de chaque noyau -- au moment de l'inference (real_detect_seg.py),
le centroide d'une composante connexe est bien plus stable/precis qu'un pic
isole, et watershed permet de separer les cellules qui se touchent.

Comme la verite terrain n'est que des POINTS (pas de vrais masques), on
genere des masques ellipsoidaux synthetiques a la taille physique reelle
d'un noyau (mesuree visuellement, voir real_detect_tune.py) autour de chaque
point annote -- c'est une pseudo-supervision, pas une verite parfaite, mais
elle donne au modele une notion d'ETENDUE spatiale que la heatmap n'a pas.

Reutilise l'infrastructure deja auditee de train_detector.py (memes
garanties de securite multiprocessing, reprise sur crash, arret automatique) :
architecture identique (HeatmapUNet3D, sortie sigmoid [0,1]), mais cible et
perte differentes, et checkpoint SEPARE (weights/seg_unet.pt) pour ne pas
ecraser le detecteur existant tant que la comparaison n'est pas faite.

Usage:
    python src/train_segmenter.py
"""

import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from real_io import open_zarr_volume
from dl_model import HeatmapUNet3D
from config import TRAIN_DIR, WEIGHTS_DIR, check_data_root
from train_detector import ensure_gt_index, split_datasets, _count_epochs_since_best, PATCH_SHAPE

WEIGHTS_DIR.mkdir(exist_ok=True)

# rayon (z, y, x) en VOXELS du masque ellipsoidal synthetique. Reduit par
# rapport a la 1ere tentative (3,10,10) -> (3,8,8) : un "coeur" plus petit
# mais confiant laisse plus d'espace entre cellules voisines (watershed plus
# facile, moins de fusion en zone dense) sans deplacer le centroide, qui
# reste le meme puisque l'ellipsoide est toujours centre sur le point annote.
ELLIPSOID_RADIUS = (3, 8, 8)
# reduit de 8 -> 4 : un poids trop agressif sur le premier plan poussait le
# reseau a predire du bruit de fond comme "cellule" (beaucoup de petits
# fragments parasites observes lors de la 1ere evaluation multi-echantillons).
POS_WEIGHT = 4.0


def make_ellipsoid_patch(shape, center_zyx, radius_zyx):
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    cz, cy, cx = center_zyx
    rz, ry, rx = radius_zyx
    dist2 = (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    return (dist2 <= 1.0).astype(np.float32)


class SparseMaskPatchDataset(Dataset):
    """Identique a SparsePointPatchDataset (train_detector.py) pour le
    chargement/clipping des patchs, mais la cible est un masque ellipsoidal
    binaire plutot qu'une gaussienne."""

    def __init__(self, nodes_df, dataset_names, patch_shape=PATCH_SHAPE, radius_zyx=ELLIPSOID_RADIUS):
        self.patch_shape = patch_shape
        self.radius_zyx = radius_zyx
        self.rows = nodes_df[nodes_df.dataset.isin(dataset_names)].reset_index(drop=True)

        self.by_frame = {}
        for row in nodes_df.itertuples():
            key = (row.dataset, row.t)
            self.by_frame.setdefault(key, []).append((row.z, row.y, row.x))

        self._zarr_cache = {}
        self._quantile_cache = {}

    def __len__(self):
        return len(self.rows)

    def _get_volume(self, dataset_name):
        if dataset_name not in self._zarr_cache:
            arr, scale, quantiles = open_zarr_volume(TRAIN_DIR / f"{dataset_name}.zarr")
            self._zarr_cache[dataset_name] = arr
            self._quantile_cache[dataset_name] = quantiles
        return self._zarr_cache[dataset_name], self._quantile_cache[dataset_name]

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        dataset_name, t, z, y, x = row.dataset, int(row.t), int(row.z), int(row.y), int(row.x)

        arr, quantiles = self._get_volume(dataset_name)
        Z, Y, X = arr.shape[1:]
        pz, py, px = self.patch_shape

        ez, ey, ex = min(pz, Z), min(py, Y), min(px, X)
        z0 = min(max(z - ez // 2, 0), max(Z - ez, 0))
        y0 = min(max(y - ey // 2, 0), max(Y - ey, 0))
        x0 = min(max(x - ex // 2, 0), max(X - ex, 0))

        frame = arr[t][:].astype(np.float32)
        patch = frame[z0:z0 + ez, y0:y0 + ey, x0:x0 + ex]

        q_lo = quantiles.get("0.1", 0.0)
        q_hi = quantiles.get("0.999", patch.max() + 1e-6)
        patch = np.clip((patch - q_lo) / max(q_hi - q_lo, 1e-6), 0.0, 1.0)

        if patch.shape != self.patch_shape:
            padded = np.zeros(self.patch_shape, dtype=np.float32)
            padded[:ez, :ey, :ex] = patch
            patch = padded

        target = np.zeros(self.patch_shape, dtype=np.float32)
        for (nz, ny, nx) in self.by_frame.get((dataset_name, t), []):
            lz, ly, lx = nz - z0, ny - y0, nx - x0
            if 0 <= lz < pz and 0 <= ly < py and 0 <= lx < px:
                target = np.maximum(target, make_ellipsoid_patch(self.patch_shape, (lz, ly, lx), self.radius_zyx))

        return torch.from_numpy(patch).unsqueeze(0), torch.from_numpy(target).unsqueeze(0)


def weighted_bce_loss(pred, target, pos_weight=POS_WEIGHT, eps=1e-7):
    pred = pred.clamp(eps, 1 - eps)
    bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    weight = 1.0 + (pos_weight - 1.0) * target
    return (weight * bce).mean()


def run_training(max_epochs, steps_per_epoch, batch_size, lr, max_datasets=None, val_samples=400,
                  device=None, seed=0, num_workers=6, resume=True, save_every_steps=100,
                  patience=6, min_delta=1e-5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  num_workers: {num_workers}")

    nodes_df = ensure_gt_index()
    train_names, val_names = split_datasets(nodes_df)
    if max_datasets:
        train_names = train_names[:max_datasets]
        val_names = val_names[: max(1, max_datasets // 4)]
    print(f"{len(train_names)} datasets train / {len(val_names)} datasets validation")

    train_ds = SparseMaskPatchDataset(nodes_df, train_names)
    val_ds = SparseMaskPatchDataset(nodes_df, val_names)
    if val_samples and len(val_ds.rows) > val_samples:
        val_ds.rows = val_ds.rows.sample(n=val_samples, random_state=seed).reset_index(drop=True)
    print(f"{len(train_ds)} points d'entrainement, {len(val_ds)} points de validation (sous-echantillonnes)")

    loader_kwargs = dict(num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4, pin_memory=(device == "cuda"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = HeatmapUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights_path = WEIGHTS_DIR / "seg_unet.pt"
    latest_path = WEIGHTS_DIR / "seg_unet_latest.pt"
    log_path = WEIGHTS_DIR / "train_log_seg.json"

    history = []
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Reprise depuis {latest_path} (epoque {start_epoch} deja effectuee)")
        if log_path.exists():
            history = json.loads(log_path.read_text())
            best_val_loss = min((h["val_loss"] for h in history if h["val_loss"] == h["val_loss"]), default=float("inf"))
            epochs_without_improvement = _count_epochs_since_best(history, best_val_loss, min_delta)

    for epoch in range(start_epoch, start_epoch + max_epochs):
        model.train()
        train_losses = []
        t0 = time.time()
        train_iter = iter(train_loader)
        for step in range(steps_per_epoch):
            try:
                patch, target = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                patch, target = next(train_iter)

            patch, target = patch.to(device, non_blocking=True), target.to(device, non_blocking=True)
            pred = model(patch)
            loss = weighted_bce_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            if save_every_steps and (step + 1) % save_every_steps == 0:
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "step_in_epoch": step + 1}, latest_path)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for patch, target in val_loader:
                patch, target = patch.to(device, non_blocking=True), target.to(device, non_blocking=True)
                pred = model(patch)
                val_losses.append(weighted_bce_loss(pred, target).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        elapsed = time.time() - t0
        print(f"epoch {epoch + 1}/{start_epoch + max_epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "seconds": elapsed})

        torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss}, latest_path)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss}, weights_path)
            print(f"  -> nouveau meilleur checkpoint sauvegarde: {weights_path}")
        else:
            epochs_without_improvement += 1

        log_path.write_text(json.dumps(history, indent=2))

        if epochs_without_improvement >= patience:
            print(f"\nArret automatique (early stopping) : pas d'amelioration depuis {patience} epoques.")
            break
    else:
        print(f"\nmax_epochs ({max_epochs}) atteint sans early stopping -- relance le script pour continuer.")

    print(f"Entrainement termine. Meilleur val_loss={best_val_loss:.5f}. Checkpoint: {weights_path}")
    return weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=100)
    args = parser.parse_args()

    check_data_root(require_train=True)
    run_training(
        args.max_epochs, args.steps_per_epoch, args.batch_size, args.lr,
        max_datasets=args.max_datasets, val_samples=args.val_samples,
        num_workers=args.num_workers, resume=not args.no_resume,
        save_every_steps=args.save_every_steps, patience=args.patience,
    )
