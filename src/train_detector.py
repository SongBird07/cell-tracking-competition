"""
Pipeline d'ENTRAINEMENT AUTOMATIQUE du detecteur de cellules (HeatmapUNet3D),
sur TOUTE la base d'entrainement, jusqu'a convergence.

Une seule commande suffit, sans rien regler :
    python src/train_detector.py

Ce que le script fait tout seul, sans intervention manuelle :
  1. charge l'index global des annotations (build_gt_index.py) -- si absent,
     le regenere automatiquement, sur les 199 datasets d'entrainement
  2. separe automatiquement les datasets en train/validation (80/20)
  3. echantillonne des patchs 3D centres sur des points annotes (+ tout autre
     point annote qui tombe dans le meme patch, pour un signal plus dense),
     en tirant sur TOUTE la base -- pas un sous-ensemble
  4. entraine le HeatmapUNet3D par descente de gradient (Adam), avec une perte
     MSE ponderee qui sur-penalise les erreurs pres des vrais centres
     cellulaires annotes et sous-penalise le reste (les vraies cellules NON
     annotees ne doivent pas etre punies comme du "fond")
  5. evalue automatiquement sur le split de validation a chaque epoque
  6. sauvegarde automatiquement le meilleur checkpoint (weights/heatmap_unet.pt),
     un checkpoint de secours a chaque 100 iterations (weights/heatmap_unet_latest.pt,
     resistant a un crash), et un log JSON (weights/train_log.json)
  7. s'ARRETE TOUTE SEULE (early stopping) des que la perte de validation ne
     s'ameliore plus pendant --patience epoques d'affilee -- pas besoin de
     deviner un nombre d'epoques a l'avance
  8. relancable a tout moment : reprend automatiquement depuis le dernier
     checkpoint plutot que de repartir de zero

Usage (defauts raisonnables pour un entrainement complet et automatique) :
    python src/train_detector.py
    python src/train_detector.py --patience 10 --num-workers 8   # reglages optionnels
"""

import sys, json, time, argparse, random, subprocess, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from real_io import open_zarr_volume
from dl_model import HeatmapUNet3D, DETECTOR_LEVELS
from config import ROOT, TRAIN_DIR, WEIGHTS_DIR, check_data_root

WEIGHTS_DIR.mkdir(exist_ok=True)

PATCH_SHAPE = (32, 64, 64)   # (Z, Y, X) -- divisible par 8 (3 niveaux de pooling)
TARGET_SIGMA = (1.8, 4.0, 4.0)  # sigma (z,y,x) de la gaussienne cible, en voxels
POS_WEIGHT = 25.0            # poids de la perte pres des centres annotes vs fond

# protection thermique : un GPU portable qui tourne a 100% en continu sur de
# longues sessions peut chauffer au point de crasher la machine (observe
# empiriquement : deux epoques anormalement lentes -- 2448s et 3368s au lieu
# de ~900s -- juste avant un premier crash). Plutot que d'attendre le
# prochain crash, on verifie periodiquement la temperature et on met
# l'entrainement en pause s'il fait trop chaud.
GPU_TEMP_PAUSE_C = 83     # au-dela, on arrete de solliciter le GPU
GPU_TEMP_RESUME_C = 75    # en-dessous, on reprend
GPU_TEMP_CHECK_EVERY_STEPS = 25


def get_gpu_temperature():
    """Interroge nvidia-smi (pas de dependance supplementaire). Retourne None
    si indisponible (pas de GPU NVIDIA, driver absent, etc.) -- dans ce cas
    la protection thermique est simplement desactivee, sans planter."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def move_optimizer_state(optimizer, device):
    """Deplace l'etat interne de l'optimiseur (moyennes mobiles d'Adam) vers
    `device`. Necessaire car `model.to(device)` deplace les parametres mais
    PAS l'etat de l'optimiseur, qui reste cree sur le device d'origine --
    sans ca, alterner GPU/CPU d'un pas a l'autre leve une erreur de device
    mismatch des le premier optimizer.step() suivant."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _cpu_shadow_step(model_cpu, patch_cpu, target_cpu, result_holder):
    """Calcule forward+backward sur CPU pour la portion parallele du batch,
    dans un thread separe pendant que le GPU calcule sa propre portion sur le
    thread principal EN MEME TEMPS (pas en alternance). PyTorch relache le
    GIL pendant les operations tensorielles lourdes, donc les deux threads
    progressent reellement en parallele."""
    model_cpu.zero_grad()
    pred = model_cpu(patch_cpu)
    loss = weighted_mse_loss(pred, target_cpu)
    loss.backward()
    result_holder["loss"] = loss.item()


def cooldown_if_too_hot(pause_at=GPU_TEMP_PAUSE_C, resume_below=GPU_TEMP_RESUME_C):
    temp = get_gpu_temperature()
    if temp is None or temp < pause_at:
        return
    print(f"\n  [thermique] GPU a {temp:.0f}C (seuil {pause_at}C) -- pause de refroidissement...")
    while temp is not None and temp > resume_below:
        time.sleep(20)
        temp = get_gpu_temperature()
        print(f"    ... {temp}C" if temp is not None else "    ... lecture temperature indisponible")
    print(f"  [thermique] reprise a {temp}C\n" if temp is not None else "  [thermique] reprise (temperature illisible, prudence)\n")


def ensure_gt_index():
    nodes_path = ROOT / "data" / "gt_index_nodes.csv"
    if not nodes_path.exists():
        print("Index GT absent, generation automatique (build_gt_index.py)...")
        import build_gt_index
        build_gt_index.main()
    return pd.read_csv(nodes_path)


def make_gaussian_patch(shape, center_zyx, sigma_zyx):
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    cz, cy, cx = center_zyx
    sz, sy, sx = sigma_zyx
    g = np.exp(-(((zz - cz) ** 2) / (2 * sz ** 2)
                 + ((yy - cy) ** 2) / (2 * sy ** 2)
                 + ((xx - cx) ** 2) / (2 * sx ** 2)))
    return g.astype(np.float32)


class SparsePointPatchDataset(Dataset):
    """Un exemple = un patch 3D centre (avec clipping aux bords) sur un point
    annote, avec une cible heatmap gaussienne pour CE point ET tout autre
    point annote (meme dataset, meme frame) qui tombe dans le meme patch."""

    def __init__(self, nodes_df, dataset_names, patch_shape=PATCH_SHAPE, augment=False):
        self.patch_shape = patch_shape
        self.augment = augment
        self.rows = nodes_df[nodes_df.dataset.isin(dataset_names)].reset_index(drop=True)

        # index (dataset, t) -> liste de points, pour trouver les voisins co-visibles
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

        # si le volume est plus petit que la taille de patch demandee sur un
        # axe (jamais le cas sur ce jeu de donnees, mais garde-fou pour
        # d'autres datasets/futurs entrainements), on ne demande que ce qui
        # existe puis on complete par du padding zero -- sinon la frame
        # coupee serait plus petite que patch_shape et ferait planter la
        # collation du DataLoader (tenseurs de tailles incompatibles).
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
                target = np.maximum(target, make_gaussian_patch(self.patch_shape, (lz, ly, lx), TARGET_SIGMA))

        if self.augment:
            patch, target = self._augment(patch, target)

        return torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0), \
            torch.from_numpy(np.ascontiguousarray(target)).unsqueeze(0)

    @staticmethod
    def _augment(patch, target):
        """Flips + rotation 90 (appliques identiquement au patch et a la
        cible) + leger jitter d'intensite (patch seulement). Aucune
        augmentation n'etait utilisee jusqu'ici -- un detecteur qui n'a vu
        chaque cellule que sous UNE seule orientation/exposition generalise
        moins bien, notamment sur les zones sombres ou les gradients
        d'illumination varient d'un dataset a l'autre."""
        if np.random.rand() < 0.5:
            patch, target = patch[::-1], target[::-1]
        if np.random.rand() < 0.5:
            patch, target = patch[:, ::-1], target[:, ::-1]
        if np.random.rand() < 0.5:
            patch, target = patch[:, :, ::-1], target[:, :, ::-1]
        k = np.random.randint(4)
        if k:
            patch = np.rot90(patch, k, axes=(1, 2))
            target = np.rot90(target, k, axes=(1, 2))

        gamma = np.random.uniform(0.8, 1.25)
        patch = np.clip(patch, 0, 1) ** gamma
        patch = np.clip(patch + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

        return patch, target


def weighted_mse_loss(pred, target, pos_weight=POS_WEIGHT):
    weight = 1.0 + (pos_weight - 1.0) * target
    return (weight * (pred - target) ** 2).mean()


def _count_epochs_since_best(history, best_val_loss, min_delta):
    """Reconstruit le compteur 'epoques sans amelioration' a partir du log,
    pour que l'early stopping reste coherent apres une reprise (resume)."""
    count = 0
    for h in reversed(history):
        if h["val_loss"] == h["val_loss"] and h["val_loss"] <= best_val_loss + min_delta:
            break
        count += 1
    return count


def split_datasets(nodes_df, val_fraction=0.2, seed=0):
    names = sorted(nodes_df.dataset.unique())
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_fraction))
    return names[n_val:], names[:n_val]


def run_training(max_epochs, steps_per_epoch, batch_size, lr, max_datasets=None, val_samples=400,
                  device=None, seed=0, num_workers=6, resume=True, save_every_steps=100,
                  patience=6, min_delta=1e-5, gpu_duty_cycle=1.0, cpu_parallel_fraction=0.0,
                  weight_decay=1e-4):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  num_workers: {num_workers}")

    nodes_df = ensure_gt_index()
    train_names, val_names = split_datasets(nodes_df)
    if max_datasets:
        train_names = train_names[:max_datasets]
        val_names = val_names[: max(1, max_datasets // 4)]
    print(f"{len(train_names)} datasets train / {len(val_names)} datasets validation")

    train_ds = SparsePointPatchDataset(nodes_df, train_names, augment=True)
    val_ds = SparsePointPatchDataset(nodes_df, val_names, augment=False)

    if val_samples and len(val_ds.rows) > val_samples:
        # sous-echantillonne la validation pour qu'elle reste rapide a chaque
        # epoque, meme quand le pool d'entrainement complet est utilise
        val_ds.rows = val_ds.rows.sample(n=val_samples, random_state=seed).reset_index(drop=True)

    print(f"{len(train_ds)} points d'entrainement, {len(val_ds)} points de validation (sous-echantillonnes)")

    # num_workers>0 : plusieurs process lisent les patchs zarr EN PARALLELE
    # pendant que le GPU calcule sur le batch precedent -- l'entrainement etait
    # limite par la lecture disque (chunks disperses sur des dizaines de
    # datasets), pas par le calcul, donc c'est le levier le plus efficace ici.
    loader_kwargs = dict(num_workers=num_workers)
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4, pin_memory=(device == "cuda"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = HeatmapUNet3D(levels=DETECTOR_LEVELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modele: {n_params:,} parametres (levels={DETECTOR_LEVELS})")

    # weight_decay contre le plateau (regularisation L2, decouplee -- AdamW) :
    # applique seulement aux poids des convolutions, PAS aux parametres de
    # BatchNorm (gamma/beta) ni aux biais -- les decayer degrade generalement
    # les performances (pratique standard).
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bn" in name.lower() or "batchnorm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr)
    print(f"weight_decay={weight_decay} sur {len(decay_params)} tenseurs de poids "
          f"({len(no_decay_params)} tenseurs BatchNorm/biais exemptes)")

    use_cpu_parallel = device == "cuda" and cpu_parallel_fraction and cpu_parallel_fraction > 0
    model_cpu = HeatmapUNet3D(levels=DETECTOR_LEVELS).to("cpu") if use_cpu_parallel else None
    if use_cpu_parallel:
        print(f"CPU en parallele actif: {cpu_parallel_fraction:.0%} du batch, en meme temps que le GPU (pas en alternance)")

    weights_path = WEIGHTS_DIR / "heatmap_unet.pt"
    latest_path = WEIGHTS_DIR / "heatmap_unet_latest.pt"
    log_path = WEIGHTS_DIR / "train_log.json"

    history = []
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as exc:
            # l'architecture du checkpoint ne correspond plus a DETECTOR_LEVELS
            # (ex: capacite du modele changee) -- on ne peut pas reprendre,
            # mieux vaut repartir de zero avec un message clair qu'un crash.
            print(f"Checkpoint incompatible avec l'architecture actuelle ({exc}); reprise ignoree, entrainement a partir de zero.")
            ckpt = None
        if ckpt is not None:
            start_epoch = ckpt.get("epoch", 0)
            print(f"Reprise depuis {latest_path} (epoque {start_epoch} deja effectuee)")
            if log_path.exists():
                history = json.loads(log_path.read_text())
                best_val_loss = min((h["val_loss"] for h in history if h["val_loss"] == h["val_loss"]), default=float("inf"))
                epochs_without_improvement = _count_epochs_since_best(history, best_val_loss, min_delta)
    elif resume and weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as exc:
            print(f"Checkpoint incompatible avec l'architecture actuelle ({exc}); reprise ignoree, entrainement a partir de zero.")
            ckpt = None
        if ckpt is not None:
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Reprise depuis {weights_path} (epoque {start_epoch} deja effectuee, meilleur val_loss={best_val_loss:.5f})")
            if log_path.exists():
                history = json.loads(log_path.read_text())
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

            step_t0 = time.time()
            batch_n = patch.shape[0]

            # calcul PARALLELE (pas en alternance) : une fraction du batch part
            # sur un thread CPU pendant que le reste tourne sur GPU sur le
            # thread principal, EN MEME TEMPS. Les gradients des deux portions
            # sont ensuite fusionnes (moyenne ponderee par taille de sous-batch)
            # avant un unique optimizer.step().
            n_cpu = int(round(batch_n * cpu_parallel_fraction)) if use_cpu_parallel else 0
            n_cpu = min(max(n_cpu, 0), batch_n - 1)  # au moins 1 echantillon reste sur GPU

            cpu_thread = None
            cpu_result = {}
            if n_cpu > 0:
                patch_cpu = patch[:n_cpu].to("cpu")
                target_cpu = target[:n_cpu].to("cpu")
                model_cpu.load_state_dict(model.state_dict())  # sync depuis les poids GPU courants
                cpu_thread = threading.Thread(
                    target=_cpu_shadow_step, args=(model_cpu, patch_cpu, target_cpu, cpu_result)
                )
                cpu_thread.start()

            patch_gpu = patch[n_cpu:].to(device, non_blocking=True)
            target_gpu = target[n_cpu:].to(device, non_blocking=True)
            pred = model(patch_gpu)
            loss = weighted_mse_loss(pred, target_gpu)

            optimizer.zero_grad()
            loss.backward()
            gpu_compute_time = time.time() - step_t0  # avant jointure CPU, pour doser le sleep independamment

            if cpu_thread is not None:
                cpu_thread.join()
                n_gpu = batch_n - n_cpu
                with torch.no_grad():
                    for p_gpu, p_cpu in zip(model.parameters(), model_cpu.parameters()):
                        if p_cpu.grad is None:
                            continue
                        grad_cpu_on_gpu = p_cpu.grad.to(device)
                        if p_gpu.grad is None:
                            p_gpu.grad = grad_cpu_on_gpu * (n_cpu / batch_n)
                        else:
                            p_gpu.grad.mul_(n_gpu / batch_n).add_(grad_cpu_on_gpu, alpha=n_cpu / batch_n)
                train_losses.append((loss.item() * n_gpu + cpu_result["loss"] * n_cpu) / batch_n)
            else:
                train_losses.append(loss.item())

            optimizer.step()

            if gpu_duty_cycle and gpu_duty_cycle < 1.0 and device == "cuda":
                # pause proportionnelle APRES la portion GPU (independante du
                # temps pris par le thread CPU en parallele) pour viser un
                # taux d'utilisation GPU cible en continu.
                torch.cuda.synchronize()
                time.sleep(gpu_compute_time * (1.0 / gpu_duty_cycle - 1.0))

            if save_every_steps and (step + 1) % save_every_steps == 0:
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "step_in_epoch": step + 1}, latest_path)

            if device == "cuda" and (step + 1) % GPU_TEMP_CHECK_EVERY_STEPS == 0:
                cooldown_if_too_hot()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for patch, target in val_loader:
                patch, target = patch.to(device, non_blocking=True), target.to(device, non_blocking=True)
                pred = model(patch)
                val_losses.append(weighted_mse_loss(pred, target).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        elapsed = time.time() - t0
        print(f"epoch {epoch + 1}/{start_epoch + max_epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "seconds": elapsed})

        # checkpoint "dernier etat" (toujours ecrase, pour reprise apres crash)
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
    parser = argparse.ArgumentParser(
        description="Entraine automatiquement le detecteur sur TOUTE la base "
                     "d'entrainement, jusqu'a convergence (early stopping) -- "
                     "aucun nombre d'epoques a deviner, relancable a tout moment."
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="plafond de securite (l'arret normal se fait par early stopping)")
    parser.add_argument("--patience", type=int, default=6, help="arret automatique apres N epoques sans amelioration de val_loss")
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                         help="regularisation L2 decouplee (AdamW), appliquee aux poids des convolutions "
                              "uniquement (pas BatchNorm/biais). 0 pour desactiver.")
    parser.add_argument("--max-datasets", type=int, default=None, help="limiter le nb de datasets (debug/demo -- omettre pour utiliser TOUTE la base)")
    parser.add_argument("--val-samples", type=int, default=400, help="nb de points de validation (sous-echantillonne)")
    parser.add_argument("--num-workers", type=int, default=6, help="process paralleles pour charger les patchs")
    parser.add_argument("--no-resume", action="store_true", help="ignorer un checkpoint existant et repartir de zero")
    parser.add_argument("--save-every-steps", type=int, default=100, help="checkpoint de secours toutes les N iterations")
    parser.add_argument("--gpu-duty-cycle", type=float, default=1.0,
                         help="pause proportionnelle apres la portion GPU de chaque iteration, pour viser "
                              "une fraction de temps GPU actif en continu (1.0 = desactivee, 0.5 = ~50%%).")
    parser.add_argument("--cpu-parallel-fraction", type=float, default=0.0,
                         help="fraction du batch traitee sur CPU EN PARALLELE du GPU (pas en alternance) a "
                              "chaque iteration -- un thread separe calcule pendant que le GPU calcule sa "
                              "portion, gradients fusionnes ensuite. 0 pour desactiver (tout sur GPU).")
    args = parser.parse_args()

    run_training(
        args.max_epochs, args.steps_per_epoch, args.batch_size, args.lr,
        max_datasets=args.max_datasets, val_samples=args.val_samples,
        num_workers=args.num_workers, resume=not args.no_resume,
        save_every_steps=args.save_every_steps, patience=args.patience,
        gpu_duty_cycle=args.gpu_duty_cycle, cpu_parallel_fraction=args.cpu_parallel_fraction,
        weight_decay=args.weight_decay,
    )
