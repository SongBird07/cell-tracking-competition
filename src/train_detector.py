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

Techniques anti-plateau actives par defaut : weight_decay decouple, gradient
clipping, augmentation avancee (flips/rotation/gamma/deformation elastique/
bruit de Poisson), Stochastic Weight Averaging en fin d'entrainement, EMA
continu des poids (weights/<nom>_ema.pt), dropout au bottleneck, supervision
profonde (pertes auxiliaires), tache auxiliaire de distance-transform, perte
focale type CenterNet (--loss-type), mining d'exemples difficiles (re-pondere
le tirage vers les points les plus mal predits), et warmup du LR en debut
d'entrainement. Desactivables individuellement (voir --help). En option : LR
schedule "cosine" (redemarrages periodiques, --lr-schedule cosine),
GroupNorm a la place de BatchNorm (--norm-type groupnorm), optimiseur SAM
(--use-sam, cherche des minima plats, cout x2), et selection periodique du
meilleur checkpoint sur la VRAIE metrique de competition plutot que sur
val_loss seul (--real-score-every).

--model-name permet d'entrainer plusieurs modeles independants (avec --seed
different) pour constituer un ENSEMBLE a l'inference -- voir
real_detect_dl.detect_dataset(model_names=(...)) qui moyenne les heatmaps
predites par plusieurs modeles (et/ou plusieurs augmentations via tta=).

Usage (defauts raisonnables pour un entrainement complet et automatique) :
    python src/train_detector.py
    python src/train_detector.py --patience 10 --num-workers 8   # reglages optionnels
"""

import sys, json, time, argparse, random, subprocess, threading, copy
from pathlib import Path
from collections import deque
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt

from real_io import open_zarr_volume
from dl_model import HeatmapUNet3D, DETECTOR_LEVELS
from config import ROOT, TRAIN_DIR, WEIGHTS_DIR, check_data_root

WEIGHTS_DIR.mkdir(exist_ok=True)

PATCH_SHAPE = (32, 64, 64)   # (Z, Y, X) -- divisible par 8 (3 niveaux de pooling)
TARGET_SIGMA = (1.8, 4.0, 4.0)  # sigma (z,y,x) de la gaussienne cible, en voxels
POS_WEIGHT = 25.0            # poids de la perte pres des centres annotes vs fond
HARD_MINING_MOMENTUM = 0.9   # inertie de la moyenne mobile de difficulte par point (mining d'exemples difficiles)

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


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021) : au lieu de suivre
    le gradient au point actuel, cherche le PIRE point dans un voisinage de
    rayon `rho` (first_step, monte dans la direction du gradient) puis
    applique le pas d'optimisation reel a partir du gradient calcule LA-BAS
    (second_step) -- pousse la recherche vers des minima "plats" (peu
    sensibles a une petite perturbation des poids), qui generalisent mieux
    et sont moins susceptibles de rester coinces dans un plateau etroit
    qu'un minimum pointu quelconque. Cout : deux passes forward+backward par
    step au lieu d'une seule.

    Usage (voir run_training) : deux forward/backward suivis de first_step()
    puis second_step() -- PAS un simple optimizer.step()."""

    def __init__(self, params, base_optimizer_cls, rho=0.05, **base_kwargs):
        defaults = dict(rho=rho, **base_kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grads = [p.grad for group in self.param_groups for p in group["params"] if p.grad is not None]
        grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads])) if grads else torch.tensor(0.0)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # revient au point de depart avant le pas reel
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()


def _update_ema(ema_model, model, decay):
    """Moyenne mobile exponentielle CONTINUE des poids, mise a jour a chaque
    step (contrairement a SWA, qui ne moyenne que les N derniers etats en fin
    d'entrainement) -- souvent plus stable, technique standard dans de
    nombreux pipelines de detection modernes (YOLO, etc.). Les buffers
    entiers (ex: num_batches_tracked) sont simplement copies, seuls les
    tenseurs flottants sont moyennes."""
    with torch.no_grad():
        ema_sd = ema_model.state_dict()
        for k, v in model.state_dict().items():
            ema_v = ema_sd[k]
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay).add_(v, alpha=1 - decay)
            else:
                ema_v.copy_(v)


def _cpu_shadow_step(model_cpu, patch_cpu, target_cpu, result_holder, aux_weight=0.3, loss_fn=None, dist_weight=0.2):
    """Calcule forward+backward sur CPU pour la portion parallele du batch,
    dans un thread separe pendant que le GPU calcule sa propre portion sur le
    thread principal EN MEME TEMPS (pas en alternance). PyTorch relache le
    GIL pendant les operations tensorielles lourdes, donc les deux threads
    progressent reellement en parallele."""
    model_cpu.zero_grad()
    pred = model_cpu(patch_cpu)
    loss = compute_loss_with_aux(model_cpu, pred, target_cpu, aux_weight, loss_fn=loss_fn, dist_weight=dist_weight)
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

    def __init__(self, nodes_df, dataset_names, patch_shape=PATCH_SHAPE, augment=False,
                 predict_distance=False, dist_norm_voxels=8.0,
                 curriculum=False, curriculum_radius=20.0):
        self.patch_shape = patch_shape
        self.augment = augment
        # predict_distance : genere une 2e cible (champ de distance normalise
        # aux centres les plus proches, voxels) empilee sur le heatmap
        # gaussien -- supervision multi-tache pour dl_model.HeatmapUNet3D(
        # predict_distance=True). dist_norm_voxels = distance (en voxels) au
        # dela de laquelle le champ normalise atteint 0.
        self.predict_distance = predict_distance
        self.dist_norm_voxels = dist_norm_voxels
        self.rows = nodes_df[nodes_df.dataset.isin(dataset_names)].reset_index(drop=True)

        # index (dataset, t) -> liste de points, pour trouver les voisins co-visibles
        self.by_frame = {}
        for row in nodes_df.itertuples():
            key = (row.dataset, row.t)
            self.by_frame.setdefault(key, []).append((row.z, row.y, row.x))

        self._zarr_cache = {}
        self._quantile_cache = {}

        # curriculum learning : poids A PRIORI (pas bases sur une perte
        # observee, contrairement au mining d'exemples difficiles) qui
        # favorisent les points ISOLES (cellules bien separees) en tout
        # debut d'entrainement -- voir _compute_curriculum_weights.
        self.curriculum_weights = self._compute_curriculum_weights(curriculum_radius) if curriculum else None

    def __len__(self):
        return len(self.rows)

    def _compute_curriculum_weights(self, radius):
        """Densite locale par frame (nombre de voisins annotes dans un rayon
        de `radius` voxels, memorisee une fois par frame et reutilisee pour
        tous ses points) -- proxy geometrique de difficulte disponible AVANT
        tout entrainement (contrairement au mining d'exemples difficiles, qui
        a besoin d'une perte observee). Poids inversement proportionnel a
        cette densite (isole = facile = poids eleve), melange 50/50 avec un
        poids uniforme -- meme logique de stabilite que le mining d'exemples
        difficiles, pour ne pas ignorer totalement les cas denses des le
        debut."""
        frame_positions = {key: np.array(pts, dtype=np.float32) for key, pts in self.by_frame.items()}
        crowding = np.zeros(len(self.rows), dtype=np.float32)
        for i, row in enumerate(self.rows.itertuples()):
            pts = frame_positions[(row.dataset, row.t)]
            center = np.array([row.z, row.y, row.x], dtype=np.float32)
            dists = np.linalg.norm(pts - center, axis=1)
            crowding[i] = np.sum((dists > 0) & (dists < radius))
        inv = 1.0 / (1.0 + crowding)
        return 0.5 * (inv / inv.mean()) + 0.5

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
        centers_mask = np.zeros(self.patch_shape, dtype=bool) if self.predict_distance else None
        for (nz, ny, nx) in self.by_frame.get((dataset_name, t), []):
            lz, ly, lx = nz - z0, ny - y0, nx - x0
            if 0 <= lz < pz and 0 <= ly < py and 0 <= lx < px:
                target = np.maximum(target, make_gaussian_patch(self.patch_shape, (lz, ly, lx), TARGET_SIGMA))
                if centers_mask is not None:
                    centers_mask[lz, ly, lx] = True

        dist_target = None
        if self.predict_distance:
            if centers_mask.any():
                dist = distance_transform_edt(~centers_mask)
                dist_target = np.clip(1.0 - dist / self.dist_norm_voxels, 0.0, 1.0).astype(np.float32)
            else:
                dist_target = np.zeros(self.patch_shape, dtype=np.float32)

        if self.augment:
            patch, target, dist_target = self._augment(patch, target, dist_target)

        if self.predict_distance:
            target_arr = np.ascontiguousarray(np.stack([target, dist_target], axis=0))  # (2, Z, Y, X)
        else:
            target_arr = np.ascontiguousarray(target)[None]  # (1, Z, Y, X)

        # idx renvoye en plus de (patch, target) : necessaire pour le mining
        # d'exemples difficiles (train_detector.py associe une perte
        # observee a CET index precis pour ajuster sa probabilite de
        # re-tirage) -- ignore partout ailleurs (val_loader, BN update SWA).
        return torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0), \
            torch.from_numpy(target_arr), \
            idx

    @staticmethod
    def _augment(patch, target, extra=None):
        """Flips + rotation 90 (appliques identiquement au patch, a la cible
        et a `extra` si fourni -- la cible auxiliaire de distance-transform,
        qui doit rester spatialement alignee) + leger jitter d'intensite
        (patch seulement). Aucune augmentation n'etait utilisee jusqu'ici --
        un detecteur qui n'a vu chaque cellule que sous UNE seule
        orientation/exposition generalise moins bien, notamment sur les
        zones sombres ou les gradients d'illumination varient d'un dataset a
        l'autre."""
        if np.random.rand() < 0.5:
            patch, target = patch[::-1], target[::-1]
            if extra is not None:
                extra = extra[::-1]
        if np.random.rand() < 0.5:
            patch, target = patch[:, ::-1], target[:, ::-1]
            if extra is not None:
                extra = extra[:, ::-1]
        if np.random.rand() < 0.5:
            patch, target = patch[:, :, ::-1], target[:, :, ::-1]
            if extra is not None:
                extra = extra[:, :, ::-1]
        k = np.random.randint(4)
        if k:
            patch = np.rot90(patch, k, axes=(1, 2))
            target = np.rot90(target, k, axes=(1, 2))
            if extra is not None:
                extra = np.rot90(extra, k, axes=(1, 2))

        gamma = np.random.uniform(0.8, 1.25)
        patch = np.clip(patch, 0, 1) ** gamma
        patch = np.clip(patch + np.random.uniform(-0.05, 0.05), 0.0, 1.0)

        if np.random.rand() < 0.4:
            patch, target, extra = SparsePointPatchDataset._elastic_deform(patch, target, extra)
        if np.random.rand() < 0.3:
            patch = SparsePointPatchDataset._poisson_noise(patch)

        return patch, target, extra

    @staticmethod
    def _elastic_deform(patch, target, extra=None, alpha=6.0, sigma=4.0):
        """Deformation elastique : un champ de deplacement aleatoire lisse
        (bruit gaussien flou = correlation spatiale realiste) est applique
        IDENTIQUEMENT au patch, a la cible et a `extra` si fourni, donc tout
        reste aligne sans recalcul de coordonnees. Particulierement utile en
        imagerie biologique ou les cellules se deforment naturellement
        (contrairement a une simple rotation/flip)."""
        shape = patch.shape
        # champ de deplacement brut, lisse par un flou gaussien (sigma=echelle
        # spatiale de la deformation), puis mis a l'echelle par alpha (amplitude)
        dz = gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma) * alpha
        dy = gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma) * alpha
        dx = gaussian_filter(np.random.randn(*shape).astype(np.float32), sigma) * alpha

        zz, yy, xx = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        coords = [
            np.clip(zz + dz, 0, shape[0] - 1),
            np.clip(yy + dy, 0, shape[1] - 1),
            np.clip(xx + dx, 0, shape[2] - 1),
        ]
        patch_deformed = map_coordinates(patch, coords, order=1, mode="reflect").astype(np.float32)
        target_deformed = map_coordinates(target, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
        extra_deformed = None
        if extra is not None:
            extra_deformed = map_coordinates(extra, coords, order=1, mode="constant", cval=0.0).astype(np.float32)
        return patch_deformed, target_deformed, extra_deformed

    @staticmethod
    def _poisson_noise(patch, scale=60.0):
        """Bruit de photon-comptage (Poisson), le modele de bruit realiste en
        microscopie a fluorescence -- plus fort dans les zones sombres,
        contrairement a un bruit gaussien additif uniforme."""
        scaled = np.clip(patch, 0, 1) * scale
        noisy = np.random.poisson(scaled).astype(np.float32) / scale
        return np.clip(noisy, 0.0, 1.0)


def weighted_mse_loss(pred, target, pos_weight=POS_WEIGHT, reduction="mean"):
    """reduction="none" renvoie une perte par echantillon (moyennee sur les
    dimensions spatiales, dim batch conservee) -- utilise pour le mining
    d'exemples difficiles, qui a besoin de savoir QUEL echantillon du batch
    est mal predit, pas seulement la moyenne du batch."""
    weight = 1.0 + (pos_weight - 1.0) * target
    se = weight * (pred - target) ** 2
    if reduction == "none":
        return se.mean(dim=tuple(range(1, se.ndim)))
    return se.mean()


def focal_heatmap_loss(pred, target, alpha=2.0, beta=4.0, eps=1e-6, reduction="mean"):
    """Perte focale a penalite reduite pres du pic (CornerNet/CenterNet),
    alternative a weighted_mse_loss : le gradient est concentre sur les VRAIS
    desaccords (pixels ou le modele se trompe vraiment) plutot que reparti
    uniformement sur tout le voisinage pondere de chaque centre annote --
    standard pour la regression de heatmaps de detection, cense accelerer la
    convergence tardive (le principal symptome d'un plateau).

    pos_mask = pixels au pic exact du gaussien-cible (target~1). Le seuil
    0.99 (plutot que ==1.0 strict) tolere la legere perte de precision
    introduite par la deformation elastique (interpolation lineaire du pic).
    """
    pred = pred.clamp(min=eps, max=1 - eps)
    pos_mask = (target > 0.99).float()
    neg_weight = torch.pow(1.0 - target, beta)

    pos_loss = torch.pow(1.0 - pred, alpha) * torch.log(pred) * pos_mask
    neg_loss = torch.pow(pred, alpha) * torch.log(1.0 - pred) * neg_weight * (1.0 - pos_mask)

    spatial_dims = tuple(range(1, pos_mask.ndim))
    n_pos = pos_mask.sum(dim=spatial_dims).clamp(min=1.0)
    loss_per_sample = -(pos_loss + neg_loss).sum(dim=spatial_dims) / n_pos
    if reduction == "none":
        return loss_per_sample
    return loss_per_sample.mean()


def compute_loss_with_aux(model, pred, target, aux_weight=0.3, loss_fn=None, dist_weight=0.2):
    """Perte principale (loss_fn, weighted_mse_loss par defaut) + perte
    auxiliaire moyenne de supervision profonde (si activee) + perte de la
    tache auxiliaire de distance-transform (si activee, toujours en MSE
    simple -- ce n'est pas un heatmap de detection, pas de raison de
    ponderer/focaliser). `target` peut avoir 2 canaux (heatmap + distance,
    voir SparsePointPatchDataset(predict_distance=True)) -- seul le canal 0
    sert de cible aux pertes heatmap/deep-supervision.

    loss_fn n'affecte QUE la perte d'ENTRAINEMENT -- val_loss (suivi de
    l'historique, early stopping, selection du meilleur checkpoint) continue
    TOUJOURS d'utiliser weighted_mse_loss ailleurs dans run_training, pour
    rester comparable a travers tout l'historique d'entrainement meme si le
    choix de loss_fn change d'une run a l'autre."""
    loss_fn = loss_fn or weighted_mse_loss
    predict_distance = getattr(model, "predict_distance", False) and target.shape[1] > 1
    heatmap_target = target[:, 0:1] if predict_distance else target

    loss = loss_fn(pred, heatmap_target)
    aux_outputs = getattr(model, "aux_outputs", None)
    if aux_outputs:
        aux_loss = sum(loss_fn(a, heatmap_target) for a in aux_outputs) / len(aux_outputs)
        loss = loss + aux_weight * aux_loss

    dist_output = getattr(model, "distance_output", None)
    if predict_distance and dist_output is not None:
        dist_target = target[:, 1:2]
        dist_loss = nn.functional.mse_loss(dist_output, dist_target)
        loss = loss + dist_weight * dist_loss

    return loss


def _pseudo_label_pass(model, train_ds, dataset_names, device, threshold=0.9, min_distance=6, max_frames=5, seed=0):
    """Pseudo-labellisation : ajoute au pool d'annotations d'ENTRAINEMENT des
    points a HAUTE confiance detectes par le modele ACTUEL mais absents des
    annotations sparses d'origine. Utile car les annotations de ce jeu de
    donnees sont clairsemees -- une vraie cellule visible mais non-annotee
    est actuellement punie a tort comme du "fond" dans la cible heatmap.

    RISQUE INHERENT : si le modele est deja confiant a tort (faux positif
    confiant), cette passe RENFORCE l'erreur au lieu de la corriger --
    d'ou un seuil de confiance eleve par defaut (0.9) et un nombre de
    frames limite par appel (max_frames), a n'activer qu'avec prudence
    (pseudo_label_every dans run_training, off par defaut).

    Modifie train_ds.by_frame IN-PLACE (jamais val_ds -- val_loss doit
    rester une mesure honnete, non influencee par les propres predictions du
    modele). Les patchs deja construits ne sont pas affectes retroactivement,
    seules les epoques SUIVANTES en beneficient (nouvelle cible heatmap au
    prochain __getitem__ de ce (dataset, frame))."""
    from skimage.feature import peak_local_max

    was_training = model.training
    model.eval()
    rng = random.Random(seed)
    keys = [k for k in train_ds.by_frame.keys() if k[0] in dataset_names]
    rng.shuffle(keys)

    n_added, frames_checked = 0, 0
    with torch.no_grad():
        for (dataset_name, t) in keys:
            if frames_checked >= max_frames:
                break
            arr, quantiles = train_ds._get_volume(dataset_name)
            frame = arr[t][:].astype(np.float32)
            q_lo = quantiles.get("0.1", 0.0)
            q_hi = quantiles.get("0.999", frame.max() + 1e-6)
            norm = np.clip((frame - q_lo) / max(q_hi - q_lo, 1e-6), 0.0, 1.0)

            pads = [(0, (-d) % 8) for d in norm.shape]  # 3 niveaux de pooling -> divisible par 8
            padded = np.pad(norm, pads, mode="constant") if any(p != (0, 0) for p in pads) else norm
            x = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
            heatmap = model(x)[0, 0].cpu().numpy()
            heatmap = heatmap[tuple(slice(0, d) for d in norm.shape)]

            coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=threshold)
            key = (dataset_name, t)
            existing = np.array(train_ds.by_frame[key], dtype=np.float32) if train_ds.by_frame[key] else np.empty((0, 3))
            for (z, y, x_) in coords:
                if existing.size:
                    if np.linalg.norm(existing - np.array([z, y, x_]), axis=1).min() < min_distance:
                        continue
                train_ds.by_frame[key].append((int(z), int(y), int(x_)))
                n_added += 1
            frames_checked += 1

    if was_training:
        model.train()
    return n_added, frames_checked


def _evaluate_real_score(model, model_name, val_names, device, n_datasets=2):
    """Evalue la vraie metrique de competition (edge Jaccard + 0.1*division
    Jaccard, voir multi_sample_evaluate.py) sur quelques datasets de
    VALIDATION, avec l'etat EN MEMOIRE actuel du modele -- pas besoin de
    sauvegarder sur disque puis recharger : on insere directement `model`
    dans le cache de real_detect_dl.load_model() le temps de l'evaluation,
    puis on restaure ce qui s'y trouvait avant (au cas ou un autre code
    utiliserait aussi ce cache dans le meme process).

    val_loss (MSE ponderee sur des points isoles) n'est qu'un proxy de ce qui
    est note -- il ne capture ni le tracking ni les divisions. Ce controle
    periodique, plus couteux (detection + tracking + calcul de la vraie
    metrique sur des volumes entiers), verifie que les deux ne divergent pas
    et permet de garder a part le checkpoint reellement le meilleur au sens
    de la competition, qui peut differer du meilleur par val_loss.
    """
    import real_detect_dl
    from multi_sample_evaluate import evaluate_dataset
    from functools import partial
    from config import DATA_DIR

    cache_key = (model_name, device)
    prev_cached = real_detect_dl._model_cache.get(cache_key)
    real_detect_dl._model_cache[cache_key] = model
    chosen = list(val_names[:n_datasets])
    try:
        detect_fn = partial(real_detect_dl.detect_dataset, model_names=(model_name,), device=device)
        total_tp = total_fp = total_fn = 0
        total_div_tp = total_div_fp = total_div_fn = 0
        for name in chosen:
            edge_m, div_m = evaluate_dataset(name, detect_fn)
            total_tp += edge_m["edge_tp"]; total_fp += edge_m["edge_fp"]; total_fn += edge_m["edge_fn"]
            total_div_tp += div_m["division_tp"]; total_div_fp += div_m["division_fp"]; total_div_fn += div_m["division_fn"]
        edge_jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0
        div_jaccard = (total_div_tp / (total_div_tp + total_div_fp + total_div_fn)
                       if (total_div_tp + total_div_fp + total_div_fn) else 0.0)
        return edge_jaccard + 0.1 * div_jaccard
    finally:
        if prev_cached is not None:
            real_detect_dl._model_cache[cache_key] = prev_cached
        else:
            real_detect_dl._model_cache.pop(cache_key, None)
        for name in chosen:
            (DATA_DIR / f"_tmp_{name}_submission.csv").unlink(missing_ok=True)


_ADDITIVE_KEY_PREFIXES = ("aux_heads.", "distance_head.")


def _load_state_dict_resumable(model, state_dict):
    """Charge un state_dict pour REPRENDRE l'entrainement, tolerant a un
    ajout purement additif d'architecture (aux_heads.* de la supervision
    profonde, distance_head.* de la tache auxiliaire de distance-transform --
    absents d'un checkpoint entraine avant leur introduction) : ces poids
    manquants restent a leur initialisation aleatoire, le reste
    (encodeur/decodeur/tete principale -- l'essentiel de l'entrainement deja
    effectue) reprend normalement plutot que de tout perdre pour un
    changement qui ne concerne qu'une sortie auxiliaire d'entrainement.

    ATTENTION : norm_type n'est PAS additif (BatchNorm <-> GroupNorm change
    le TYPE de couche, pas juste des cles en plus) -- voir l'avertissement en
    tete de dl_model.py. L'appelant doit verifier ckpt.get("norm_type") avant
    de resumer si le norm_type courant a change.

    Un vrai changement d'architecture incompatible (ex: DETECTOR_LEVELS
    modifie -> tailles de tenseurs differentes) leve quand meme une
    RuntimeError (non absorbee par strict=False), geree par l'appelant."""
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith(_ADDITIVE_KEY_PREFIXES)]
    unexpected = [k for k in unexpected if not k.startswith(_ADDITIVE_KEY_PREFIXES)]
    if missing or unexpected:
        print(f"  (reprise partielle -- cles manquantes={missing}, inattendues={unexpected})")


def _count_epochs_since_best(history, best_val_loss, min_delta, best_ema_val_loss=None):
    """Reconstruit le compteur 'epoques sans amelioration' a partir du log,
    pour que l'early stopping reste coherent apres une reprise (resume).
    S'arrete a la derniere epoque ou soit val_loss, soit ema_val_loss (si
    fourni) etait a son meilleur -- coherent avec le reset combine du
    compteur pendant l'entrainement (voir run_training)."""
    count = 0
    for h in reversed(history):
        val_ok = h["val_loss"] == h["val_loss"] and h["val_loss"] <= best_val_loss + min_delta
        ema_val = h.get("ema_val_loss")
        ema_ok = (best_ema_val_loss is not None and ema_val is not None
                  and ema_val == ema_val and ema_val <= best_ema_val_loss + min_delta)
        if val_ok or ema_ok:
            break
        count += 1
    return count


def split_datasets(nodes_df, val_fraction=0.2, seed=0):
    names = sorted(nodes_df.dataset.unique())
    rng = random.Random(seed)
    rng.shuffle(names)
    n_val = max(1, int(len(names) * val_fraction))
    return names[n_val:], names[:n_val]


def run_training(max_epochs, steps_per_epoch, batch_size, lr, max_datasets=None, val_samples=1000,
                  device=None, seed=0, num_workers=6, resume=True, save_every_steps=100,
                  patience=6, min_delta=1e-5, gpu_duty_cycle=1.0, cpu_parallel_fraction=0.0,
                  weight_decay=1e-4, lr_patience=3, lr_factor=0.5, grad_clip=1.0, swa_last_n=5,
                  dropout=0.1, deep_supervision=True, aux_loss_weight=0.3,
                  lr_schedule="plateau", model_name="heatmap_unet",
                  hard_mining=True, hard_mining_start_epoch=3, hard_mining_power=1.0,
                  real_score_every=0, real_score_datasets=2,
                  loss_type="focal", predict_distance=True, dist_loss_weight=0.2,
                  norm_type="batch", ema_decay=0.999, warmup_steps=300,
                  use_sam=False, sam_rho=0.05,
                  curriculum=True, curriculum_radius=20.0,
                  pseudo_label_every=0, pseudo_label_threshold=0.9,
                  pseudo_label_min_distance=6, pseudo_label_max_frames=5):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  num_workers: {num_workers}")

    # seed distincte par modele -> initialisation des poids ET ordre de
    # tirage des donnees differents, condition necessaire pour qu'un
    # ensemble de plusieurs modeles apporte reellement de la diversite
    # (sinon plusieurs runs convergeraient vers un modele quasi-identique)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    nodes_df = ensure_gt_index()
    train_names, val_names = split_datasets(nodes_df)
    if max_datasets:
        train_names = train_names[:max_datasets]
        val_names = val_names[: max(1, max_datasets // 4)]
    print(f"{len(train_names)} datasets train / {len(val_names)} datasets validation")

    train_ds = SparsePointPatchDataset(nodes_df, train_names, augment=True, predict_distance=predict_distance,
                                        curriculum=curriculum, curriculum_radius=curriculum_radius)
    val_ds = SparsePointPatchDataset(nodes_df, val_names, augment=False, predict_distance=predict_distance)

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

    # mining d'exemples difficiles : au lieu de tirer les patchs uniformement,
    # on pondere le tirage par la perte recente observee sur chaque point
    # (moyenne mobile) -- un WeightedRandomSampler dont les poids sont mis a
    # jour a la fin de chaque epoque. Les points systematiquement mal predits
    # (cellules qui se touchent, zones sombres...) sont alors revus plus
    # souvent, sans qu'il soit besoin de les identifier a la main. Melange a
    # 50% avec un poids uniforme pour ne pas negliger les points faciles
    # (utiles a la regularisation) ni sur-ajuster sur quelques points
    # bruites. sample_loss demarre a 1.0 (neutre) pour tous -- les vrais
    # ecarts de difficulte n'emergent qu'apres quelques epoques
    # (hard_mining_start_epoch retarde l'activation, le signal initial est
    # trop bruite pour etre utile).
    # curriculum learning : au tout debut (avant que hard_mining_start_epoch
    # ne soit atteint), le MEME sampler part des poids de curriculum
    # (statiques, favorisent les points isoles -- voir
    # SparsePointPatchDataset._compute_curriculum_weights) plutot que d'un
    # tirage uniforme. Des que le mining d'exemples difficiles s'active, ses
    # poids (bases sur la perte REELLEMENT observee) ecrasent naturellement
    # ceux du curriculum -- transition sans code special.
    sample_loss = np.ones(len(train_ds), dtype=np.float64)
    need_sampler = hard_mining or curriculum
    if need_sampler:
        initial_weights = (torch.as_tensor(train_ds.curriculum_weights, dtype=torch.double)
                            if curriculum and train_ds.curriculum_weights is not None
                            else torch.ones(len(train_ds), dtype=torch.double))
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=initial_weights, num_samples=len(train_ds), replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **loader_kwargs)
        if curriculum:
            print("Curriculum actif : privilegie les points isoles (cellules bien separees) au debut de l'entrainement.")
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # architecture partagee par TOUTES les copies du modele (model, model_cpu,
    # swa_model, ema_model) -- doivent rester identiques pour que la copie de
    # state_dict entre elles (sync CPU-parallele, moyennage SWA/EMA) reste valide.
    arch_kwargs = dict(levels=DETECTOR_LEVELS, dropout=dropout, deep_supervision=deep_supervision,
                        predict_distance=predict_distance, norm_type=norm_type)

    model = HeatmapUNet3D(**arch_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modele: {n_params:,} parametres (levels={DETECTOR_LEVELS}, dropout={dropout}, "
          f"deep_supervision={deep_supervision}, predict_distance={predict_distance}, norm_type={norm_type})")

    main_loss_fn = focal_heatmap_loss if loss_type == "focal" else weighted_mse_loss
    print(f"loss_type={loss_type}" + ("  (val_loss/early-stopping restent en weighted_mse pour comparabilite)" if loss_type != "weighted_mse" else ""))

    # SAM double le cout par step (2 forward/backward) et gere ses propres
    # perturbations de poids -- combiner avec le calcul CPU-parallele
    # (qui suppose un unique optimizer.step()) ajouterait une complexite
    # disproportionnee pour un gain incertain, donc mutuellement exclusifs.
    if use_sam and cpu_parallel_fraction and cpu_parallel_fraction > 0:
        print("ATTENTION: use_sam=True desactive cpu_parallel_fraction (les deux ne sont pas combinables).")
        cpu_parallel_fraction = 0.0

    # weight_decay contre le plateau (regularisation L2, decouplee -- AdamW) :
    # applique seulement aux poids des convolutions, PAS aux parametres de
    # BatchNorm/GroupNorm (gamma/beta) ni aux biais -- les decayer degrade
    # generalement les performances (pratique standard).
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bn" in name.lower() or "batchnorm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if use_sam:
        optimizer = SAM(param_groups, torch.optim.AdamW, rho=sam_rho, lr=lr)
        print(f"SAM actif (rho={sam_rho}) -- deux forward/backward par step, cout ~x2.")
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    print(f"weight_decay={weight_decay} sur {len(decay_params)} tenseurs de poids "
          f"({len(no_decay_params)} tenseurs BatchNorm/biais exemptes)")

    # reduit lr automatiquement quand val_loss stagne -- complement du
    # weight_decay : celui-ci regularise, le scheduler affine la convergence
    # une fois que les pas actuels sont trop grands pour progresser encore.
    # "cosine" est une alternative qui remonte lr periodiquement (warm
    # restarts) au lieu de seulement le reduire -- peut aider a s'echapper
    # d'un minimum local pointu plutot que de s'y installer.
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        scheduler_needs_val_loss = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=lr_factor, patience=lr_patience
        )
        scheduler_needs_val_loss = True

    # SWA (Stochastic Weight Averaging) : conserve les N derniers etats du
    # modele (copies CPU, ~22 Mo chacune) pour les moyenner en fin
    # d'entrainement -- souvent un gain "gratuit" de generalisation une fois
    # proche du plateau, sans entrainement supplementaire.
    swa_snapshots = deque(maxlen=swa_last_n) if swa_last_n and swa_last_n > 1 else None

    use_cpu_parallel = device == "cuda" and cpu_parallel_fraction and cpu_parallel_fraction > 0
    model_cpu = HeatmapUNet3D(**arch_kwargs).to("cpu") if use_cpu_parallel else None
    if use_cpu_parallel:
        print(f"CPU en parallele actif: {cpu_parallel_fraction:.0%} du batch, en meme temps que le GPU (pas en alternance)")

    # model_name permet d'entrainer plusieurs modeles independants (seeds
    # differentes) sans qu'ils s'ecrasent -- utile pour constituer un
    # ensemble a l'inference (moyenne des predictions de plusieurs modeles).
    # Avec le nom par defaut, les chemins sont IDENTIQUES a avant (compatible
    # avec les checkpoints/logs deja existants).
    weights_path = WEIGHTS_DIR / f"{model_name}.pt"
    latest_path = WEIGHTS_DIR / f"{model_name}_latest.pt"
    log_path = WEIGHTS_DIR / ("train_log.json" if model_name == "heatmap_unet" else f"train_log_{model_name}.json")

    history = []
    start_epoch = 0
    best_val_loss = float("inf")
    # reconstruit depuis le log a la reprise (comme best_val_loss) -- sert a
    # calculer correctement epochs_without_improvement (voir plus bas, le
    # compteur de patience reset desormais sur une amelioration de val_loss
    # OU d'EMA). Le MODELE ema_model, lui, n'est pas restaure depuis
    # l'ancien fichier _ema.pt (voir plus bas) -- seule cette valeur l'est.
    best_ema_val_loss = float("inf")
    epochs_without_improvement = 0
    # non restaure a la reprise (recalcule vite: quelques evaluations reelles
    # suffisent a retrouver un ordre de grandeur correct) -- weights/<nom>_bestreal.pt
    # sur disque garde neanmoins le meilleur checkpoint jamais vu sur le vrai score.
    best_real_score = float("-inf")

    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        ckpt_norm_type = ckpt.get("norm_type", "batch")
        if ckpt_norm_type != norm_type:
            print(f"  ATTENTION: ce checkpoint a ete entraine avec norm_type='{ckpt_norm_type}', different "
                  f"du norm_type actuel ('{norm_type}') -- les statistiques de normalisation ne sont PAS "
                  f"compatibles (voir dl_model.py). Relance avec --norm-type {ckpt_norm_type} pour une reprise propre.")
        try:
            _load_state_dict_resumable(model, ckpt["model_state"])
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
                best_ema_val_loss = min((h["ema_val_loss"] for h in history
                                          if h.get("ema_val_loss") is not None and h["ema_val_loss"] == h["ema_val_loss"]),
                                         default=float("inf"))
                epochs_without_improvement = _count_epochs_since_best(history, best_val_loss, min_delta, best_ema_val_loss)
    elif resume and weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        ckpt_norm_type = ckpt.get("norm_type", "batch")
        if ckpt_norm_type != norm_type:
            print(f"  ATTENTION: ce checkpoint a ete entraine avec norm_type='{ckpt_norm_type}', different "
                  f"du norm_type actuel ('{norm_type}') -- les statistiques de normalisation ne sont PAS "
                  f"compatibles (voir dl_model.py). Relance avec --norm-type {ckpt_norm_type} pour une reprise propre.")
        try:
            _load_state_dict_resumable(model, ckpt["model_state"])
        except RuntimeError as exc:
            print(f"Checkpoint incompatible avec l'architecture actuelle ({exc}); reprise ignoree, entrainement a partir de zero.")
            ckpt = None
        if ckpt is not None:
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Reprise depuis {weights_path} (epoque {start_epoch} deja effectuee, meilleur val_loss={best_val_loss:.5f})")
            if log_path.exists():
                history = json.loads(log_path.read_text())
                best_ema_val_loss = min((h["ema_val_loss"] for h in history
                                          if h.get("ema_val_loss") is not None and h["ema_val_loss"] == h["ema_val_loss"]),
                                         default=float("inf"))
                epochs_without_improvement = _count_epochs_since_best(history, best_val_loss, min_delta, best_ema_val_loss)

    # EMA (moyenne mobile continue, voir _update_ema) : le MODELE est
    # initialise a partir de l'etat COURANT du modele (donc APRES la reprise
    # ci-dessus, resumed ou non) -- non restaure depuis un ancien fichier
    # _ema.pt (se reaccumule vite depuis ce point de depart, meme
    # simplification que best_real_score plus haut). best_ema_val_loss (la
    # VALEUR, pour les comparaisons "meilleur jusqu'ici"), en revanche, EST
    # reconstruite depuis le log ci-dessus.
    use_ema = ema_decay and ema_decay > 0
    ema_model = None
    best_ema_val_loss = float("inf")
    if use_ema:
        ema_model = HeatmapUNet3D(**arch_kwargs).to(device)
        ema_model.load_state_dict(model.state_dict())
        print(f"EMA active (decay={ema_decay}) -- moyenne mobile continue des poids, evaluee chaque epoque.")

    global_step = 0

    for epoch in range(start_epoch, start_epoch + max_epochs):
        model.train()
        train_losses = []
        t0 = time.time()
        train_iter = iter(train_loader)
        for step in range(steps_per_epoch):
            try:
                patch, target, idx = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                patch, target, idx = next(train_iter)

            step_t0 = time.time()
            batch_n = patch.shape[0]

            # warmup lineaire du LR sur les warmup_steps premiers pas d'un
            # entrainement FRESH (pas au resume -- start_epoch>0 -- ou ca n'a
            # pas de sens) : stabilise le tout debut, avant que le scheduler
            # (plateau/cosine) ne prenne le relais normalement.
            if warmup_steps and start_epoch == 0 and global_step < warmup_steps:
                warmup_factor = (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = lr * warmup_factor

            if use_sam:
                # SAM : deux forward/backward -- le premier localise la
                # "pire" direction proche (first_step la applique aux poids),
                # le second calcule le VRAI gradient a ce point perturbe puis
                # revient au point de depart pour y appliquer le pas reel
                # (second_step). Pas de split CPU-parallele ici (voir
                # avertissement a la construction de l'optimiseur).
                patch_gpu = patch.to(device, non_blocking=True)
                target_gpu = target.to(device, non_blocking=True)

                pred = model(patch_gpu)
                loss = compute_loss_with_aux(model, pred, target_gpu, aux_loss_weight,
                                              loss_fn=main_loss_fn, dist_weight=dist_loss_weight)

                if train_sampler is not None:
                    with torch.no_grad():
                        heatmap_target = target_gpu[:, 0:1] if predict_distance else target_gpu
                        per_sample_loss = main_loss_fn(pred, heatmap_target, reduction="none").cpu().numpy()
                    idx_np = idx.numpy()
                    sample_loss[idx_np] = (
                        HARD_MINING_MOMENTUM * sample_loss[idx_np] + (1 - HARD_MINING_MOMENTUM) * per_sample_loss
                    )

                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.first_step(zero_grad=True)

                pred2 = model(patch_gpu)
                loss2 = compute_loss_with_aux(model, pred2, target_gpu, aux_loss_weight,
                                               loss_fn=main_loss_fn, dist_weight=dist_loss_weight)
                loss2.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.second_step(zero_grad=True)

                train_losses.append(loss.item())  # perte au point de depart, pas au point perturbe
                gpu_compute_time = time.time() - step_t0
            else:
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
                        target=_cpu_shadow_step,
                        args=(model_cpu, patch_cpu, target_cpu, cpu_result, aux_loss_weight, main_loss_fn, dist_loss_weight),
                    )
                    cpu_thread.start()

                patch_gpu = patch[n_cpu:].to(device, non_blocking=True)
                target_gpu = target[n_cpu:].to(device, non_blocking=True)
                pred = model(patch_gpu)
                loss = compute_loss_with_aux(model, pred, target_gpu, aux_loss_weight,
                                              loss_fn=main_loss_fn, dist_weight=dist_loss_weight)

                if train_sampler is not None:
                    # met a jour la difficulte observee (moyenne mobile) pour
                    # CHAQUE point du sous-batch GPU -- la portion CPU (si
                    # cpu_parallel_fraction>0) n'est pas trackee ici par
                    # simplicite, elle reste a poids neutre. Sans effet sur le
                    # gradient (juste de la comptabilite pour le prochain tirage).
                    with torch.no_grad():
                        heatmap_target = target_gpu[:, 0:1] if predict_distance else target_gpu
                        per_sample_loss = main_loss_fn(pred, heatmap_target, reduction="none").cpu().numpy()
                    idx_gpu = idx[n_cpu:].numpy()
                    sample_loss[idx_gpu] = (
                        HARD_MINING_MOMENTUM * sample_loss[idx_gpu] + (1 - HARD_MINING_MOMENTUM) * per_sample_loss
                    )

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

                if grad_clip and grad_clip > 0:
                    # empeche un gradient occasionnellement enorme (patch bruite,
                    # cellule saturee) de faire un pas destabilisant -- reduit le
                    # bruit observe dans les oscillations de val_loss d'une
                    # epoque a l'autre.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            if use_ema:
                _update_ema(ema_model, model, ema_decay)
            global_step += 1

            if gpu_duty_cycle and gpu_duty_cycle < 1.0 and device == "cuda":
                # pause proportionnelle APRES la portion GPU (independante du
                # temps pris par le thread CPU en parallele) pour viser un
                # taux d'utilisation GPU cible en continu.
                torch.cuda.synchronize()
                time.sleep(gpu_compute_time * (1.0 / gpu_duty_cycle - 1.0))

            if save_every_steps and (step + 1) % save_every_steps == 0:
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "step_in_epoch": step + 1, "norm_type": norm_type}, latest_path)

            if device == "cuda" and (step + 1) % GPU_TEMP_CHECK_EVERY_STEPS == 0:
                cooldown_if_too_hot()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for patch, target, _idx in val_loader:
                patch, target = patch.to(device, non_blocking=True), target.to(device, non_blocking=True)
                pred = model(patch)
                heatmap_target = target[:, 0:1] if predict_distance else target
                # val_loss reste TOUJOURS en weighted_mse_loss (pas main_loss_fn)
                # -- comparable a travers tout l'historique meme si loss_type change.
                val_losses.append(weighted_mse_loss(pred, heatmap_target).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        elapsed = time.time() - t0
        if history:
            prev_best_epoch = max((h["epoch"] for h in history if h["val_loss"] == best_val_loss), default=None)
            best_info = f"[meilleur actuel: epoque {prev_best_epoch}, val_loss={best_val_loss:.5f}]"
        else:
            best_info = "[premiere epoque, pas encore de meilleur]"
        print(f"epoch {epoch + 1}/{start_epoch + max_epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
              f"({elapsed:.1f}s)  {best_info}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "seconds": elapsed})

        # checkpoint "dernier etat" (toujours ecrase, pour reprise apres crash)
        torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss,
                    "norm_type": norm_type}, latest_path)

        # val_improved/ema_improved decident ENSEMBLE du reset du compteur de
        # patience (voir plus bas) -- avant, seule une amelioration de
        # val_loss le resetait, ce qui pouvait arreter l'entrainement trop
        # tot si le modele "brut" plafonne pendant que l'EMA continue de
        # progresser discretement (frequent une fois proche de la convergence).
        val_improved = val_loss < best_val_loss - min_delta
        if val_improved:
            improvement_str = f"amelioration de {best_val_loss - val_loss:.5f}" if best_val_loss != float("inf") else "premier resultat"
            best_val_loss = val_loss
            torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "val_loss": val_loss,
                        "norm_type": norm_type}, weights_path)
            print(f"  -> NOUVEAU MEILLEUR ({improvement_str}), checkpoint sauvegarde: {weights_path}")
        else:
            print(f"  -> pas d'amelioration (val_loss)")

        ema_improved = False
        if use_ema:
            # EMA evaluee chaque epoque (peu couteux -- juste un passage de
            # plus sur le petit set de validation), meilleur checkpoint EMA
            # garde a part -- souvent legerement meilleur/plus stable que le
            # modele "brut" une fois proche du plateau.
            ema_model.eval()
            ema_val_losses = []
            with torch.no_grad():
                for patch, target, _idx in val_loader:
                    patch, target = patch.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    pred = ema_model(patch)
                    heatmap_target = target[:, 0:1] if predict_distance else target
                    ema_val_losses.append(weighted_mse_loss(pred, heatmap_target).item())
            ema_val_loss = float(np.mean(ema_val_losses)) if ema_val_losses else float("nan")
            history[-1]["ema_val_loss"] = ema_val_loss
            ema_improved = ema_val_loss < best_ema_val_loss - min_delta
            if ema_improved:
                best_ema_val_loss = ema_val_loss
                ema_path = WEIGHTS_DIR / f"{model_name}_ema.pt"
                torch.save({"model_state": ema_model.state_dict(), "epoch": epoch + 1, "val_loss": ema_val_loss,
                            "norm_type": norm_type}, ema_path)
                print(f"  -> EMA val_loss={ema_val_loss:.5f} : NOUVEAU MEILLEUR (EMA), sauvegarde: {ema_path}")
            else:
                print(f"  -> EMA val_loss={ema_val_loss:.5f}  (meilleur EMA actuel: {best_ema_val_loss:.5f})")

        # arret automatique base sur les DEUX signaux : soit le modele
        # "brut" soit l'EMA suffit a montrer que l'entrainement progresse
        # encore -- reset le compteur si l'un ou l'autre s'ameliore.
        if val_improved or ema_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(f"  -> {epochs_without_improvement}/{patience} epoques sans amelioration (val_loss ou EMA) avant arret automatique")

        if hard_mining and train_sampler is not None and (epoch + 1) >= hard_mining_start_epoch:
            # repondere le tirage pour la prochaine epoque a partir de la
            # difficulte REELLEMENT observee -- ecrase naturellement les
            # poids de curriculum (statiques, a priori) une fois ce point
            # atteint. Si hard_mining=False (curriculum seul), ce bloc ne
            # s'execute jamais : les poids de curriculum restent actifs pour
            # toute la duree de l'entrainement.
            w = np.clip(sample_loss, 1e-6, None) ** hard_mining_power
            w = 0.5 * (w / w.mean()) + 0.5
            train_sampler.weights = torch.as_tensor(w, dtype=torch.double)

        if real_score_every and (epoch + 1) % real_score_every == 0:
            # controle periodique, plus couteux (detection + tracking sur des
            # volumes entiers) : verifie que val_loss (proxy) et la vraie
            # metrique de competition restent alignees, et garde a part le
            # meilleur checkpoint au sens de CETTE metrique.
            eval_t0 = time.time()
            was_training = model.training
            model.eval()
            try:
                real_score = _evaluate_real_score(model, model_name, val_names, device, n_datasets=real_score_datasets)
            except Exception as exc:
                print(f"  (evaluation du vrai score echouee, ignoree: {exc})")
                real_score = None
            if was_training:
                model.train()
            if real_score is not None:
                history[-1]["real_score"] = real_score
                if real_score > best_real_score:
                    best_real_score = real_score
                    bestreal_path = WEIGHTS_DIR / f"{model_name}_bestreal.pt"
                    torch.save({"model_state": model.state_dict(), "epoch": epoch + 1, "real_score": real_score,
                                "norm_type": norm_type}, bestreal_path)
                    print(f"  -> vrai score = {real_score:.4f} ({time.time() - eval_t0:.1f}s) : NOUVEAU MEILLEUR "
                          f"(par score reel), sauvegarde: {bestreal_path}")
                else:
                    print(f"  -> vrai score = {real_score:.4f} ({time.time() - eval_t0:.1f}s)  "
                          f"(meilleur reel actuel: {best_real_score:.4f})")

        if pseudo_label_every and (epoch + 1) % pseudo_label_every == 0:
            # densifie le signal sparse en ajoutant les detections a HAUTE
            # confiance non-annotees -- voir avertissement de risque dans
            # _pseudo_label_pass. Uniquement train_ds (jamais val_ds).
            pl_t0 = time.time()
            n_added, n_frames = _pseudo_label_pass(
                model, train_ds, train_names, device,
                threshold=pseudo_label_threshold, min_distance=pseudo_label_min_distance,
                max_frames=pseudo_label_max_frames, seed=seed + epoch,
            )
            print(f"  -> pseudo-labellisation: {n_added} nouveaux points ajoutes sur "
                  f"{n_frames} frames examinees ({time.time() - pl_t0:.1f}s)")

        log_path.write_text(json.dumps(history, indent=2))

        if swa_snapshots is not None:
            swa_snapshots.append(copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}))

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss) if scheduler_needs_val_loss else scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        if lr_after < lr_before:
            reason = "descente cosine" if lr_schedule == "cosine" else "val_loss stagne"
            print(f"  -> lr reduit: {lr_before:.2e} -> {lr_after:.2e} ({reason})")
        elif lr_after > lr_before:
            print(f"  -> lr remonte: {lr_before:.2e} -> {lr_after:.2e} (redemarrage cosine)")

        if epochs_without_improvement >= patience:
            print(f"\nArret automatique (early stopping) : pas d'amelioration depuis {patience} epoques.")
            break
    else:
        print(f"\nmax_epochs ({max_epochs}) atteint sans early stopping -- relance le script pour continuer.")

    if swa_snapshots is not None and len(swa_snapshots) >= 2:
        print(f"\nSWA : moyenne des {len(swa_snapshots)} derniers etats du modele...")
        avg_state = {}
        for k in swa_snapshots[0].keys():
            stacked = torch.stack([sd[k].float() for sd in swa_snapshots], dim=0)
            avg_state[k] = stacked.mean(dim=0)

        swa_model = HeatmapUNet3D(**arch_kwargs).to(device)
        swa_model.load_state_dict(avg_state)

        # une simple moyenne des poids rend les stats BatchNorm (running_mean/
        # running_var) incoherentes -- il faut un passage avant (sans
        # retropropagation) sur des donnees d'entrainement pour les recalculer,
        # c'est l'etape "update_bn" standard de SWA. Sans objet si norm_type=
        # "groupnorm" (aucune statistique cumulee -- la boucle ne trouve alors
        # aucun BatchNorm3d et ne fait rien).
        has_batchnorm = any(isinstance(m, nn.BatchNorm3d) for m in swa_model.modules())
        if has_batchnorm:
            for module in swa_model.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.reset_running_stats()
                    module.momentum = None  # moyenne cumulative plutot qu'exponentielle
            swa_model.train()
            with torch.no_grad():
                bn_update_iter = iter(train_loader)
                for _ in range(30):
                    try:
                        bn_patch, _, _idx = next(bn_update_iter)
                    except StopIteration:
                        break
                    swa_model(bn_patch.to(device))

        swa_model.eval()
        swa_val_losses = []
        with torch.no_grad():
            for patch, target, _idx in val_loader:
                patch, target = patch.to(device, non_blocking=True), target.to(device, non_blocking=True)
                pred = swa_model(patch)
                heatmap_target = target[:, 0:1] if predict_distance else target
                swa_val_losses.append(weighted_mse_loss(pred, heatmap_target).item())
        swa_val_loss = float(np.mean(swa_val_losses)) if swa_val_losses else float("nan")

        swa_path = WEIGHTS_DIR / f"{model_name}_swa.pt"
        torch.save({"model_state": swa_model.state_dict(), "val_loss": swa_val_loss, "norm_type": norm_type}, swa_path)
        verdict = "MEILLEUR que le modele classique" if swa_val_loss < best_val_loss else "pas meilleur, garde a titre de reference"
        print(f"SWA val_loss={swa_val_loss:.5f}  (meilleur classique={best_val_loss:.5f}) -> {verdict}")
        print(f"Sauvegarde: {swa_path}")

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
    parser.add_argument("--lr-patience", type=int, default=3, help="epoques sans amelioration avant reduction de lr")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="facteur de reduction de lr (0.5 = divise par 2)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="norme max des gradients (0 pour desactiver)")
    parser.add_argument("--swa-last-n", type=int, default=5,
                         help="moyenne des N derniers etats du modele en fin d'entrainement (Stochastic Weight "
                              "Averaging). 0 ou 1 pour desactiver.")
    parser.add_argument("--max-datasets", type=int, default=None, help="limiter le nb de datasets (debug/demo -- omettre pour utiliser TOUTE la base)")
    parser.add_argument("--val-samples", type=int, default=1000,
                         help="nb de points de validation (sous-echantillonne) -- augmente de 400 a 1000 pour un "
                              "signal de plateau moins bruite")
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
    parser.add_argument("--dropout", type=float, default=0.1,
                         help="Dropout3d au bottleneck (0.0 pour desactiver, comportement d'origine)")
    parser.add_argument("--no-deep-supervision", action="store_false", dest="deep_supervision",
                         help="desactive les pertes auxiliaires aux etages intermediaires du decodeur "
                              "(actives par defaut)")
    parser.add_argument("--aux-loss-weight", type=float, default=0.3,
                         help="poids des pertes auxiliaires (deep supervision) dans la perte totale")
    parser.add_argument("--lr-schedule", choices=["plateau", "cosine"], default="plateau",
                         help="'plateau' = ReduceLROnPlateau (reduit si val_loss stagne), "
                              "'cosine' = CosineAnnealingWarmRestarts (redemarrages periodiques, "
                              "aide a s'echapper des plateaux/minima locaux)")
    parser.add_argument("--model-name", default="heatmap_unet",
                         help="nom du modele -- determine les fichiers de checkpoint "
                              "(weights/<nom>.pt, <nom>_latest.pt, <nom>_swa.pt) et le log "
                              "(train_log_<nom>.json). Utiliser des noms differents pour entrainer "
                              "plusieurs modeles independants (ensemble).")
    parser.add_argument("--seed", type=int, default=0,
                         help="graine aleatoire -- varier entre modeles d'un meme ensemble pour "
                              "obtenir une diversite utile (init des poids + ordre des augmentations)")
    parser.add_argument("--no-hard-mining", action="store_false", dest="hard_mining",
                         help="desactive le mining d'exemples difficiles (tirage uniforme classique, "
                              "actif par defaut)")
    parser.add_argument("--hard-mining-start-epoch", type=int, default=3,
                         help="n'ajuste le tirage qu'a partir de cette epoque -- le signal de "
                              "difficulte des toutes premieres epoques est trop bruite pour etre utile")
    parser.add_argument("--hard-mining-power", type=float, default=1.0,
                         help="exposant applique a la difficulte observee avant normalisation -- "
                              "plus grand = concentre davantage le tirage sur les points les plus durs")
    parser.add_argument("--real-score-every", type=int, default=0,
                         help="toutes les N epoques, evalue la VRAIE metrique de competition "
                              "(detection+tracking+jaccard sur des datasets de validation entiers, "
                              "plus couteux que val_loss) et garde a part le meilleur checkpoint selon "
                              "ce score reel (weights/<nom>_bestreal.pt). 0 pour desactiver (defaut).")
    parser.add_argument("--real-score-datasets", type=int, default=2,
                         help="nombre de datasets de validation utilises pour l'evaluation periodique "
                              "du vrai score (plus = plus fiable mais plus lent)")
    parser.add_argument("--loss-type", choices=["weighted_mse", "focal"], default="focal",
                         help="'weighted_mse' = MSE ponderee classique (comportement d'origine), "
                              "'focal' (defaut) = perte focale type CenterNet, concentre le gradient sur "
                              "les vrais desaccords. N'affecte QUE la perte d'entrainement -- val_loss "
                              "reste toujours en weighted_mse pour rester comparable a l'historique.")
    parser.add_argument("--no-predict-distance", action="store_false", dest="predict_distance",
                         help="desactive la tache auxiliaire de distance-transform (active par defaut)")
    parser.add_argument("--dist-loss-weight", type=float, default=0.2,
                         help="poids de la perte de distance-transform dans la perte totale")
    parser.add_argument("--norm-type", choices=["batch", "groupnorm"], default="batch",
                         help="'batch' (defaut, compatibilite totale avec les checkpoints existants) ou "
                              "'groupnorm' (independant de la taille de batch, potentiellement plus stable "
                              "avec de petits sous-batchs -- CHANGE le type de couche partout dans le "
                              "reseau, voir avertissement dans dl_model.py, pas un simple ajout)")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                         help="moyenne mobile exponentielle CONTINUE des poids (mise a jour a chaque step, "
                              "contrairement a SWA qui n'agit qu'en fin d'entrainement), meilleur checkpoint "
                              "garde a part (weights/<nom>_ema.pt). 0 pour desactiver.")
    parser.add_argument("--warmup-steps", type=int, default=300,
                         help="montee lineaire du lr sur les N premiers pas d'un entrainement FRESH (pas "
                              "au resume) avant que le scheduler ne prenne le relais. 0 pour desactiver.")
    parser.add_argument("--use-sam", action="store_true",
                         help="active l'optimiseur SAM (Sharpness-Aware Minimization) -- cherche des minima "
                              "plats, generalise souvent mieux, mais DOUBLE le cout par step (deux "
                              "forward/backward). Incompatible avec --cpu-parallel-fraction (desactive "
                              "automatiquement). Off par defaut vu le cout.")
    parser.add_argument("--sam-rho", type=float, default=0.05,
                         help="rayon du voisinage explore par SAM (n'a d'effet que si --use-sam)")
    parser.add_argument("--no-curriculum", action="store_false", dest="curriculum",
                         help="desactive le curriculum learning (actif par defaut) -- sans lui, le tirage "
                              "est uniforme jusqu'a ce que le mining d'exemples difficiles s'active")
    parser.add_argument("--curriculum-radius", type=float, default=20.0,
                         help="rayon (voxels) utilise pour estimer la densite locale de points annotes "
                              "(proxy de difficulte a priori pour le curriculum)")
    parser.add_argument("--pseudo-label-every", type=int, default=0,
                         help="toutes les N epoques, ajoute au pool d'annotations d'entrainement les "
                              "detections a HAUTE confiance non-annotees (densifie le signal sparse). "
                              "RISQUE: peut renforcer les erreurs du modele si mal calibre -- 0 pour "
                              "desactiver (defaut, prudence recommandee).")
    parser.add_argument("--pseudo-label-threshold", type=float, default=0.9,
                         help="confiance minimale (heatmap) pour qu'une detection devienne un pseudo-label")
    parser.add_argument("--pseudo-label-min-distance", type=int, default=6,
                         help="distance minimale (voxels) a un point deja annote pour compter comme nouveau")
    parser.add_argument("--pseudo-label-max-frames", type=int, default=5,
                         help="nombre de frames examinees a chaque passe de pseudo-labellisation")
    args = parser.parse_args()

    run_training(
        args.max_epochs, args.steps_per_epoch, args.batch_size, args.lr,
        max_datasets=args.max_datasets, val_samples=args.val_samples,
        num_workers=args.num_workers, resume=not args.no_resume,
        save_every_steps=args.save_every_steps, patience=args.patience,
        gpu_duty_cycle=args.gpu_duty_cycle, cpu_parallel_fraction=args.cpu_parallel_fraction,
        weight_decay=args.weight_decay, lr_patience=args.lr_patience, lr_factor=args.lr_factor,
        grad_clip=args.grad_clip, swa_last_n=args.swa_last_n,
        dropout=args.dropout, deep_supervision=args.deep_supervision,
        aux_loss_weight=args.aux_loss_weight, lr_schedule=args.lr_schedule,
        model_name=args.model_name, seed=args.seed,
        hard_mining=args.hard_mining, hard_mining_start_epoch=args.hard_mining_start_epoch,
        hard_mining_power=args.hard_mining_power,
        real_score_every=args.real_score_every, real_score_datasets=args.real_score_datasets,
        loss_type=args.loss_type, predict_distance=args.predict_distance,
        dist_loss_weight=args.dist_loss_weight, norm_type=args.norm_type,
        ema_decay=args.ema_decay, warmup_steps=args.warmup_steps,
        use_sam=args.use_sam, sam_rho=args.sam_rho,
        curriculum=args.curriculum, curriculum_radius=args.curriculum_radius,
        pseudo_label_every=args.pseudo_label_every, pseudo_label_threshold=args.pseudo_label_threshold,
        pseudo_label_min_distance=args.pseudo_label_min_distance,
        pseudo_label_max_frames=args.pseudo_label_max_frames,
    )
