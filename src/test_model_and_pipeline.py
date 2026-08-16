"""
Suite de verifications AVANT un entrainement long : pas de tests unitaires
formels (pas de framework de test dans ce projet), mais une serie de checks
executables qui valident concretement les points a risque plutot que de se
fier a une relecture visuelle du code.

Verifie :
  1. Forward + backward sans NaN/Inf, sortie sigmoid bien dans [0, 1]
  2. BatchNorm3d : train() plante sur batch_size=1 (comportement attendu de
     PyTorch) -- documente pourquoi drop_last=True est necessaire sur le
     DataLoader d'entrainement
  3. BatchNorm3d : eval() ne plante PAS sur batch_size=1 (utilise les
     statistiques courantes, pas celles du batch) -- confirme que
     real_detect_dl.py (batch_size=1 par frame) est sur
  4. Le garde-fou "volume plus petit que patch_shape" (train_detector.py)
     produit bien un tenseur de la bonne forme, sans crash
  5. Le DataLoader avec num_workers>0 (multiprocessing) fonctionne sur un
     petit sous-ensemble reel, sur PLUSIEURS iterations successives
     (persistent_workers=True), sans exception ni deadlock
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from dl_model import HeatmapUNet3D

results = []


def check(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"[OK]   {name}")
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


def test_forward_backward_finite():
    model = HeatmapUNet3D()
    x = torch.rand(4, 1, 32, 64, 64)
    target = torch.rand(4, 1, 32, 64, 64)
    pred = model(x)
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    assert torch.isfinite(pred).all(), "sortie non-finie (NaN/Inf)"
    assert pred.min() >= 0.0 and pred.max() <= 1.0, f"sortie hors [0,1]: [{pred.min()},{pred.max()}]"

    loss = ((pred - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"gradient manquant pour {name}"
        assert torch.isfinite(p.grad).all(), f"gradient non-fini pour {name}"


def test_extreme_inputs():
    """Entree toute a zero / toute a un -- ne doit produire ni NaN ni crash."""
    model = HeatmapUNet3D()
    model.eval()
    with torch.no_grad():
        for val in [0.0, 1.0]:
            x = torch.full((1, 1, 32, 64, 64), val)
            out = model(x)
            assert torch.isfinite(out).all(), f"NaN/Inf sur entree constante={val}"


def test_batchnorm_train_batch1_is_actually_safe():
    """A priori BatchNorm3d peut planter en train() avec batch_size=1 (variance
    non definie sur un seul echantillon) -- mais BatchNorm3d normalise sur
    (N, D, H, W), pas seulement N. Au bottleneck (patch 32x64x64 -> 4x8x8
    apres 3 poolings), il reste 4*8*8=256 elements meme avec un batch de 1,
    largement assez pour une variance non-degeneree. Ce test le CONFIRME
    empiriquement plutot que de supposer un risque qui n'existe pas ici --
    ca aurait ete le cas avec un modele qui pool jusqu'a du 1x1x1 (ex: un
    classifieur), ce qui n'est pas notre architecture."""
    model = HeatmapUNet3D()
    model.train()
    x = torch.rand(1, 1, 32, 64, 64)
    out = model(x)  # ne doit PAS lever d'exception
    assert torch.isfinite(out).all()


def test_batchnorm_eval_batch1_ok():
    """En eval(), BatchNorm utilise les stats courantes (pas celles du
    batch) -- batch_size=1 doit fonctionner sans probleme. C'est le mode
    utilise par real_detect_dl.py (une frame = un batch de taille 1)."""
    model = HeatmapUNet3D()
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 1, 32, 64, 64)
        out = model(x)
    assert torch.isfinite(out).all()


def test_small_volume_patch_padding():
    """Simule un volume plus petit que PATCH_SHAPE sur l'axe Z (jamais le cas
    sur les vraies donnees, mais le garde-fou de train_detector.py doit
    quand meme produire un tenseur de la bonne forme sans planter."""
    import train_detector as td

    class FakeArr:
        shape = (5, 16, 100, 100)  # T=5, Z=16 < PATCH_SHAPE[0]=32, Y=X=100
        def __getitem__(self, t):
            return np.random.rand(16, 100, 100).astype(np.float32) * 500

    ds = td.SparsePointPatchDataset.__new__(td.SparsePointPatchDataset)
    ds.patch_shape = td.PATCH_SHAPE
    ds.by_frame = {("fake", 0): [(8, 50, 50)]}
    ds._zarr_cache = {"fake": FakeArr()}
    ds._quantile_cache = {"fake": {"0.1": 20.0, "0.999": 480.0}}

    import pandas as pd
    ds.rows = pd.DataFrame([{"dataset": "fake", "t": 0, "z": 8, "y": 50, "x": 50}])

    patch, target = ds[0]
    assert patch.shape == (1, *td.PATCH_SHAPE), f"forme patch incorrecte: {patch.shape}"
    assert target.shape == (1, *td.PATCH_SHAPE), f"forme target incorrecte: {target.shape}"
    assert torch.isfinite(patch).all() and torch.isfinite(target).all()


def test_dataloader_multiworker_multiple_iterations():
    """Cree un DataLoader avec num_workers>0 sur un vrai petit sous-ensemble
    de donnees, et tire des batches sur PLUSIEURS iterations successives
    (comme le fait train_detector.py a chaque 'epoque') pour verifier que
    persistent_workers=True ne deadlock pas et ne renvoie pas de donnees
    corrompues/incoherentes entre iterations."""
    import train_detector as td
    from torch.utils.data import DataLoader

    nodes_df = td.ensure_gt_index()
    names = sorted(nodes_df.dataset.unique())[:3]  # 3 datasets seulement, rapide
    ds = td.SparsePointPatchDataset(nodes_df, names)
    assert len(ds) > 0, "dataset de test vide, verifie l'index GT"

    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2,
                         persistent_workers=True, drop_last=True)

    seen_shapes = set()
    for iteration in range(3):  # simule 3 "epoques"
        it = iter(loader)
        for _ in range(3):
            patch, target = next(it)
            seen_shapes.add(tuple(patch.shape))
            assert patch.shape == target.shape
            assert torch.isfinite(patch).all() and torch.isfinite(target).all()

    assert seen_shapes == {(4, 1, *td.PATCH_SHAPE)}, f"formes incoherentes rencontrees: {seen_shapes}"


def test_patch_target_coordinate_alignment():
    """Verifie qu'il n'y a PAS d'inversion d'axes (bug classique en 3D :
    confondre z/y/x quelque part dans le pipeline). Pour un vrai point
    annote, le pic de la cible gaussienne doit tomber exactement au centre
    du patch (a nz-z0 pres), ET l'intensite brute doit etre localement
    elevee a cet endroit (un vrai centre de cellule est brillant, pas dans
    le fond)."""
    import train_detector as td

    nodes_df = td.ensure_gt_index()
    ds = td.SparsePointPatchDataset(nodes_df, sorted(nodes_df.dataset.unique())[:1])

    # choisit un point suffisamment loin des bords du volume pour que le
    # clipping de train_detector.py ne decale PAS le crop (sinon le pic
    # n'est legitimement pas au centre, ce qui n'a rien a voir avec un bug
    # d'inversion d'axes -- voir Z=64/X=256 avec patch (32,64,64))
    pz, py, px = ds.patch_shape
    arr0, _ = ds._get_volume(ds.rows.iloc[0].dataset)
    Z, Y, X = arr0.shape[1:]
    margin = ds.rows.apply(
        lambda r: min(r.z - pz // 2, Z - pz // 2 - r.z, r.y - py // 2, Y - py // 2 - r.y,
                       r.x - px // 2, X - px // 2 - r.x),
        axis=1,
    )
    interior_idx = margin[margin > 2].index
    assert len(interior_idx) > 0, "aucun point interieur trouve pour le test (verifie le dataset choisi)"
    idx = interior_idx[0]

    patch, target = ds[idx]
    patch, target = patch[0].numpy(), target[0].numpy()  # retire le canal

    # le pic de la cible doit etre proche du centre du patch (le point annote
    # a ete utilise pour centrer le crop)
    peak = np.unravel_index(np.argmax(target), target.shape)
    center = np.array(td.PATCH_SHAPE) // 2
    dist_to_center = np.abs(np.array(peak) - center)
    assert (dist_to_center <= 2).all(), f"pic de la cible trop loin du centre du patch: {peak} vs centre {tuple(center)}"

    # au voisinage immediat du pic de la cible, l'intensite brute doit etre
    # nettement au-dessus de la mediane du patch (un centre de cellule est
    # lumineux, pas dans le fond) -- sinon les coordonnees ne pointent pas
    # vraiment sur une cellule
    z, y, x = peak
    r = 2
    local_patch = patch[max(z - r, 0):z + r + 1, max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    assert local_patch.max() > np.median(patch), "intensite brute au centre annote pas au-dessus du fond -- possible inversion d'axes"


def test_concurrent_worker_stress():
    """Stress-test plus consequent : 6 workers, ~25 datasets differents,
    plusieurs dizaines de batches tires, pour s'assurer qu'un acces concurrent
    de plusieurs process au MEME fichier zarr (statistiquement frequent avec
    6 workers sur un pool de datasets) ne cause ni exception, ni hang, ni
    donnees incoherentes (formes/valeurs)."""
    import time
    import train_detector as td
    from torch.utils.data import DataLoader

    nodes_df = td.ensure_gt_index()
    names = sorted(nodes_df.dataset.unique())[:25]
    ds = td.SparsePointPatchDataset(nodes_df, names)

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=6,
                         persistent_workers=True, prefetch_factor=4, drop_last=True)

    t0 = time.time()
    n_batches = 0
    it = iter(loader)
    for _ in range(30):
        patch, target = next(it)
        assert patch.shape == (8, 1, *td.PATCH_SHAPE)
        assert torch.isfinite(patch).all() and torch.isfinite(target).all()
        n_batches += 1
    elapsed = time.time() - t0
    assert elapsed < 120, f"trop lent ou bloque: {elapsed:.1f}s pour {n_batches} batches"
    print(f"       ({n_batches} batches, {elapsed:.1f}s, {n_batches * 8 / elapsed:.1f} patchs/s)")


if __name__ == "__main__":
    check("forward+backward finis, sortie dans [0,1]", test_forward_backward_finite)
    check("entrees extremes (tout-0 / tout-1) sans NaN", test_extreme_inputs)
    check("BatchNorm train() batch=1 -> reste sur (256 elements au bottleneck)", test_batchnorm_train_batch1_is_actually_safe)
    check("BatchNorm eval() batch=1 -> OK (utilise par l'inference)", test_batchnorm_eval_batch1_ok)
    check("garde-fou volume < patch_shape", test_small_volume_patch_padding)
    check("DataLoader multi-workers, plusieurs iterations (persistent_workers)", test_dataloader_multiworker_multiple_iterations)
    check("alignement patch/cible (pas d'inversion d'axes z/y/x)", test_patch_target_coordinate_alignment)
    check("stress-test acces concurrent (6 workers, 25 datasets, 30 batches)", test_concurrent_worker_stress)

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} verifications passees")
    sys.exit(1 if n_fail else 0)
