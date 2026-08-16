"""
Genere un mini jeu de donnees 3D+t synthetique pour tester le pipeline de bout en bout
sans avoir besoin des vraies donnees Kaggle.

Produit :
  data/synth1.zarr        -> volume (T, Z, Y, X), groupe zarr avec array "0"
  data/synth1_ground_truth.csv -> verite terrain au format EXACT de la competition
                                   (mêmes colonnes que submission.csv)

La simulation :
  - 4 cellules initiales, positions aleatoires, vitesses aleatoires (mouvement brownien + derive)
  - 2 evenements de division programmes (une cellule mere -> deux filles)
  - chaque cellule est rendue comme un blob gaussien anisotrope dans le volume
  - un bruit de fond gaussien est ajoute pour imiter le bruit de microscopie
"""

import numpy as np
import zarr
import pandas as pd
from pathlib import Path

RNG_SEED = 42
T = 8
Z, Y, X = 24, 96, 96
VOXEL_SIZE_UM = np.array([1.625, 0.40625, 0.40625])  # z, y, x -- meme echelle que l'enonce
DATASET_NAME = "synth1"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class Cell:
    _next_id = 1

    def __init__(self, pos, vel, birth_t, parent_id=None):
        self.id = Cell._next_id
        Cell._next_id += 1
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.birth_t = birth_t
        self.parent_id = parent_id
        self.positions = {}  # t -> np.array([z, y, x]) en voxels
        self.alive = True


def simulate_tracks(rng):
    Cell._next_id = 1
    cells0 = []
    for _ in range(4):
        pos = rng.uniform([4, 20, 20], [Z - 4, Y - 20, X - 20])
        vel = rng.uniform(-1.2, 1.2, size=3)
        cells0.append(Cell(pos, vel, birth_t=0))

    # deux divisions programmees : cellule 0 se divise au temps t=3,
    # cellule 2 se divise au temps t=5
    div_plan = {3: cells0[0], 5: cells0[2]}

    active = list(cells0)
    all_cells = list(cells0)
    division_events = []  # (t_division, mother_id, [daughter_ids])

    for t in range(T):
        for c in active:
            if c.alive and t >= c.birth_t:
                c.positions[t] = c.pos.copy()

        if t in div_plan:
            mother = div_plan[t]
            if mother.alive and t in mother.positions:
                mother.alive = False
                daughters = []
                for _ in range(2):
                    dvel = mother.vel + rng.uniform(-0.8, 0.8, size=3)
                    dpos = mother.pos + rng.uniform(-2.5, 2.5, size=3)
                    daughter = Cell(dpos, dvel, birth_t=t + 1, parent_id=mother.id)
                    active.append(daughter)
                    all_cells.append(daughter)
                    daughters.append(daughter.id)
                division_events.append((t, mother.id, daughters))

        for c in active:
            if c.alive and t in c.positions:
                step = c.vel + rng.normal(0, 0.35, size=3)
                c.pos = c.pos + step
                c.pos = np.clip(c.pos, [2, 8, 8], [Z - 2, Y - 8, X - 8])

    return all_cells, division_events


def render_volume(shape, cells, t, rng, sigma=(1.3, 3.0, 3.0), amplitude=180, bg=15, noise_std=6):
    vol = np.full(shape, bg, dtype=np.float32)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    for c in cells:
        if t not in c.positions:
            continue
        pz, py, px = c.positions[t]
        blob = amplitude * np.exp(
            -(((zz - pz) ** 2) / (2 * sigma[0] ** 2)
              + ((yy - py) ** 2) / (2 * sigma[1] ** 2)
              + ((xx - px) ** 2) / (2 * sigma[2] ** 2))
        )
        vol += blob
    vol += rng.normal(0, noise_std, size=shape)
    vol = np.clip(vol, 0, 255)
    return vol.astype(np.uint8)


def build_ground_truth_table(all_cells, division_events, dataset_name):
    """Construit une table au meme format que submission.csv (nodes + edges)."""
    rows = []
    row_id = 0

    # node_id unique par (cell.id, t) -> on encode simplement "cellid*100 + t"
    def node_id_for(cell_id, t):
        return cell_id * 100 + t

    for c in all_cells:
        for t, pos in sorted(c.positions.items()):
            rows.append({
                "id": row_id, "dataset": dataset_name, "row_type": "node",
                "node_id": node_id_for(c.id, t), "t": t,
                "z": int(round(pos[0])), "y": int(round(pos[1])), "x": int(round(pos[2])),
                "source_id": -1, "target_id": -1,
            })
            row_id += 1

    cells_by_id = {c.id: c for c in all_cells}
    mother_ids_dividing = {mid: t for (t, mid, _) in division_events}

    for c in all_cells:
        ts_sorted = sorted(c.positions.keys())
        for a, b in zip(ts_sorted[:-1], ts_sorted[1:]):
            if b == a + 1:
                rows.append({
                    "id": row_id, "dataset": dataset_name, "row_type": "edge",
                    "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                    "source_id": node_id_for(c.id, a), "target_id": node_id_for(c.id, b),
                })
                row_id += 1

    for t_div, mother_id, daughter_ids in division_events:
        mother_last_t = max(t for t in cells_by_id[mother_id].positions if t <= t_div)
        for did in daughter_ids:
            daughter_first_t = min(cells_by_id[did].positions.keys())
            rows.append({
                "id": row_id, "dataset": dataset_name, "row_type": "edge",
                "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": node_id_for(mother_id, mother_last_t),
                "target_id": node_id_for(did, daughter_first_t),
            })
            row_id += 1

    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    return pd.DataFrame(rows, columns=cols)


def main():
    rng = np.random.default_rng(RNG_SEED)
    all_cells, division_events = simulate_tracks(rng)

    print(f"{len(all_cells)} cellules simulees (dont {len(division_events)} divisions)")
    for t_div, mother_id, daughters in division_events:
        print(f"  division a t={t_div}: cellule {mother_id} -> filles {daughters}")

    volume = np.zeros((T, Z, Y, X), dtype=np.uint8)
    for t in range(T):
        volume[t] = render_volume((Z, Y, X), all_cells, t, rng)

    store_path = DATA_DIR / f"{DATASET_NAME}.zarr"
    group = zarr.open_group(str(store_path), mode="w")
    group.create_dataset("0", data=volume, chunks=(1, Z, Y, X), dtype="uint8")
    group.attrs["voxel_size_um"] = VOXEL_SIZE_UM.tolist()
    print(f"Volume zarr ecrit: {store_path}  shape={volume.shape} dtype={volume.dtype}")

    gt_df = build_ground_truth_table(all_cells, division_events, DATASET_NAME)
    gt_path = DATA_DIR / f"{DATASET_NAME}_ground_truth.csv"
    gt_df.to_csv(gt_path, index=False)
    n_nodes = (gt_df.row_type == "node").sum()
    n_edges = (gt_df.row_type == "edge").sum()
    print(f"Ground truth ecrite: {gt_path}  ({n_nodes} nodes, {n_edges} edges)")


if __name__ == "__main__":
    main()
