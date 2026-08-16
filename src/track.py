"""
ETAPE 2 : Tracking (association temporelle) + detection des divisions.

On prend les detections par frame (sortie de detect.py) et on les relie dans le
temps avec laptrack, qui resout un probleme d'assignation lineaire (LAP) :
  - lien normal t -> t+1 : cout = distance physique au carre entre centroides
  - division (split) : une cellule a t peut avoir DEUX enfants a t+1 si le cout
    (distance physique) de chacun des deux liens est sous le seuil
  - gap closing : si une cellule "disparait" une frame (detection manquee) puis
    reapparait, on essaie de reconnecter par-dessus le trou (jusqu'a
    gap_closing_max_frame_count frames sautees)

Le seuil utilise (cutoff) correspond a la distance max de 7.0 um utilisee par la
metrique officielle de la competition, en distance physique au carre
(metric="sqeuclidean"), avec l'echelle z=1.625, y=x=0.40625 um/voxel.

Sortie :
  data/synth1_track_df.csv  -> une ligne par detection, avec track_id / tree_id
  data/synth1_split_df.csv  -> les evenements de division (parent_track_id -> child_track_id)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from laptrack import LapTrack

ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "synth1"
VOXEL_SIZE_UM = np.array([1.625, 0.40625, 0.40625])  # z, y, x
MAX_LINK_DIST_UM = 7.0


def run_tracking(dataset_name):
    det_path = ROOT / "data" / f"{dataset_name}_detections.csv"
    df = pd.read_csv(det_path)

    # coordonnees physiques (um) pour le cout de tracking -- IMPORTANT : le z
    # compte plus que y/x a cause de l'anisotropie du voxel (1.625 vs 0.40625)
    df["z_um"] = df["z"] * VOXEL_SIZE_UM[0]
    df["y_um"] = df["y"] * VOXEL_SIZE_UM[1]
    df["x_um"] = df["x"] * VOXEL_SIZE_UM[2]

    cutoff_sq = MAX_LINK_DIST_UM ** 2  # metric="sqeuclidean" -> cutoff en distance^2

    lt = LapTrack(
        metric="sqeuclidean",
        cutoff=cutoff_sq,
        gap_closing_cutoff=cutoff_sq,
        gap_closing_max_frame_count=1,
        splitting_cutoff=cutoff_sq,   # active la detection de divisions (1 -> 2)
    )

    track_df, split_df, merge_df = lt.predict_dataframe(
        df, coordinate_cols=["z_um", "y_um", "x_um"], frame_col="frame",
    )
    track_df = track_df.reset_index(drop=True)

    n_tracks = track_df["track_id"].nunique()
    n_trees = track_df["tree_id"].nunique()
    print(f"{len(track_df)} detections liees en {n_tracks} segments de piste (track_id),")
    print(f"regroupes en {n_trees} lignees (tree_id).")
    print(f"{len(split_df)} evenement(s) de division detecte(s):")
    if not split_df.empty:
        print(split_df.to_string(index=False))
    if not merge_df.empty:
        print(f"ATTENTION: {len(merge_df)} evenement(s) de fusion detecte(s) (inhabituel pour des cellules)")

    track_out = ROOT / "data" / f"{dataset_name}_track_df.csv"
    split_out = ROOT / "data" / f"{dataset_name}_split_df.csv"
    track_df.to_csv(track_out, index=False)
    split_df.to_csv(split_out, index=False)
    print(f"\nEcrit: {track_out}")
    print(f"Ecrit: {split_out}")
    return track_df, split_df


if __name__ == "__main__":
    run_tracking(DATASET_NAME)
