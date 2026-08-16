"""Tracking sur des detections reelles (equivalent de track.py), avec l'echelle
physique lue dynamiquement depuis les metadonnees du dataset plutot que codee
en dur."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from laptrack import LapTrack

MAX_LINK_DIST_UM = 7.0
SPLITTING_DIST_UM = 7.0     # seuil de cout pour qu'un split soit CANDIDAT
MIN_DAUGHTER_PERSISTENCE = 4  # frames min. de survie des DEUX filles pour valider le split (tune_division.py)


def filter_spurious_divisions(track_df, split_df, min_persistence=MIN_DAUGHTER_PERSISTENCE):
    """Rejette les divisions candidates dont au moins une des deux filles ne
    survit pas assez longtemps (bruit de detection cree souvent un split
    d'une seule frame avant que la piste ne s'arrete) -- observe
    empiriquement comme la cause dominante des fausses divisions (des
    dizaines a des centaines de FP contre 0-2 TP, quel que soit le
    detecteur)."""
    if split_df.empty:
        return split_df

    track_lengths = track_df.groupby("track_id").size()

    def both_daughters_persist(row):
        return (track_lengths.get(row.parent_track_id, 0) >= 1
                and track_lengths.get(row.child_track_id, 0) >= min_persistence)

    # une division a deux lignes dans split_df (une par fille) partageant le
    # meme parent_track_id -- on ne garde le parent que si SES DEUX filles
    # passent le filtre
    split_df = split_df.copy()
    split_df["_persists"] = split_df.apply(both_daughters_persist, axis=1)
    valid_parents = split_df.groupby("parent_track_id")["_persists"].transform("all")
    return split_df[valid_parents].drop(columns="_persists").reset_index(drop=True)


def run_tracking(detections_df, scale_zyx, min_daughter_persistence=MIN_DAUGHTER_PERSISTENCE):
    if detections_df.empty:
        # laptrack plante sur un pool de detections vide (dataset degenere) --
        # on retourne des tables vides mais bien formees plutot que de
        # laisser planter tout le pipeline.
        empty_track = pd.DataFrame(columns=["frame", "z", "y", "x", "track_id", "tree_id"])
        empty_pairs = pd.DataFrame(columns=["parent_track_id", "child_track_id"])
        return empty_track, empty_pairs.copy(), empty_pairs.copy()

    df = detections_df.copy()
    df["z_um"] = df["z"] * scale_zyx[0]
    df["y_um"] = df["y"] * scale_zyx[1]
    df["x_um"] = df["x"] * scale_zyx[2]

    cutoff_sq = MAX_LINK_DIST_UM ** 2
    lt = LapTrack(
        metric="sqeuclidean",
        cutoff=cutoff_sq,
        gap_closing_cutoff=cutoff_sq,
        gap_closing_max_frame_count=1,
        splitting_cutoff=SPLITTING_DIST_UM ** 2,
    )
    track_df, split_df, merge_df = lt.predict_dataframe(
        df, coordinate_cols=["z_um", "y_um", "x_um"], frame_col="frame",
    )
    track_df = track_df.reset_index(drop=True)

    if min_daughter_persistence:
        split_df = filter_spurious_divisions(track_df, split_df, min_daughter_persistence)

    return track_df, split_df, merge_df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    dataset_name = "44b6_0113de3b"
    scale_zyx = np.array([1.625, 0.40625, 0.40625])  # lue par real_detect.py, reportee ici

    det_path = ROOT / "data" / f"{dataset_name}_real_detections.csv"
    detections_df = pd.read_csv(det_path)

    print(f"Tracking sur {len(detections_df)} detections...")
    track_df, split_df, merge_df = run_tracking(detections_df, scale_zyx)

    n_tracks = track_df["track_id"].nunique()
    n_trees = track_df["tree_id"].nunique()
    print(f"{len(track_df)} detections liees en {n_tracks} segments (track_id), {n_trees} lignees (tree_id)")
    print(f"{len(split_df)} division(s) detectee(s)")
    if len(merge_df):
        print(f"ATTENTION: {len(merge_df)} fusion(s) detectee(s) (inhabituel)")

    track_df.to_csv(ROOT / "data" / f"{dataset_name}_real_track_df.csv", index=False)
    split_df.to_csv(ROOT / "data" / f"{dataset_name}_real_split_df.csv", index=False)
    print(f"\nEcrit: {dataset_name}_real_track_df.csv / _real_split_df.csv")
