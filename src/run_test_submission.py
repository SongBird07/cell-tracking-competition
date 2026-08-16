"""
Pipeline complet sur le VRAI test set de la competition :
  pour chaque dataset de test/*.zarr -> detection -> tracking -> lignes de submission
puis ecrit un seul submission.csv combinant tous les datasets, comme l'exige
le format de soumission Kaggle ("chaque dataset du test set doit apparaitre").
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_io import open_zarr_volume, list_datasets
from real_detect import detect_dataset
from real_track import run_tracking
from real_build_submission import build_rows_for_dataset, write_submission
from config import ROOT, TEST_DIR, check_data_root

# NOTE : ce script est la baseline CLASSIQUE (seuillage), gardee comme
# reference/comparaison. Le generateur de submission.csv "officiel" du
# projet est desormais run_test_submission_dl.py (detecteur entraine),
# qui est aussi plus robuste (gere les erreurs par dataset).


def main():
    check_data_root(require_test=True)
    dataset_names = list_datasets(TEST_DIR)
    print(f"{len(dataset_names)} dataset(s) de test trouves: {dataset_names}\n")

    all_rows = []
    for name in dataset_names:
        t0 = time.time()
        zarr_path = TEST_DIR / f"{name}.zarr"
        print(f"=== {name} ===")

        print("  detection...")
        det_df, scale_zyx = detect_dataset(zarr_path, verbose=False)
        print(f"  {len(det_df)} detections")

        print("  tracking...")
        track_df, split_df, merge_df = run_tracking(det_df, scale_zyx)
        n_tracks = track_df["track_id"].nunique()
        print(f"  {len(track_df)} noeuds, {n_tracks} segments, {len(split_df)} division(s)")

        rows = build_rows_for_dataset(name, track_df, split_df)
        all_rows.extend(rows)

        print(f"  termine en {time.time() - t0:.1f}s\n")

    out_path = ROOT / "data" / "submission_classic_baseline.csv"
    submission = write_submission(all_rows, out_path)
    n_nodes = (submission.row_type == "node").sum()
    n_edges = (submission.row_type == "edge").sum()
    print(f"submission.csv final ecrit: {out_path}")
    print(f"  {n_nodes} nodes, {n_edges} edges, {submission.dataset.nunique()} datasets")


if __name__ == "__main__":
    main()
