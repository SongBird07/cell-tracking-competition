"""
Pipeline complet sur le VRAI test set, avec le detecteur DEEP LEARNING
(HeatmapUNet3D entraine) a la place du seuillage classique.
Remplace run_test_submission.py comme generateur du submission.csv final.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_io import list_datasets
from real_detect_dl import detect_dataset
from real_track import run_tracking
from real_build_submission import build_rows_for_dataset, write_submission
from config import TEST_DIR, SUBMISSION_PATH, check_data_root


def main():
    check_data_root(require_test=True)
    dataset_names = list_datasets(TEST_DIR)
    if not dataset_names:
        raise RuntimeError(f"Aucun .zarr trouve dans {TEST_DIR} -- rien a soumettre.")
    print(f"{len(dataset_names)} dataset(s) de test trouves: {dataset_names}\n")

    all_rows = []
    failed = []
    for name in dataset_names:
        t0 = time.time()
        zarr_path = TEST_DIR / f"{name}.zarr"
        print(f"=== {name} ===")

        try:
            print("  detection (deep learning)...")
            det_df, scale_zyx = detect_dataset(zarr_path, verbose=False)
            print(f"  {len(det_df)} detections")

            if det_df.empty:
                # Cas degenere (volume vide/corrompu, jamais observe en
                # pratique) : la consigne exige que CHAQUE dataset de test
                # apparaisse dans le CSV, meme sans vraie detection. On pose
                # donc un unique noeud de secours au centre du volume plutot
                # que de laisser le dataset totalement absent du fichier.
                print("  ATTENTION: aucune detection -- noeud de secours insere pour que le dataset apparaisse")
                import zarr
                shape = zarr.open_group(str(zarr_path), mode="r")["0"].shape  # (T, Z, Y, X)
                all_rows.append({
                    "dataset": name, "row_type": "node", "node_id": 0, "t": 0,
                    "z": shape[1] // 2, "y": shape[2] // 2, "x": shape[3] // 2,
                    "source_id": -1, "target_id": -1,
                })
                continue

            print("  tracking...")
            track_df, split_df, merge_df = run_tracking(det_df, scale_zyx)
            n_tracks = track_df["track_id"].nunique()
            print(f"  {len(track_df)} noeuds, {n_tracks} segments, {len(split_df)} division(s)")

            rows = build_rows_for_dataset(name, track_df, split_df)
            all_rows.extend(rows)
            print(f"  termine en {time.time() - t0:.1f}s\n")
        except Exception as exc:
            failed.append(name)
            print(f"  ECHEC sur {name}: {type(exc).__name__}: {exc}\n")

    if failed:
        print(f"ATTENTION: {len(failed)} dataset(s) ont echoue et sont absents de la submission: {failed}")
        print("-> chaque dataset de test DOIT apparaitre dans le CSV final (voir consignes de soumission).")

    submission = write_submission(all_rows, SUBMISSION_PATH)
    n_nodes = (submission.row_type == "node").sum()
    n_edges = (submission.row_type == "edge").sum()
    print(f"submission.csv final ecrit: {SUBMISSION_PATH}")
    print(f"  {n_nodes} nodes, {n_edges} edges, {submission.dataset.nunique()}/{len(dataset_names)} datasets")


if __name__ == "__main__":
    main()
