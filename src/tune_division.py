"""
Balayage rapide du filtre de persistance des divisions (real_track.py),
SANS reentrainer le detecteur -- les detections sont calculees UNE SEULE
FOIS (l'inference est le cout dominant), puis chaque valeur de
min_daughter_persistence est testee a partir de ce cache.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from real_detect_dl import detect_dataset
from real_track import run_tracking
from real_build_submission import build_rows_for_dataset, write_submission
from real_io import load_geff_ground_truth
from evaluate_metric import edge_jaccard, division_jaccard_simplified
from config import TRAIN_DIR, DATA_DIR

DATASET_NAME = "6bba_ef7b4f7e"
PERSISTENCE_VALUES = [0, 2, 3, 4, 6, 8, 12]  # 0 = pas de filtre (comportement d'origine)


def main():
    print(f"Detection (heatmap) sur {DATASET_NAME}...")
    det_df, scale_zyx = detect_dataset(TRAIN_DIR / f"{DATASET_NAME}.zarr", verbose=False)
    print(f"{len(det_df)} detections\n")

    gt_nodes, gt_edges, t_true_estimate = load_geff_ground_truth(TRAIN_DIR / f"{DATASET_NAME}.geff")

    results = []
    for persistence in PERSISTENCE_VALUES:
        t0 = time.time()
        track_df, split_df, merge_df = run_tracking(det_df, scale_zyx, min_daughter_persistence=persistence)
        rows = build_rows_for_dataset(DATASET_NAME, track_df, split_df)
        submission = write_submission(rows, DATA_DIR / "_tmp_tune_div_submission.csv")

        pred_nodes = submission[submission.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
        pred_edges = submission[submission.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)

        edge_m, matched = edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true_estimate=t_true_estimate)
        div_m = division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched)
        score = edge_m["adj_edge_jaccard"] + 0.1 * div_m["division_jaccard"]

        results.append({
            "min_persistence": persistence, "n_forks_avant_filtre": len(split_df) if persistence == 0 else None,
            "edge_tp": edge_m["edge_tp"], "edge_fp": edge_m["edge_fp"], "edge_fn": edge_m["edge_fn"],
            "adj_edge_jaccard": edge_m["adj_edge_jaccard"],
            "div_tp": div_m["division_tp"], "div_fp": div_m["division_fp"], "div_fn": div_m["division_fn"],
            "division_jaccard": div_m["division_jaccard"], "score": score,
        })
        print(f"persistence={persistence:>2}  edge TP/FP/FN={edge_m['edge_tp']}/{edge_m['edge_fp']}/{edge_m['edge_fn']}  "
              f"div TP/FP/FN={div_m['division_tp']}/{div_m['division_fp']}/{div_m['division_fn']}  "
              f"adj_edge={edge_m['adj_edge_jaccard']:.4f}  div_jac={div_m['division_jaccard']:.4f}  "
              f"score={score:.4f}  ({time.time() - t0:.1f}s)")

    (DATA_DIR / "_tmp_tune_div_submission.csv").unlink(missing_ok=True)

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\n=== Classement (meilleur en premier) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
