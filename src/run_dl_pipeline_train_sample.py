"""Pipeline complet (detection DL -> tracking -> submission -> evaluation)
sur l'echantillon d'entrainement annote, pour comparer au score baseline
(detection classique) obtenu precedemment avec real_evaluate.py."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from real_detect_dl import detect_dataset
from real_track import run_tracking
from real_build_submission import build_rows_for_dataset, write_submission
from real_io import load_geff_ground_truth
from evaluate_metric import edge_jaccard, division_jaccard_simplified
from config import ROOT, TRAIN_DIR

dataset_name = "44b6_0113de3b"
zarr_path = TRAIN_DIR / f"{dataset_name}.zarr"
geff_path = TRAIN_DIR / f"{dataset_name}.geff"

print("=== Detection (deep learning) ===")
det_df, scale_zyx = detect_dataset(zarr_path, verbose=False)
print(f"{len(det_df)} detections")

print("\n=== Tracking ===")
track_df, split_df, merge_df = run_tracking(det_df, scale_zyx)
n_tracks = track_df["track_id"].nunique()
print(f"{len(track_df)} noeuds, {n_tracks} segments, {len(split_df)} division(s)")

print("\n=== Construction submission ===")
rows = build_rows_for_dataset(dataset_name, track_df, split_df)
out_path = ROOT / "data" / f"{dataset_name}_dl_submission.csv"
submission = write_submission(rows, out_path)
print(f"Ecrit: {out_path}")

print("\n=== Evaluation locale ===")
pred_nodes = submission[submission.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
pred_edges = submission[submission.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)
gt_nodes, gt_edges, t_true_estimate = load_geff_ground_truth(geff_path)

edge_metrics, matched = edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true_estimate=t_true_estimate)
div_metrics = division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched)
score = edge_metrics["adj_edge_jaccard"] + 0.1 * div_metrics["division_jaccard"]

print("Edge metrics:")
for k, v in edge_metrics.items():
    print(f"  {k}: {v}")
print("Division metrics:")
for k, v in div_metrics.items():
    print(f"  {k}: {v}")
print(f"\nScore combine (deep learning) = {score:.4f}")
print("Score combine (baseline classique, pour comparaison) = 0.7251")
