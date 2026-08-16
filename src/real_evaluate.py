"""Evaluation locale (edge Jaccard fidele + division Jaccard simplifie, voir
evaluate_metric.py) sur un VRAI dataset d'entrainement, verite terrain chargee
depuis le .geff officiel plutot que depuis un CSV synthetique."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from real_io import load_geff_ground_truth
from evaluate_metric import edge_jaccard, division_jaccard_simplified
from config import ROOT, TRAIN_DIR


def load_submission_nodes_edges(csv_path):
    df = pd.read_csv(csv_path)
    nodes = df[df.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
    edges = df[df.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)
    return nodes, edges


if __name__ == "__main__":
    dataset_name = "44b6_0113de3b"
    geff_path = TRAIN_DIR / f"{dataset_name}.geff"

    pred_nodes, pred_edges = load_submission_nodes_edges(ROOT / "data" / f"{dataset_name}_real_submission.csv")
    gt_nodes, gt_edges, t_true_estimate = load_geff_ground_truth(geff_path)

    print(f"Predit: {len(pred_nodes)} nodes, {len(pred_edges)} edges")
    print(f"Verite terrain (sparse): {len(gt_nodes)} nodes, {len(gt_edges)} edges")
    print(f"Estimation du nombre total de vraies cellules (T_true): {t_true_estimate}")

    edge_metrics, matched = edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true_estimate=t_true_estimate)
    div_metrics = division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched)
    score = edge_metrics["adj_edge_jaccard"] + 0.1 * div_metrics["division_jaccard"]

    print("\n=== Edge metrics ===")
    for k, v in edge_metrics.items():
        print(f"  {k}: {v}")
    print("\n=== Division metrics (simplifie) ===")
    for k, v in div_metrics.items():
        print(f"  {k}: {v}")
    print(f"\n=== Score combine (approx.) = {score:.4f} ===")
