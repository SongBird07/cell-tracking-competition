"""
Evaluation sur PLUSIEURS echantillons annotes (pas un seul !) pour avoir une
mesure fiable du detecteur -- un score calcule sur un unique petit
echantillon (comme 44b6_0113de3b, 52 nodes/50 edges) est tres bruite : une
poignee de correspondances qui basculent pres du seuil de 7um suffit a faire
varier le score de +/-0.3, sans que ca reflete un vrai changement de qualite.

Choisit les datasets de VALIDATION (jamais vus a l'entrainement, voir
split_datasets() dans train_detector.py) les mieux annotes, agrege
TP/FP/FN sur tous les echantillons (comme le fait la vraie metrique
officielle : edge Jaccard pondere par TP+FP+FN, division Jaccard
micro-moyenne), et rapporte un score global bien plus robuste qu'un score
par-echantillon.
"""

import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_track import run_tracking
from real_build_submission import build_rows_for_dataset, write_submission
from real_io import load_geff_ground_truth
from evaluate_metric import edge_jaccard, division_jaccard_simplified
from train_detector import split_datasets, ensure_gt_index
from config import TRAIN_DIR, DATA_DIR


def evaluate_dataset(dataset_name, detect_dataset):
    zarr_path = TRAIN_DIR / f"{dataset_name}.zarr"
    geff_path = TRAIN_DIR / f"{dataset_name}.geff"

    det_df, scale_zyx = detect_dataset(zarr_path, verbose=False)
    track_df, split_df, merge_df = run_tracking(det_df, scale_zyx)
    rows = build_rows_for_dataset(dataset_name, track_df, split_df)
    submission = write_submission(rows, DATA_DIR / f"_tmp_{dataset_name}_submission.csv")

    pred_nodes = submission[submission.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
    pred_edges = submission[submission.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)
    gt_nodes, gt_edges, t_true_estimate = load_geff_ground_truth(geff_path)

    edge_metrics, matched = edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true_estimate=t_true_estimate)
    div_metrics = division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched)
    return edge_metrics, div_metrics


def main(n_samples=8, detector="heatmap"):
    if detector == "heatmap":
        from real_detect_dl import detect_dataset
    elif detector == "seg":
        from real_detect_seg import detect_dataset
    else:
        raise ValueError(f"detecteur inconnu: {detector} (attendu: 'heatmap' ou 'seg')")

    nodes_df = ensure_gt_index()
    train_names, val_names = split_datasets(nodes_df)

    counts = nodes_df.groupby("dataset").size()
    val_counts = counts[counts.index.isin(val_names)].sort_values(ascending=False)
    chosen = list(val_counts.head(n_samples).index)
    print(f"Evaluation ({detector}) sur {len(chosen)} datasets de VALIDATION (jamais entraines dessus):")
    print(f"  {chosen}\n")

    total_tp = total_fp = total_fn = 0
    total_div_tp = total_div_fp = total_div_fn = 0
    per_sample_adj = []  # (adj_jaccard, weight)

    for name in chosen:
        t0 = time.time()
        edge_m, div_m = evaluate_dataset(name, detect_dataset)
        total_tp += edge_m["edge_tp"]
        total_fp += edge_m["edge_fp"]
        total_fn += edge_m["edge_fn"]
        total_div_tp += div_m["division_tp"]
        total_div_fp += div_m["division_fp"]
        total_div_fn += div_m["division_fn"]

        weight = edge_m["edge_tp"] + edge_m["edge_fp"] + edge_m["edge_fn"]
        per_sample_adj.append((edge_m["adj_edge_jaccard"], weight))

        print(f"[{name}] edge TP/FP/FN={edge_m['edge_tp']}/{edge_m['edge_fp']}/{edge_m['edge_fn']}  "
              f"adj_jaccard={edge_m['adj_edge_jaccard']:.4f}  "
              f"div TP/FP/FN={div_m['division_tp']}/{div_m['division_fp']}/{div_m['division_fn']}  "
              f"({time.time() - t0:.1f}s)")

    global_edge_jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0
    weighted_adj = sum(a * w for a, w in per_sample_adj) / sum(w for _, w in per_sample_adj) if per_sample_adj else 0.0
    div_jaccard = total_div_tp / (total_div_tp + total_div_fp + total_div_fn) if (total_div_tp + total_div_fp + total_div_fn) else 0.0
    score = weighted_adj + 0.1 * div_jaccard

    print(f"\n=== Agrege sur {len(chosen)} echantillons ===")
    print(f"Edge: TP={total_tp} FP={total_fp} FN={total_fn}  edge_jaccard(brut)={global_edge_jaccard:.4f}")
    print(f"adj_edge_jaccard pondere (TP+FP+FN) = {weighted_adj:.4f}")
    print(f"Division: TP={total_div_tp} FP={total_div_fp} FN={total_div_fn}  division_jaccard={div_jaccard:.4f}")
    print(f"Score combine = {score:.4f}")

    # nettoyage des fichiers temporaires
    for name in chosen:
        (DATA_DIR / f"_tmp_{name}_submission.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["heatmap", "seg"], default="heatmap")
    parser.add_argument("--n-samples", type=int, default=8)
    args = parser.parse_args()
    main(n_samples=args.n_samples, detector=args.detector)
