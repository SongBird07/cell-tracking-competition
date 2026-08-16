"""
Balayage rapide des parametres de watershed (prob_threshold, min_distance,
min_size) SANS reentrainer : les cartes de probabilite sont calculees UNE
SEULE FOIS (l'inference GPU est le cout dominant), puis chaque combinaison
de parametres est testee a partir de ce cache -- watershed + tracking +
evaluation sont rapides sur CPU, donc on peut tester une grille entiere en
quelques minutes.
"""

import sys, time, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from real_io import open_zarr_volume, load_geff_ground_truth
from real_detect_seg import load_model, segment_frame, _pad_to_multiple, _run_model
from real_track import run_tracking
from real_build_submission import build_rows_for_dataset, write_submission
from evaluate_metric import edge_jaccard, division_jaccard_simplified
from config import TRAIN_DIR, DATA_DIR

DATASET_NAME = "6bba_ef7b4f7e"  # dataset de validation bien annote (jamais entraine dessus)

PROB_THRESHOLDS = [0.5, 0.7]
MIN_DISTANCES = [8, 10]
MIN_SIZES = [100, 200, 400]


def compute_prob_maps(dataset_name, device):
    model = load_model(device)
    arr, scale_zyx, quantiles = open_zarr_volume(TRAIN_DIR / f"{dataset_name}.zarr")
    q_lo, q_hi = quantiles.get("0.1", 0.0), quantiles.get("0.999", 2000.0)
    T = arr.shape[0]

    prob_maps = []
    t0 = time.time()
    for t in range(T):
        frame = arr[t][:].astype(np.float32)
        norm = np.clip((frame - q_lo) / max(q_hi - q_lo, 1e-6), 0.0, 1.0)
        padded, pads = _pad_to_multiple(norm)
        x = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)
        prob = _run_model(model, x, device)
        sl = tuple(slice(0, dim) for dim in norm.shape)
        prob_maps.append(prob[sl])
    print(f"{T} cartes de probabilite calculees en {time.time() - t0:.1f}s")
    return prob_maps, scale_zyx


def detections_from_cache(prob_maps, prob_threshold, min_distance, min_size):
    rows = []
    for t, prob_map in enumerate(prob_maps):
        labels, n_instances = segment_frame(prob_map, prob_threshold, min_distance, min_size)
        if n_instances:
            from scipy import ndimage as ndi
            centroids = ndi.center_of_mass(labels > 0, labels, range(1, n_instances + 1))
            for (z, y, x_) in centroids:
                rows.append({"frame": t, "z": int(round(z)), "y": int(round(y)), "x": int(round(x_))})
    return pd.DataFrame(rows, columns=["frame", "z", "y", "x"])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calcul des cartes de probabilite pour {DATASET_NAME}...")
    prob_maps, scale_zyx = compute_prob_maps(DATASET_NAME, device)

    gt_nodes, gt_edges, t_true_estimate = load_geff_ground_truth(TRAIN_DIR / f"{DATASET_NAME}.geff")

    results = []
    combos = list(itertools.product(PROB_THRESHOLDS, MIN_DISTANCES, MIN_SIZES))
    print(f"\n{len(combos)} combinaisons a tester...\n")

    for prob_threshold, min_distance, min_size in combos:
        t0 = time.time()
        det_df = detections_from_cache(prob_maps, prob_threshold, min_distance, min_size)
        if det_df.empty:
            print(f"thr={prob_threshold} min_dist={min_distance} min_size={min_size} -> 0 detections, ignore")
            continue

        track_df, split_df, merge_df = run_tracking(det_df, scale_zyx)
        rows = build_rows_for_dataset(DATASET_NAME, track_df, split_df)
        submission = write_submission(rows, DATA_DIR / "_tmp_tune_submission.csv")

        pred_nodes = submission[submission.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
        pred_edges = submission[submission.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)

        edge_m, matched = edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true_estimate=t_true_estimate)
        div_m = division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched)
        score = edge_m["adj_edge_jaccard"] + 0.1 * div_m["division_jaccard"]

        results.append({
            "prob_threshold": prob_threshold, "min_distance": min_distance, "min_size": min_size,
            "n_detections": len(det_df), "edge_tp": edge_m["edge_tp"], "edge_fp": edge_m["edge_fp"],
            "edge_fn": edge_m["edge_fn"], "adj_edge_jaccard": edge_m["adj_edge_jaccard"],
            "division_jaccard": div_m["division_jaccard"], "score": score,
        })
        print(f"thr={prob_threshold} min_dist={min_distance} min_size={min_size} -> "
              f"{len(det_df)} det, edge TP/FP/FN={edge_m['edge_tp']}/{edge_m['edge_fp']}/{edge_m['edge_fn']}, "
              f"score={score:.4f}  ({time.time() - t0:.1f}s)")

    (DATA_DIR / "_tmp_tune_submission.csv").unlink(missing_ok=True)

    results_df = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\n=== Classement (meilleur en premier) ===")
    print(results_df.to_string(index=False))

    out_path = DATA_DIR / "watershed_tuning_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nEcrit: {out_path}")


if __name__ == "__main__":
    main()
