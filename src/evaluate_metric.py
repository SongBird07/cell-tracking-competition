"""
ETAPE 4 : evaluation locale, INSPIREE de la vraie metrique officielle de la
competition (voir https://github.com/royerlab/kaggle-cell-tracking-competition
-> metrics.md et src/tracking_cellmot/metrics.py).

IMPORTANT -- ce script n'est PAS le code officiel :
  - le package officiel `tracking_cellmot` necessite Python >= 3.11, torch, et
    `tracksdata` (dependance git maintenue par royerlab) -- trop lourd pour de
    l'iteration locale rapide sur une machine de dev.
  - l'edge Jaccard (+ adjusted edge Jaccard) est reimplemente ici de facon
    FIDELE a la spec exacte de metrics.md (memes regles TP/FP/FN, meme formule
    d'ajustement).
  - le division Jaccard est une version SIMPLIFIEE de l'algorithme officiel
    (qui gere des cas limites tres fins : remontee sur les petits-enfants,
    ambiguite de composantes connexes GT, etc.). Ici on implemente le coeur de
    la logique (fenetre locale +/-1 frame autour de la division, 2 branches
    filles distinctes, topologie dirigee) -- suffisant pour comparer des
    variantes de pipeline entre elles, mais le chiffre exact ne sera pas
    identique au score Kaggle officiel.

Pour un score 100% fidele sur les vraies donnees, installer le package
officiel (Python 3.11+) :
    pip install "tracksdata @ git+https://github.com/royerlab/tracksdata@main"
    pip install "tracking-cellmot @ git+https://github.com/royerlab/kaggle-cell-tracking-competition@main"
et utiliser scripts/evaluate.py du depot.
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
VOXEL_SIZE_UM = np.array([1.625, 0.40625, 0.40625])  # z, y, x
MAX_DIST_UM = 7.0
ADJ_PENALTY_A = 0.1


def _physical_dist_matrix(pred_coords, gt_coords):
    p = pred_coords[:, None, :] * VOXEL_SIZE_UM
    g = gt_coords[None, :, :] * VOXEL_SIZE_UM
    return np.linalg.norm(p - g, axis=2)


def match_nodes_per_frame(pred_nodes, gt_nodes):
    """Bipartite matching des centroides par frame (<= 7um). Retourne un dict
    pred_node_id -> gt_node_id."""
    matched = {}
    frames = sorted(set(pred_nodes.t) | set(gt_nodes.t))
    for t in frames:
        p = pred_nodes[pred_nodes.t == t]
        g = gt_nodes[gt_nodes.t == t]
        if len(p) == 0 or len(g) == 0:
            continue
        p_coords = p[["z", "y", "x"]].to_numpy(dtype=float)
        g_coords = g[["z", "y", "x"]].to_numpy(dtype=float)
        dist = _physical_dist_matrix(p_coords, g_coords)
        cost = np.where(dist <= MAX_DIST_UM, dist, 1e6)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if dist[r, c] <= MAX_DIST_UM:
                matched[int(p.iloc[r].node_id)] = int(g.iloc[c].node_id)
    return matched


def edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges, t_true_estimate=None):
    matched = match_nodes_per_frame(pred_nodes, gt_nodes)

    gt_edge_set = set(zip(gt_edges.source_id, gt_edges.target_id))
    gt_outgoing = {}
    gt_incoming = {}
    for s, t in gt_edge_set:
        gt_outgoing.setdefault(s, set()).add(t)
        gt_incoming.setdefault(t, set()).add(s)

    tp = fp = 0
    covered_gt_edges = set()

    for _, e in pred_edges.iterrows():
        ps, pt = int(e.source_id), int(e.target_id)
        gs = matched.get(ps)
        gt_ = matched.get(pt)

        if gs is not None and gt_ is not None and (gs, gt_) in gt_edge_set:
            tp += 1
            covered_gt_edges.add((gs, gt_))
            continue

        is_fp = False
        if gt_ is not None and gt_ in gt_incoming and gs not in gt_incoming[gt_]:
            is_fp = True
        if gs is not None and gs in gt_outgoing and gt_ not in gt_outgoing[gs]:
            is_fp = True
        if is_fp:
            fp += 1
        # sinon : ignore (pas de preuve GT dans un sens ou l'autre)

    fn = len(gt_edge_set - covered_gt_edges)

    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    t_pred = len(pred_nodes)
    t_true = t_true_estimate if t_true_estimate is not None else len(gt_nodes)
    if t_true > 0:
        adj = max(0.0, jaccard * (1 - ADJ_PENALTY_A * (t_pred - t_true) / t_true))
    else:
        adj = jaccard

    return {
        "edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
        "edge_jaccard": jaccard, "adj_edge_jaccard": adj,
        "n_pred_nodes": t_pred, "n_gt_nodes_estimate": t_true,
    }, matched


def _build_nx(nodes_df, edges_df):
    g = nx.DiGraph()
    for _, n in nodes_df.iterrows():
        g.add_node(int(n.node_id), t=int(n.t))
    for _, e in edges_df.iterrows():
        g.add_edge(int(e.source_id), int(e.target_id))
    return g


def division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched_pred_to_gt):
    """Version simplifiee (voir avertissement en tete de fichier)."""
    gt_g = _build_nx(gt_nodes, gt_edges)
    pred_g = _build_nx(pred_nodes, pred_edges)
    matched_gt_to_pred = {}
    for p, g in matched_pred_to_gt.items():
        matched_gt_to_pred.setdefault(g, []).append(p)

    gt_divisions = [n for n in gt_g.nodes if gt_g.out_degree(n) >= 2]
    pred_forks = [n for n in pred_g.nodes if pred_g.out_degree(n) >= 2]

    # matrice candidate[gt_div][pred_fork] = True si les criteres locaux sont remplis
    candidates = {}  # (gt_div_idx, fork_idx) -> True
    for gi, gt_div in enumerate(gt_divisions):
        gt_children = list(gt_g.successors(gt_div))
        # ancre locale autorisee : le noeud GT lui-meme ou son predecesseur immediat
        gt_predecessors = list(gt_g.predecessors(gt_div))
        anchor_gt_nodes = {gt_div} | set(gt_predecessors)

        for fi, fork in enumerate(pred_forks):
            anchor_pred_candidates = {fork} | set(pred_g.predecessors(fork))
            if not any(matched_pred_to_gt.get(p) in anchor_gt_nodes for p in anchor_pred_candidates):
                continue

            pred_children = list(pred_g.successors(fork))
            matched_child_targets = {matched_pred_to_gt.get(pc) for pc in pred_children}
            n_daughters_recovered = sum(1 for gc in gt_children if gc in matched_child_targets)
            if n_daughters_recovered >= 2:
                candidates[(gi, fi)] = True

    # appariement bipartite max-cardinalite (1 fork <-> 1 division max)
    B = nx.Graph()
    B.add_nodes_from([f"gt_{gi}" for gi in range(len(gt_divisions))], bipartite=0)
    B.add_nodes_from([f"pred_{fi}" for fi in range(len(pred_forks))], bipartite=1)
    for (gi, fi) in candidates:
        B.add_edge(f"gt_{gi}", f"pred_{fi}")
    pairing = nx.algorithms.bipartite.maximum_matching(
        B, top_nodes=[f"gt_{gi}" for gi in range(len(gt_divisions))]
    ) if len(gt_divisions) and len(pred_forks) else {}

    tp = sum(1 for k in pairing if k.startswith("gt_"))
    fn = len(gt_divisions) - tp

    matched_pred_fork_indices = {int(v.split("_")[1]) for k, v in pairing.items() if k.startswith("gt_")}
    fp = 0
    for fi, fork in enumerate(pred_forks):
        if fi in matched_pred_fork_indices:
            continue
        # FP seulement si le fork est "ancre" dans du vrai signal GT (evite de
        # penaliser des faux forks totalement hors-sol, comme pour les edges)
        if matched_pred_to_gt.get(fork) is not None:
            fp += 1

    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "division_tp": tp, "division_fp": fp, "division_fn": fn,
        "division_jaccard": jaccard,
        "n_gt_divisions": len(gt_divisions), "n_pred_forks": len(pred_forks),
    }


def load_submission_like(path, dataset):
    df = pd.read_csv(path)
    df = df[df.dataset == dataset]
    nodes = df[df.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
    edges = df[df.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)
    return nodes, edges


def main():
    dataset = "synth1"
    pred_nodes, pred_edges = load_submission_like(ROOT / "submission.csv", dataset)
    gt_nodes, gt_edges = load_submission_like(ROOT / "data" / f"{dataset}_ground_truth.csv", dataset)

    edge_metrics, matched = edge_jaccard(pred_nodes, pred_edges, gt_nodes, gt_edges)
    div_metrics = division_jaccard_simplified(pred_nodes, pred_edges, gt_nodes, gt_edges, matched)

    score = edge_metrics["adj_edge_jaccard"] + 0.1 * div_metrics["division_jaccard"]

    print("=== Edge metrics ===")
    for k, v in edge_metrics.items():
        print(f"  {k}: {v}")
    print("\n=== Division metrics (version simplifiee, voir avertissement en tete de fichier) ===")
    for k, v in div_metrics.items():
        print(f"  {k}: {v}")
    print(f"\n=== Score combine (approx.) = adj_edge_jaccard + 0.1 * division_jaccard = {score:.4f} ===")


if __name__ == "__main__":
    main()
