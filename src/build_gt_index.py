"""
Scanne TOUS les .geff du train set et construit un index global des points
annotes (dataset, node_id, t, z, y, x) + les edges (pour distinguer plus tard
les divisions). C'est la base de l'echantillonnage de patchs d'entrainement :
plutot que de parcourir des volumes entiers (essentiellement vides d'annotations
vu le sparsite des GT), on tire directement des points annotes au hasard.

Sortie : data/gt_index_nodes.csv, data/gt_index_edges.csv
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from real_io import load_geff_ground_truth, list_datasets
from config import ROOT, TRAIN_DIR, check_data_root


def main():
    check_data_root(require_train=True)
    dataset_names = list_datasets(TRAIN_DIR)
    print(f"{len(dataset_names)} datasets d'entrainement trouves")

    all_nodes = []
    all_edges = []
    t0 = time.time()
    for i, name in enumerate(dataset_names):
        geff_path = TRAIN_DIR / f"{name}.geff"
        try:
            nodes_df, edges_df, t_true = load_geff_ground_truth(geff_path)
        except Exception as exc:
            print(f"  SKIP {name}: {type(exc).__name__}: {exc}")
            continue

        nodes_df = nodes_df.copy()
        nodes_df["dataset"] = name
        all_nodes.append(nodes_df)

        edges_df = edges_df.copy()
        edges_df["dataset"] = name
        all_edges.append(edges_df)

        if (i + 1) % 25 == 0 or i == len(dataset_names) - 1:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(dataset_names)} datasets scannes ({elapsed:.0f}s)")

    nodes_all = pd.concat(all_nodes, ignore_index=True)
    edges_all = pd.concat(all_edges, ignore_index=True)

    nodes_out = ROOT / "data" / "gt_index_nodes.csv"
    edges_out = ROOT / "data" / "gt_index_edges.csv"
    nodes_all.to_csv(nodes_out, index=False)
    edges_all.to_csv(edges_out, index=False)

    print(f"\n{len(nodes_all)} noeuds annotes au total sur {nodes_all.dataset.nunique()} datasets")
    print(f"{len(edges_all)} edges annotes au total")
    print(f"Ecrit: {nodes_out}")
    print(f"Ecrit: {edges_out}")


if __name__ == "__main__":
    main()
