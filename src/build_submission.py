"""
ETAPE 3 : Construire le graphe de lignee final et l'exporter au format EXACT
demande par la competition (submission.csv).

Entree : data/<dataset>_track_df.csv (une ligne par detection, avec track_id)
         data/<dataset>_split_df.csv (evenements de division : parent_track_id -> child_track_id)

Logique :
  - node_id : on utilise simplement le numero de ligne du tableau de detections
    (unique par dataset, ce qui est la seule contrainte du format)
  - edges de continuite : au sein d'un meme track_id, on relie les detections
    consecutives triees par frame (t -> t suivant observe)
  - edges de division : pour chaque ligne de split_df, on relie la DERNIERE
    detection du track parent a la PREMIERE detection de chaque track enfant

Sortie : submission.csv (a la racine du projet), au format :
  id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_NAME = "synth1"
SUBMISSION_COLS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def build_submission_for_dataset(dataset_name):
    track_df = pd.read_csv(ROOT / "data" / f"{dataset_name}_track_df.csv").reset_index(drop=True)
    split_df = pd.read_csv(ROOT / "data" / f"{dataset_name}_split_df.csv")

    # node_id = position dans le tableau (unique par dataset)
    track_df["node_id"] = track_df.index

    node_rows = []
    for _, row in track_df.iterrows():
        node_rows.append({
            "dataset": dataset_name, "row_type": "node",
            "node_id": int(row.node_id), "t": int(row.frame),
            "z": int(round(row.z)), "y": int(round(row.y)), "x": int(round(row.x)),
            "source_id": -1, "target_id": -1,
        })

    edge_rows = []
    # edges de continuite : consecutives dans le temps, a l'interieur d'un meme track_id
    for track_id, group in track_df.sort_values("frame").groupby("track_id"):
        node_ids = group["node_id"].tolist()
        for src, tgt in zip(node_ids[:-1], node_ids[1:]):
            edge_rows.append({
                "dataset": dataset_name, "row_type": "edge",
                "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": int(src), "target_id": int(tgt),
            })

    # edges de division : derniere detection du parent -> premiere detection de chaque enfant
    for _, split in split_df.iterrows():
        parent_group = track_df[track_df.track_id == split.parent_track_id].sort_values("frame")
        child_group = track_df[track_df.track_id == split.child_track_id].sort_values("frame")
        if parent_group.empty or child_group.empty:
            continue
        source_id = int(parent_group.iloc[-1].node_id)
        target_id = int(child_group.iloc[0].node_id)
        edge_rows.append({
            "dataset": dataset_name, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": source_id, "target_id": target_id,
        })

    all_rows = node_rows + edge_rows
    df = pd.DataFrame(all_rows)
    df.insert(0, "id", range(len(df)))
    return df[SUBMISSION_COLS]


def main():
    submission = build_submission_for_dataset(DATASET_NAME)
    out_path = ROOT / "submission.csv"
    submission.to_csv(out_path, index=False)

    n_nodes = (submission.row_type == "node").sum()
    n_edges = (submission.row_type == "edge").sum()
    print(f"submission.csv ecrit: {out_path}")
    print(f"  {n_nodes} nodes, {n_edges} edges, dataset(s): {submission.dataset.unique().tolist()}")
    print("\nAperçu:")
    print(submission.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
