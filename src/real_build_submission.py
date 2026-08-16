"""Construit un submission.csv a partir des resultats de tracking reels,
pour un ou plusieurs datasets a la fois (equivalent de build_submission.py)."""

import pandas as pd
from pathlib import Path

SUBMISSION_COLS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def build_rows_for_dataset(dataset_name, track_df, split_df):
    track_df = track_df.reset_index(drop=True)
    track_df["node_id"] = track_df.index

    node_rows = [
        {
            "dataset": dataset_name, "row_type": "node",
            "node_id": int(row.node_id), "t": int(row.frame),
            "z": int(round(row.z)), "y": int(round(row.y)), "x": int(round(row.x)),
            "source_id": -1, "target_id": -1,
        }
        for row in track_df.itertuples()
    ]

    edge_rows = []
    for track_id, group in track_df.sort_values("frame").groupby("track_id"):
        node_ids = group["node_id"].tolist()
        for src, tgt in zip(node_ids[:-1], node_ids[1:]):
            edge_rows.append({
                "dataset": dataset_name, "row_type": "edge",
                "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": int(src), "target_id": int(tgt),
            })

    for split in split_df.itertuples():
        parent_group = track_df[track_df.track_id == split.parent_track_id].sort_values("frame")
        child_group = track_df[track_df.track_id == split.child_track_id].sort_values("frame")
        if parent_group.empty or child_group.empty:
            continue
        edge_rows.append({
            "dataset": dataset_name, "row_type": "edge",
            "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": int(parent_group.iloc[-1].node_id),
            "target_id": int(child_group.iloc[0].node_id),
        })

    return node_rows + edge_rows


def write_submission(all_rows, out_path):
    df = pd.DataFrame(all_rows)
    df.insert(0, "id", range(len(df)))
    df = df[SUBMISSION_COLS]
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    dataset_name = "44b6_0113de3b"

    track_df = pd.read_csv(ROOT / "data" / f"{dataset_name}_real_track_df.csv")
    split_df = pd.read_csv(ROOT / "data" / f"{dataset_name}_real_split_df.csv")

    rows = build_rows_for_dataset(dataset_name, track_df, split_df)
    out_path = ROOT / "data" / f"{dataset_name}_real_submission.csv"
    submission = write_submission(rows, out_path)

    n_nodes = (submission.row_type == "node").sum()
    n_edges = (submission.row_type == "edge").sum()
    print(f"Ecrit: {out_path}  ({n_nodes} nodes, {n_edges} edges)")
