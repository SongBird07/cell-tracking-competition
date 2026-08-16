"""
Utilitaires d'entree/sortie pour les VRAIES donnees de la competition
(format OME-Zarr v3 pour les images, format GEFF pour la verite terrain).

A la difference des donnees synthetiques (generate_synthetic_data.py), on ne
charge JAMAIS tout le volume (T, Z, Y, X) en memoire d'un coup : on garde
l'objet zarr "paresseux" et on ne lit que la frame demandee (arr[t]), ce qui
est indispensable vu la taille des vrais volumes (des dizaines de frames de
64x256x256 uint16 chacune, x plusieurs centaines de datasets).
"""

import zarr
import numpy as np
import pandas as pd
import geff
from pathlib import Path

DEFAULT_SCALE_ZYX = np.array([1.625, 0.40625, 0.40625])  # fallback si absent des metadonnees


def open_zarr_volume(zarr_path):
    """Ouvre un volume OME-Zarr (T, Z, Y, X) sans charger les donnees.

    Retourne (arr, scale_zyx, quantiles) :
      - arr : zarr.Array, indexable comme un numpy array (arr[t] charge une frame)
      - scale_zyx : echelle physique (um/voxel) lue dans les metadonnees OME-Zarr
      - quantiles : dict de quantiles d'intensite precalcules par les organisateurs
                    (tres utile pour choisir un seuil de detection adapte a CE
                    dataset precis, plutot qu'une constante globale)
    """
    group = zarr.open_group(str(zarr_path), mode="r")
    arr = group["0"]

    try:
        transform = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        scale_tzyx = transform["scale"]
        scale_zyx = np.array(scale_tzyx[1:4])
    except (KeyError, IndexError):
        scale_zyx = DEFAULT_SCALE_ZYX

    quantiles = group.attrs.get("image_statistics", {}).get("quantiles", {})
    return arr, scale_zyx, quantiles


def load_geff_ground_truth(geff_path):
    """Charge un fichier .geff (verite terrain) en (nodes_df, edges_df, t_true_estimate).

    nodes_df : colonnes [node_id, t, z, y, x]
    edges_df : colonnes [source_id, target_id]
    t_true_estimate : nombre total de vraies cellules estime par les organisateurs
                       (meme si la verite terrain n'en annote qu'une fraction) --
                       utilise dans la formule d'adjusted edge Jaccard.
    """
    graph, meta = geff.read(str(geff_path))

    node_rows = [
        {"node_id": nid, "t": int(d["t"]), "z": int(d["z"]), "y": int(d["y"]), "x": int(d["x"])}
        for nid, d in graph.nodes(data=True)
    ]
    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x"])

    edges_df = pd.DataFrame(list(graph.edges()), columns=["source_id", "target_id"])

    extra = meta.extra or {}
    t_true_estimate = extra.get("estimated_number_of_nodes", np.nan)
    return nodes_df, edges_df, t_true_estimate


def list_datasets(split_dir):
    """Liste les noms de datasets (sans extension) presents dans train/ ou test/."""
    split_dir = Path(split_dir)
    return sorted({p.stem for p in split_dir.glob("*.zarr")})
