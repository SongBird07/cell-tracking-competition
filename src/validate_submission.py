"""
Valide un submission.csv AVANT de le soumettre a Kaggle -- une soumission mal
formee genere une erreur de scoring sur la plateforme (voir le "Code
Debugging" doc de Kaggle cite dans l'enonce), et ces essais sont limites : il
vaut mieux attraper les problemes ici, localement, gratuitement.

Verifie :
  - colonnes exactes et dans l'ordre attendu
  - id = compteur sequentiel 0..N-1
  - row_type in {node, edge}
  - lignes node : node_id/t/z/y/x renseignes (!= -1), source_id=target_id=-1
  - lignes edge : node_id/t/z/y/x = -1, source_id et target_id renseignes
  - node_id unique PAR DATASET (l'enonce dit juste "unique", pas forcement
    globalement unique -- mais l'unicite par dataset est la contrainte reelle
    puisque les edges d'un dataset ne referencent que ses propres node_id)
  - chaque edge reference des node_id existants DANS LE MEME DATASET
  - pas d'auto-boucle (source_id == target_id)
  - chaque edge va bien vers l'avant dans le temps (t_target > t_source)
  - tous les datasets requis (dossier test/ ou sample_submission.csv) sont
    presents -- et aucun dataset "en trop"
  - (optionnel) coordonnees z/y/x/t dans les bornes du volume zarr correspondant

Usage:
    python src/validate_submission.py submission.csv
    python src/validate_submission.py submission.csv --check-bounds
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import TEST_DIR, SAMPLE_SUBMISSION_PATH, SUBMISSION_PATH

EXPECTED_COLUMNS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _required_datasets():
    if SAMPLE_SUBMISSION_PATH.exists():
        return set(pd.read_csv(SAMPLE_SUBMISSION_PATH, usecols=["dataset"]).dataset.unique())
    if TEST_DIR.exists():
        return {p.stem for p in TEST_DIR.glob("*.zarr")}
    return None  # pas de reference disponible, on saute cette verification


def validate(csv_path, check_bounds=False):
    errors = []
    warnings = []

    df = pd.read_csv(csv_path)

    if list(df.columns) != EXPECTED_COLUMNS:
        errors.append(f"Colonnes incorrectes: {list(df.columns)} (attendu: {EXPECTED_COLUMNS})")
        return errors, warnings  # le reste des checks suppose les bonnes colonnes

    if not (df["id"] == range(len(df))).all():
        errors.append("La colonne 'id' n'est pas une sequence 0..N-1 sans trou")

    bad_row_type = set(df.row_type.unique()) - {"node", "edge"}
    if bad_row_type:
        errors.append(f"row_type invalide(s): {bad_row_type}")

    nodes = df[df.row_type == "node"]
    edges = df[df.row_type == "edge"]

    bad_nodes = nodes[(nodes.node_id == -1) | (nodes.t == -1) | (nodes.z == -1)
                       | (nodes.y == -1) | (nodes.x == -1)
                       | (nodes.source_id != -1) | (nodes.target_id != -1)]
    if len(bad_nodes):
        errors.append(f"{len(bad_nodes)} ligne(s) node mal formees (node_id/t/z/y/x=-1 ou source/target != -1)")

    bad_edges = edges[(edges.node_id != -1) | (edges.t != -1) | (edges.z != -1)
                       | (edges.y != -1) | (edges.x != -1)
                       | (edges.source_id == -1) | (edges.target_id == -1)]
    if len(bad_edges):
        errors.append(f"{len(bad_edges)} ligne(s) edge mal formees (node/t/z/y/x != -1 ou source/target=-1)")

    self_loops = edges[edges.source_id == edges.target_id]
    if len(self_loops):
        errors.append(f"{len(self_loops)} edge(s) en auto-boucle (source_id == target_id)")

    for dataset, dnodes in nodes.groupby("dataset"):
        dupes = dnodes.node_id[dnodes.node_id.duplicated()]
        if len(dupes):
            errors.append(f"[{dataset}] {len(dupes)} node_id duplique(s)")

    node_lookup = {}  # (dataset, node_id) -> t
    for row in nodes.itertuples():
        node_lookup[(row.dataset, row.node_id)] = row.t

    for row in edges.itertuples():
        src_key = (row.dataset, row.source_id)
        tgt_key = (row.dataset, row.target_id)
        if src_key not in node_lookup:
            errors.append(f"[{row.dataset}] edge id={row.id}: source_id {row.source_id} n'existe pas comme node")
            continue
        if tgt_key not in node_lookup:
            errors.append(f"[{row.dataset}] edge id={row.id}: target_id {row.target_id} n'existe pas comme node")
            continue
        if node_lookup[tgt_key] <= node_lookup[src_key]:
            errors.append(
                f"[{row.dataset}] edge id={row.id}: ne va pas strictement vers l'avant dans le temps "
                f"(t_source={node_lookup[src_key]}, t_target={node_lookup[tgt_key]})"
            )

    required = _required_datasets()
    present = set(df.dataset.unique())
    if required is not None:
        missing = required - present
        extra = present - required
        if missing:
            errors.append(f"Dataset(s) requis absent(s) de la soumission: {sorted(missing)}")
        if extra:
            warnings.append(f"Dataset(s) present(s) mais non requis (ignores par le scoring): {sorted(extra)}")
    else:
        warnings.append("Impossible de verifier la couverture des datasets (ni sample_submission.csv ni test/ trouves)")

    for dataset, dnodes in nodes.groupby("dataset"):
        if len(dnodes) == 0:
            warnings.append(f"[{dataset}] 0 node -- dataset present mais vide")

    if check_bounds and TEST_DIR.exists():
        import zarr
        for dataset in present:
            zarr_path = TEST_DIR / f"{dataset}.zarr"
            if not zarr_path.exists():
                continue
            shape = zarr.open_group(str(zarr_path), mode="r")["0"].shape  # (T, Z, Y, X)
            dnodes = nodes[nodes.dataset == dataset]
            oob = dnodes[(dnodes.t < 0) | (dnodes.t >= shape[0])
                         | (dnodes.z < 0) | (dnodes.z >= shape[1])
                         | (dnodes.y < 0) | (dnodes.y >= shape[2])
                         | (dnodes.x < 0) | (dnodes.x >= shape[3])]
            if len(oob):
                errors.append(f"[{dataset}] {len(oob)} node(s) hors des bornes du volume {shape}")

    return errors, warnings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default=str(SUBMISSION_PATH))
    parser.add_argument("--check-bounds", action="store_true", help="verifie aussi z/y/x/t contre la forme des zarr de test")
    args = parser.parse_args()

    print(f"Validation de {args.csv_path}...\n")
    errors, warnings = validate(Path(args.csv_path), check_bounds=args.check_bounds)

    if warnings:
        print(f"{len(warnings)} avertissement(s):")
        for w in warnings:
            print(f"  ! {w}")
        print()

    if errors:
        print(f"{len(errors)} ERREUR(S) -- cette soumission SERA REJETEE ou mal scoree par Kaggle:")
        for e in errors:
            print(f"  X {e}")
        sys.exit(1)
    else:
        print("OK -- aucune erreur detectee, la soumission est bien formee.")
        sys.exit(0)
