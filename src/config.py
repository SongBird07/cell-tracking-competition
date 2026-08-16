"""
Configuration centrale des chemins du projet.

Toutes les autres scripts importent DATA_ROOT/TRAIN_DIR/TEST_DIR/WEIGHTS_PATH
d'ici plutot que de coder le chemin en dur -- ca evite d'avoir a modifier
20 fichiers si les donnees changent d'emplacement (autre disque, machine
Kaggle, etc.), et ca documente en un seul endroit ce dont le projet depend.

Priorite de resolution (comme le fait le depot officiel de la competition
avec CELLMOT_DATA_DIR) :
  1. variable d'environnement CELLTRACK_DATA_DIR (si definie)
  2. chemin de montage Kaggle standard (present dans un notebook Kaggle)
  3. chemin local par defaut de developpement (D:/biohub-...)
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT = PROJECT_ROOT  # alias court, utilise par convention dans les autres scripts

_KAGGLE_ROOT = Path("/kaggle/input/biohub-cell-tracking-during-development")
_LOCAL_DEFAULT_ROOT = Path("D:/biohub-cell-tracking-during-development")


def _resolve_data_root() -> Path:
    env = os.environ.get("CELLTRACK_DATA_DIR")
    if env:
        return Path(env)
    if _KAGGLE_ROOT.exists():
        return _KAGGLE_ROOT
    return _LOCAL_DEFAULT_ROOT


DATA_ROOT = _resolve_data_root()
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"
SAMPLE_SUBMISSION_PATH = DATA_ROOT / "sample_submission.csv"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "heatmap_unet.pt"
WEIGHTS_LATEST_PATH = WEIGHTS_DIR / "heatmap_unet_latest.pt"

DATA_DIR = PROJECT_ROOT / "data"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def check_data_root(require_train=False, require_test=False):
    """Verifie que les dossiers attendus existent, avec un message clair
    plutot qu'une FileNotFoundError obscure au milieu du pipeline."""
    problems = []
    if not DATA_ROOT.exists():
        problems.append(
            f"DATA_ROOT introuvable: {DATA_ROOT}\n"
            f"  -> definis la variable d'environnement CELLTRACK_DATA_DIR vers le "
            f"dossier de la competition (celui contenant train/, test/, sample_submission.csv)"
        )
    else:
        if require_train and not TRAIN_DIR.exists():
            problems.append(f"TRAIN_DIR introuvable: {TRAIN_DIR}")
        if require_test and not TEST_DIR.exists():
            problems.append(f"TEST_DIR introuvable: {TEST_DIR}")
    if problems:
        raise FileNotFoundError("\n".join(problems))


if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT:    {DATA_ROOT}  (existe: {DATA_ROOT.exists()})")
    print(f"TRAIN_DIR:    {TRAIN_DIR}  (existe: {TRAIN_DIR.exists()})")
    print(f"TEST_DIR:     {TEST_DIR}  (existe: {TEST_DIR.exists()})")
    print(f"WEIGHTS_PATH: {WEIGHTS_PATH}  (existe: {WEIGHTS_PATH.exists()})")
