# Installation

## 1. Python

Python **3.11 ou plus recent** est requis (le format Zarr v3 utilise par les
donnees de la competition n'est lisible qu'a partir de `zarr>=3`, qui exige
Python 3.11+).

Verifie ta version : `python --version`

## 2. Environnement virtuel

```bash
python -m venv .venv
```

Windows (Git Bash) :
```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
```

macOS/Linux :
```bash
.venv/bin/python -m pip install --upgrade pip
```

## 3. Dependances

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 4. PyTorch (a part -- depend de ta machine)

**Si tu as un GPU NVIDIA** : va sur https://pytorch.org/get-started/locally/,
choisis ton OS + la version CUDA de ton driver (`nvidia-smi` l'affiche en
haut a droite), et utilise la commande donnee. Exemple (CUDA 12.8) :

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Si tu n'as pas de GPU NVIDIA** (CPU seulement -- l'entrainement sera
beaucoup plus lent) :

```bash
.venv/Scripts/python.exe -m pip install torch torchvision
```

## 5. Verification

```bash
.venv/Scripts/python.exe -c "import torch, zarr, geff, laptrack; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available()); print('zarr', zarr.__version__)"
```

## 6. Lancer l'entrainement

Adapte le chemin `.venv/Scripts/python.exe` si tu as nomme ton environnement
autrement (ce depot a ete developpe avec un dossier `.venv314`, mais
n'importe quel nom fonctionne du moment que la commande pointe dessus) :

```bash
.venv/Scripts/python.exe src/train_detector.py --patience 10 --num-workers 6 --batch-size 20
```

Voir `TUTORIAL.md` pour le detail du pipeline complet.
