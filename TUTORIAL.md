# Tutoriel — Biohub Cell Tracking During Development (Kaggle)

Ce tutoriel explique comment construire, étape par étape, une solution pour la compétition
**"Biohub - Cell Tracking During Development"** : détecter des cellules dans des volumes 3D
au cours du temps, les relier entre elles (tracking), détecter les divisions cellulaires
(mitose), et reconstruire les lignées cellulaires.

Il est **conceptuel** : chaque section explique le principe et fournit des extraits de code
illustratifs (à adapter une fois que vous aurez les données réelles). L'idée est de vous
donner une feuille de route complète, du chargement des données jusqu'à la génération du
`submission.csv`.

---

## 1. Bien comprendre ce qui est demandé

La tâche se décompose en 3 sous-problèmes qui s'enchaînent :

1. **Détection** : pour chaque timepoint `t` d'un volume 3D, trouver le centre (centroïde
   `z, y, x`) de chaque cellule. Ce sont les **nodes** du graphe de sortie.
2. **Tracking (association temporelle)** : relier une cellule au temps `t` à la même cellule
   au temps `t+1`. Ce sont les **edges** du graphe.
3. **Division cellulaire (mitose)** : détecter les cas où une cellule mère donne naissance à
   deux cellules filles — un node avec **2 edges sortants ou plus**.

Le tout doit être exporté sous forme d'un graphe de lignée (lineage graph), au format CSV
décrit dans l'énoncé (voir section 6).

**La métrique** combine :
- Un **Jaccard sur les edges** (qualité du tracking frame-à-frame), avec un appariement
  optimal des centroïdes prédits/vérité-terrain (distance max 7.0 µm, avec l'échelle physique
  `z=1.625, y=x=0.40625 µm/voxel` — donc **le z compte 4x plus** en distance physique que
  y/x, c'est important pour le matching).
- Un **Jaccard sur les divisions**, qui vérifie que la composante connexe autour d'une
  division touche bien la cellule mère et les deux lignées filles.

👉 Conséquence pratique : un bon tracking global est nécessaire mais **les divisions sont
notées séparément** — un pipeline qui ignore les mitoses (ex: simple appariement 1-vers-1)
plafonnera son score même avec un edge-Jaccard élevé.

---

## 2. Mise en place de l'environnement

Les données de compétitions Kaggle "Code Competition" doivent être manipulées dans un
notebook Kaggle (CPU ou GPU, 12h max, **sans accès internet** au moment de la soumission).
Pour développer en local d'abord :

```bash
# Environnement conda recommandé
conda create -n celltrack python=3.11 -y
conda activate celltrack

pip install zarr numpy scipy scikit-image pandas networkx tqdm
pip install stardist tensorflow          # détection par deep learning (option A)
pip install cellpose                     # détection par deep learning (option B)
pip install laptrack                     # tracking par programmation linéaire
pip install napari[all]                  # visualisation 3D (très utile pour débugger)
```

Comme les organisateurs (Loïc Royer, Jordão Bragantini, etc.) viennent du **Royer Lab**,
qui maintient la librairie open-source **[ultrack](https://github.com/royerlab/ultrack)**
spécifiquement conçue pour ce type de problème (détection + tracking + divisions en 3D+t),
c'est un excellent point de départ ou de comparaison :

```bash
pip install ultrack
```

---

## 3. Récupérer et explorer les données

1. Rejoignez la compétition sur Kaggle et acceptez les règles.
2. Téléchargez les données via l'API Kaggle :

```bash
pip install kaggle
kaggle competitions download -c biohub-cell-tracking-during-development
```

3. Les volumes sont au format **Zarr** (`.zarr`), un format de tableaux chunkés adapté aux
   gros volumes 3D+t qui ne tiennent pas en RAM. Exploration typique :

```python
import zarr

store = zarr.open("path/to/dataset_44b6.zarr", mode="r")
print(store.tree())          # structure interne (souvent des groupes multi-résolution)
arr = store["0"]              # niveau de résolution 0 (le plus fin), souvent (T, Z, Y, X)
print(arr.shape, arr.dtype)

frame0 = arr[0]                # volume 3D au temps t=0, shape (Z, Y, X)
```

Utilisez **napari** pour visualiser un volume et vous familiariser avec la densité des
cellules, le bruit, et le contraste :

```python
import napari
viewer = napari.Viewer()
viewer.add_image(arr, name="raw", scale=(1.625, 0.40625, 0.40625))  # échelle physique donnée dans l'énoncé
napari.run()
```

Regardez si un fichier d'annotations d'exemple (ground truth) est fourni pour un petit
sous-ensemble ("train") — cela vous permettra de calculer la métrique localement avant de
soumettre.

---

## 4. Étape 1 — Détection des cellules par frame

Objectif : pour chaque `t`, produire une liste de centroïdes `(z, y, x)`.

### Option A — Détection classique (rapide à mettre en place, bonne baseline)
Si les cellules apparaissent comme des blobs relativement contrastés (marquage nucléaire par
exemple), un pipeline classique fonctionne bien pour démarrer :

```python
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from scipy import ndimage as ndi

def detect_cells_blob(volume, sigma=1.5, min_distance=4, threshold_rel=0.1):
    smoothed = gaussian(volume, sigma=sigma, preserve_range=True)
    coords = peak_local_max(
        smoothed, min_distance=min_distance, threshold_rel=threshold_rel
    )
    return coords  # array (N, 3) -> z, y, x
```

Pour une segmentation plus propre (pas juste des pics mais des masques de cellules, utile si
vous voulez ensuite des features de forme/taille), ajoutez un watershed :

```python
from skimage.segmentation import watershed

def segment_watershed(volume, coords):
    markers = np.zeros(volume.shape, dtype=int)
    for i, (z, y, x) in enumerate(coords, start=1):
        markers[z, y, x] = i
    labels = watershed(-volume, markers=markers, mask=volume > volume.mean())
    return labels  # chaque cellule = un label entier
```

### Option B — Détection par deep learning (meilleure précision, plus lourd)
- **StarDist3D** : excellent pour des noyaux convexes/étoilés, modèles pré-entraînés
  disponibles (autorisé par le règlement : "Freely & publicly available external data /
  pre-trained models allowed").
- **Cellpose (mode 3D ou 2.5D par plans)** : plus généraliste, gère mieux les formes
  irrégulières.

```python
from stardist.models import StarDist3D
from csbdeep.utils import normalize

model = StarDist3D.from_pretrained("3D_demo")  # ou un modèle fine-tuné sur vos données
labels, details = model.predict_instances(normalize(frame0))
centroids = [r.centroid for r in regionprops(labels)]
```

💡 Conseil : commencez avec l'option A (rapide) pour avoir un pipeline de bout en bout
fonctionnel, puis remplacez uniquement l'étape de détection par un modèle deep learning une
fois que le reste (tracking, export, évaluation) marche.

---

## 5. Étape 2 — Tracking (association entre frames)

Une fois que vous avez, pour chaque `t`, une liste de centroïdes, il faut les relier dans le
temps. Trois niveaux de sophistication :

### Niveau 1 — Nearest neighbor gourmand (baseline très simple)
Pour chaque cellule à `t`, chercher la plus proche à `t+1` sous un seuil de distance
(**en distance physique, avec l'échelle z=1.625, y=x=0.40625**). Rapide mais fragile en
zones denses.

### Niveau 2 — Assignment optimal frame-à-frame (recommandé comme baseline sérieuse)
Utiliser l'algorithme hongrois (linear sum assignment) entre `t` et `t+1`, avec une matrice
de coût = distance physique entre centroïdes (+ pénalité si distance > seuil ⇒ pas de lien).
C'est exactement ce que fait **laptrack**, qui gère aussi nativement la logique de division
(un node source peut avoir 2 targets) :

```python
import numpy as np
import pandas as pd
from laptrack import LapTrack

# construire un DataFrame avec colonnes: frame, z, y, x (coordonnées physiques, pas voxel!)
coords_df = pd.DataFrame(all_centroids, columns=["frame", "z", "y", "x"])
coords_df[["z", "y", "x"]] *= [1.625, 0.40625, 0.40625]  # passage voxel -> µm

lt = LapTrack(
    track_dist_metric="euclidean",
    track_cost_cutoff=7.0**2,      # même seuil que la métrique (7 µm), en distance^2
    gap_closing_cost_cutoff=False, # activer si vous voulez combler des détections manquées
    splitting_cost_cutoff=7.0**2,  # autorise les divisions (branchement 1 -> 2)
)

track_df, split_df, merge_df = lt.predict_dataframe(
    coords_df, coordinate_cols=["z", "y", "x"], frame_col="frame"
)
```

`laptrack` produit directement une table avec un `track_id` par cellule et une table des
événements de split (division) — ce qui simplifie beaucoup la construction du graphe final.

### Niveau 3 — Approches avancées (si le temps le permet)
- **ultrack** (royerlab) : pipeline complet détection+tracking+division optimisé pour ce
  genre de données denses en 3D+t, avec gestion robuste des divisions et fusions apparentes.
- **Trackastra** : modèle transformer pré-entraîné pour le tracking cellulaire, capable
  d'apprendre les patterns de division directement depuis les données d'exemple.
- **btrack** : tracking bayésien, bon pour les mouvements complexes/non-linéaires.

Étant donné que les organisateurs viennent du Royer Lab, tester **ultrack** en premier est
probablement le choix le plus aligné avec la nature du benchmark.

---

## 6. Étape 3 — Construire le graphe de lignée et l'exporter

Une fois le tracking fait (avec les divisions identifiées), construisez un graphe orienté
(`networkx.DiGraph`) où :
- chaque **node** = une cellule détectée à un temps donné, avec un `node_id` **unique par
  dataset** (pas besoin d'être unique entre datasets différents, chaque dataset a ses propres
  IDs) ;
- chaque **edge** = un lien `source_id -> target_id` entre `t` et `t+1` (une division = un
  node avec 2 edges sortants).

```python
import pandas as pd

def build_submission(all_datasets_tracks):
    """
    all_datasets_tracks: dict {dataset_name: (nodes_df, edges_df)}
      nodes_df colonnes: node_id, t, z, y, x   (z,y,x en VOXELS entiers, pas en µm)
      edges_df colonnes: source_id, target_id
    """
    rows = []
    idx = 0
    for dataset, (nodes_df, edges_df) in all_datasets_tracks.items():
        for _, n in nodes_df.iterrows():
            rows.append({
                "id": idx, "dataset": dataset, "row_type": "node",
                "node_id": int(n.node_id), "t": int(n.t),
                "z": int(round(n.z)), "y": int(round(n.y)), "x": int(round(n.x)),
                "source_id": -1, "target_id": -1,
            })
            idx += 1
        for _, e in edges_df.iterrows():
            rows.append({
                "id": idx, "dataset": dataset, "row_type": "edge",
                "node_id": -1, "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": int(e.source_id), "target_id": int(e.target_id),
            })
            idx += 1

    submission = pd.DataFrame(rows, columns=[
        "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id",
    ])
    submission.to_csv("submission.csv", index=False)
    return submission
```

⚠️ Points d'attention sur le format :
- `z, y, x` des **nodes** sont en **coordonnées voxel entières**, pas en µm (contrairement
  aux distances physiques utilisées en interne pour le matching/coût de tracking).
- Chaque dataset du jeu de test doit apparaître dans la soumission, même si votre pipeline
  n'y détecte rien d'intéressant (sinon erreur de soumission).
- `id` est juste un compteur séquentiel sur tout le fichier, pas par dataset.

---

## 7. Étape 4 — Valider localement avant de soumettre

Ne découvrez pas votre score uniquement via Kaggle (soumissions limitées/jour). Le
repository officiel fournit le détail de la métrique :
https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md

Reproduisez-la en local (bipartite matching des centroïdes avec la distance physique,
seuil 7 µm, puis calcul edge Jaccard + division Jaccard) sur votre split de train/validation
avant chaque soumission — cela vous fait gagner énormément de temps d'itération.

Idée de découpage : gardez un ou deux volumes annotés de côté comme "validation locale",
n'entraînez/tunez rien dessus, et calculez votre propre score avant de committer sur Kaggle.

---

## 8. Contraintes spécifiques à respecter pour la soumission Kaggle

- Le notebook doit tourner en **≤ 12h** (CPU ou GPU), **sans accès internet** activé au
  moment du commit final → téléchargez/committez vos poids de modèles pré-entraînés comme
  "Kaggle Dataset" annexe à l'avance, ne comptez pas sur un `pip install` ou téléchargement
  en ligne pendant l'exécution.
- Le fichier produit doit s'appeler exactement `submission.csv`.
- Testez la vitesse de votre pipeline de détection sur **un seul volume** en local pour
  extrapoler le temps total sur l'ensemble du test set — le deep learning 3D peut être lent,
  pensez à traiter par patches/tuiles ou à sous-échantillonner si nécessaire pour tenir dans
  les 12h.

---

## 9. Feuille de route suggérée (itérative)

1. **Baseline minimale de bout en bout** : détection blob simple (§4 Option A) +
   nearest-neighbor (§5 Niveau 1) + export CSV (§6). But : avoir un `submission.csv` valide
   et un premier score, même faible, pour valider toute la mécanique.
2. **Améliorer le tracking** : passer à `laptrack` (§5 Niveau 2) avec gestion des divisions
   — normalement le plus gros gain de score rapide, car ça active le score de division.
3. **Améliorer la détection** : StarDist3D ou Cellpose fine-tuné sur les données
   d'entraînement fournies, pour réduire les faux positifs/négatifs de nodes (qui pénalisent
   directement l'edge Jaccard).
4. **Explorer ultrack / Trackastra** en remplacement du pipeline détection+tracking complet,
   si le temps le permet — conçus spécifiquement pour ce type de données denses avec
   divisions.
5. **Tuning des seuils** (distance de tracking, seuils de détection) en fonction de votre
   métrique locale (§7), dataset par dataset si leur densité/bruit varie beaucoup.

---

## Ressources utiles

- Détails de la métrique : https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
- FAQ Code Competitions Kaggle : https://www.kaggle.com/docs/competitions#notebooks-only-FAQ
- Debug des erreurs de soumission : https://www.kaggle.com/code-competition-debugging
- `ultrack` (Royer Lab) : https://github.com/royerlab/ultrack
- `laptrack` : https://laptrack.readthedocs.io/
- `StarDist` : https://github.com/stardist/stardist
- `Cellpose` : https://github.com/MouseLand/cellpose
