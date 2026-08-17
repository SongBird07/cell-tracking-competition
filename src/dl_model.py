"""
Petit U-Net 3D entierement convolutif, pour predire une carte de chaleur
(heatmap) de presence de centre cellulaire a partir d'un volume brut.

Contrairement a detect.py (seuillage + maxima locaux sur l'image brute lissee),
ce modele APPREND a quoi ressemble un centre de cellule a partir des donnees
d'entrainement -- utile quand l'intensite brute ne suffit pas a distinguer les
cellules du bruit/fond (zones sombres, cellules qui se touchent, etc.).

Le reseau est entierement convolutif (aucune couche dense) : il peut donc
etre applique a un patch d'entrainement de petite taille OU directement a un
volume entier (Z, Y, X) de taille differente, tant que chaque dimension est
divisible par 2^n_levels (ici 8, pour 3 niveaux de pooling).

`dropout`, `deep_supervision` et `predict_distance` sont desactives par
defaut (0.0 / False / False) -- avec ces valeurs, l'architecture et le
state_dict sont IDENTIQUES a la version d'origine, donc les checkpoints deja
entraines restent compatibles (chargement strict=False tolerant aux cles
purement additives, voir train_detector.py::_load_state_dict_resumable et
real_detect_dl.py::load_model).

`norm_type` (defaut "batch") N'EST PAS un ajout purement additif -- passer a
"groupnorm" REMPLACE le type de couche de normalisation partout dans le
reseau. Contrairement a BatchNorm3d, GroupNorm n'accumule aucune statistique
de "running_mean/running_var" : un checkpoint "groupnorm" charge dans un
modele "batch" (ou l'inverse) transfererait bien les poids/biais (memes
formes) mais PAS les vraies statistiques de normalisation, produisant des
predictions silencieusement fausses sans erreur de chargement. C'est
pourquoi train_detector.py persiste `norm_type` DANS chaque checkpoint et
real_detect_dl.py le relit pour reconstruire la bonne architecture avant de
charger les poids -- ne jamais forcer `norm_type` a l'inference sans passer
par ce mecanisme.
"""

import torch
import torch.nn as nn

# capacite utilisee par le detecteur heatmap "v2" (train_detector.py / real_detect_dl.py) --
# le segmenteur (train_segmenter.py / real_detect_seg.py) continue d'utiliser le defaut
# de la classe (16,32,64), sur lequel son checkpoint existant a ete entraine.
DETECTOR_LEVELS = (32, 64, 128)


def _make_norm(out_ch, norm_type):
    if norm_type == "groupnorm":
        # 8 groupes : divise exactement tous les canaux utilises ici (16/32/64/128/256),
        # valeur pratique courante quand le nombre de canaux ne justifie pas le
        # defaut historique de 32 groupes (GroupNorm original paper).
        return nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
    return nn.BatchNorm3d(out_ch)


def _conv_block(in_ch, out_ch, norm_type="batch"):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        _make_norm(out_ch, norm_type),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        _make_norm(out_ch, norm_type),
        nn.ReLU(inplace=True),
    )


class HeatmapUNet3D(nn.Module):
    """U-Net 3D minimal : 1 canal en entree (intensite), 1 canal en sortie
    (heatmap de presence cellulaire, sigmoid dans [0, 1]).

    dropout : probabilite de Dropout3d applique au bottleneck (0.0 = aucun,
        comportement d'origine). Sans effet en eval() -- inoffensif pour
        l'inference meme si le modele a ete entraine avec dropout>0.
    deep_supervision : si True, calcule aussi des predictions auxiliaires a
        chaque niveau intermediaire du decodeur (upsamplees a la resolution
        finale), accessibles apres un forward() via `self.aux_outputs`.
    predict_distance : si True, calcule EN PLUS une carte de distance
        normalisee aux centres cellulaires les plus proches (tache auxiliaire
        de supervision multi-tache -- force le reseau a apprendre une
        representation geometrique plus riche que la seule reponse
        gaussienne), accessible apres un forward() via `self.distance_output`.
    norm_type : "batch" (defaut, BatchNorm3d) ou "groupnorm" (GroupNorm,
        independant de la taille de batch -- voir avertissement en tete de
        fichier, ce n'est PAS un changement purement additif.

    Le RETOUR de forward() reste toujours un seul tenseur (compatibilite
    totale avec le code d'inference existant) -- aux_outputs/distance_output
    sont des attributs annexes, uniquement utilises pour les pertes
    supplementaires a l'entrainement.
    """

    def __init__(self, in_channels=1, base_channels=16, levels=(16, 32, 64),
                 dropout=0.0, deep_supervision=False, predict_distance=False,
                 norm_type="batch"):
        super().__init__()
        levels = list(levels)

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in levels:
            self.encoders.append(_conv_block(prev, ch, norm_type))
            prev = ch
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = _conv_block(levels[-1], levels[-1] * 2, norm_type)
        self.dropout = nn.Dropout3d(dropout) if dropout and dropout > 0 else nn.Identity()

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = levels[-1] * 2
        for ch in reversed(levels):
            self.upsamples.append(nn.ConvTranspose3d(prev, ch, kernel_size=2, stride=2))
            self.decoders.append(_conv_block(ch * 2, ch, norm_type))
            prev = ch

        self.head = nn.Sequential(
            nn.Conv3d(levels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.deep_supervision = deep_supervision
        if deep_supervision:
            # une tete auxiliaire par etage du decodeur SAUF le dernier (qui
            # correspond deja a la tete principale ci-dessus)
            aux_channels = list(reversed(levels))[:-1]
            self.aux_heads = nn.ModuleList([
                nn.Sequential(nn.Conv3d(ch, 1, kernel_size=1), nn.Sigmoid())
                for ch in aux_channels
            ])
        self.aux_outputs = []

        self.predict_distance = predict_distance
        if predict_distance:
            self.distance_head = nn.Sequential(
                nn.Conv3d(levels[0], 1, kernel_size=1),
                nn.Sigmoid(),
            )
        self.distance_output = None

    def forward(self, x):
        # x: (B, 1, Z, Y, X)
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        aux_raw = []
        n_stages = len(self.upsamples)
        for i, (up, dec, skip) in enumerate(zip(self.upsamples, self.decoders, reversed(skips))):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
            if self.deep_supervision and i < n_stages - 1:
                aux_raw.append(self.aux_heads[i](x))

        out = self.head(x)  # (B, 1, Z, Y, X), valeurs dans [0, 1]

        if self.deep_supervision:
            self.aux_outputs = [
                nn.functional.interpolate(a, size=out.shape[2:], mode="trilinear", align_corners=False)
                for a in aux_raw
            ]
        else:
            self.aux_outputs = []

        self.distance_output = self.distance_head(x) if self.predict_distance else None

        return out


if __name__ == "__main__":
    model = HeatmapUNet3D()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HeatmapUNet3D: {n_params:,} parametres")

    dummy = torch.randn(2, 1, 32, 64, 64)
    out = model(dummy)
    print(f"input {tuple(dummy.shape)} -> output {tuple(out.shape)}")

    dummy_full = torch.randn(1, 1, 64, 256, 256)
    out_full = model(dummy_full)
    print(f"input {tuple(dummy_full.shape)} -> output {tuple(out_full.shape)} (taille frame reelle)")

    print("\n--- test dropout + deep_supervision + predict_distance + groupnorm ---")
    model2 = HeatmapUNet3D(levels=DETECTOR_LEVELS, dropout=0.1, deep_supervision=True,
                            predict_distance=True, norm_type="groupnorm")
    out2 = model2(dummy)
    print(f"sortie principale: {tuple(out2.shape)}")
    print(f"sorties auxiliaires: {[tuple(a.shape) for a in model2.aux_outputs]}")
    print(f"sortie distance: {tuple(model2.distance_output.shape)}")
    has_running_stats = any("running_mean" in k for k in model2.state_dict().keys())
    print(f"groupnorm actif -- presence de running_mean dans le state_dict: {has_running_stats} (doit etre False)")

    # verifie que dropout=0/deep_supervision=False/predict_distance=False/
    # norm_type="batch" donne EXACTEMENT le meme nombre de parametres que
    # l'architecture d'origine (compatibilite checkpoint garantie)
    model3 = HeatmapUNet3D(levels=DETECTOR_LEVELS)
    n_params3 = sum(p.numel() for p in model3.parameters())
    print(f"\nHeatmapUNet3D(levels=DETECTOR_LEVELS) sans les nouvelles options: {n_params3:,} parametres")
