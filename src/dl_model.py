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
"""

import torch
import torch.nn as nn

# capacite utilisee par le detecteur heatmap "v2" (train_detector.py / real_detect_dl.py) --
# le segmenteur (train_segmenter.py / real_detect_seg.py) continue d'utiliser le defaut
# de la classe (16,32,64), sur lequel son checkpoint existant a ete entraine.
DETECTOR_LEVELS = (32, 64, 128)


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class HeatmapUNet3D(nn.Module):
    """U-Net 3D minimal : 1 canal en entree (intensite), 1 canal en sortie
    (heatmap de presence cellulaire, sigmoid dans [0, 1])."""

    def __init__(self, in_channels=1, base_channels=16, levels=(16, 32, 64)):
        super().__init__()
        levels = list(levels)

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in levels:
            self.encoders.append(_conv_block(prev, ch))
            prev = ch
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = _conv_block(levels[-1], levels[-1] * 2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = levels[-1] * 2
        for ch in reversed(levels):
            self.upsamples.append(nn.ConvTranspose3d(prev, ch, kernel_size=2, stride=2))
            self.decoders.append(_conv_block(ch * 2, ch))
            prev = ch

        self.head = nn.Sequential(
            nn.Conv3d(levels[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, 1, Z, Y, X)
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.head(x)  # (B, 1, Z, Y, X), valeurs dans [0, 1]


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
