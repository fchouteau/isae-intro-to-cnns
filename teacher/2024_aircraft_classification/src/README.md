# Khumeia

>""*alchimie, du latin médiéval alchemia issu de l’arabe كِيمِيَاءُ, (al)kîmiyâʾ (« (la) chimie, art de faire de l'or, art de purifier son coeur »), lui-même issu du grec ancien χυμεία, khumeía (« art de faire fondre les métaux »).*

*`khumeia`* est un petit framework d'aide à l'interaction avec les images satellites - qui sont des images de dimensions assez importantes de l'ordre de 6000x6000 pixels.

Il permet notamment de ne pas avoir à réécrire le décodage des fichiers labels, ni les fichiers, et vise à faciliter l'interaction avec la donnée avant et après l'entraînement du modèle en lui même.

Sont mis à disposition des utilitaires tels que la gestion des zones d'intérêts ("tuiles") dans des grandes images, ainsi que des fenêtres glissantes, et d'une proposition d'implémentation de mécanismes d'échantillonages.

Sont aussi fournies des wrappers "haut niveau" sur le framework dans le package `khumeia.helpers`
