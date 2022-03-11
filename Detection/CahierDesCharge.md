[200~# Cahier des charges d'une solution de detection de maladies dans la vigne

## Cas particulier étudier pendant la rédaction du cahier : La flavescence dorée

-------------

- Une camera couleur (RGB) pour identifier plus facilement les maladies avec des symptomes existants.
- Un apprentissage sur peu de données. (Few shot learning) (plus en classif mais marche en segmentation)
- Un apprentissage permettant l'ajout de nouvelle classe pendant l'entrainement. (apprentissage continue)
- Un modèle unique pour tous les capteurs sur la caméra.
- Prédiction sur chaque longeur d'onde puis fusion pour augmenter la qualité du résultat. (un modele par longueur d'onde)
- Le modèle doit expliquer la zone qui lui à permit de prendre une décision. (Teacher/Studient)
- Utilisation de la profondeur de l'image pour permetre d'utiliser que le premier plan.
- Modèle de segmentation, la classification étant adapté pour un sujet sur une image.
- Ne pas donner les images recalées d'un coup au réseaux mais de faire une prédiction par images. (il a un petit décalage avec les images recalé ce qui fait baisser la précision ?) (RGB and multispectral images were combined as one input then fedto the SegNet. The results obtained were poorer. This is due to the re-gistration of the visible and infrared images. In fact, there is always asmall random shift between the multispectral images, which impliesthat the same pixels are never aligned exactly in the same position.) (Sinon utiliser de la convolution depthwise)
- Le nombre d'exemple est à déterminer mais surement 200-2000 par classe.
- Model de base VddNet
