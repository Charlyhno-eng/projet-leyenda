### Autoencodeur

Réseau de neurones qui apprend à compresser une donnée en une représentation plus petite puis à la reconstruire. Il est composé d'un **encodeur** et d'un **décodeur**.

### MLP (Multi-Layer Perceptron)

Réseau de neurones constitué de plusieurs couches de neurones entièrement connectées, utilisé pour la classification, la régression et d'autres tâches d'apprentissage automatique.

### Bruit Gaussien

Bruit aléatoire suivant une distribution normale (courbe de Gauss), souvent ajouté aux données pour rendre un modèle plus robuste.

### `c2 = BatchNormalization()(c2)`

Applique une **normalisation des activations** de la couche `c2`, ce qui stabilise et accélère l'entraînement du réseau.

### `u1 = UpSampling2D((2,2))(b)`

Agrandit la taille spatiale des caractéristiques de `b` en doublant la hauteur et la largeur de l'image ou de la carte de caractéristiques.

### `u1 = Concatenate()([u1, c2])`

Fusionne les caractéristiques de `u1` et `c2` en les concaténant, afin de combiner des informations provenant de différentes couches du réseau.

### Bottleneck

Couche centrale d'un réseau (souvent dans un autoencodeur ou un U-Net) où l'information est la plus compressée. Elle force le modèle à conserver uniquement les caractéristiques les plus importantes.
