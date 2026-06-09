### Batch size

Le batch size correspond au nombre d’exemples traités par le modèle avant de mettre à jour ses poids. Un petit batch rend l’apprentissage plus précis mais plus lent, tandis qu’un grand batch accélère l’entraînement mais peut rendre l’apprentissage moins stable.

### Epochs

Une epoch correspond à un passage complet de toutes les données dans le modèle. Plus il y a d’epochs, plus le modèle voit les données plusieurs fois et apprend, mais trop d’epochs peut entraîner un surapprentissage.

### Seed

Le seed est une valeur fixe utilisée pour rendre les résultats reproductibles. Avec le même seed, on obtient les mêmes séparations de données et les mêmes résultats d’entraînement.

### Autotune

Autotune permet à TensorFlow d’optimiser automatiquement certains paramètres comme le préchargement des données. Cela améliore les performances sans avoir à configurer manuellement les valeurs.

### Softmax

Softmax est une fonction qui transforme un ensemble de scores en probabilités. Chaque sortie devient une probabilité comprise entre 0 et 1, et la somme de toutes les probabilités vaut 1.

### EarlyStopping

EarlyStopping arrête automatiquement l’entraînement lorsque les performances du modèle ne s’améliorent plus. Cela permet d’éviter le surapprentissage et de gagner du temps.

### ReduceLROnPlateau

ReduceLROnPlateau réduit automatiquement le learning rate lorsque la performance du modèle stagne. Cela permet d’affiner l’apprentissage pour obtenir de meilleurs résultats.

### Stride

Il s'agit du nombre de pixels dont un filtre se déplace à chaque étape lors d'une convolution. Un stride de 1 examine l'image plus finement, tandis qu'un stride plus grand (par exemple 2) saute des positions. On l'utilise pour réduire la taille des données, accélérer les calculs et diminuer l'utilisation de la mémoire, au prix d'une possible perte de détails.

---
### L1 vs Dropout

La régularisation L1 et le dropout ont le même objectif général : réduire le surapprentissage (overfitting), mais ils agissent différemment.

La régularisation L1 ajoute une pénalité sur la valeur absolue des poids du modèle. Cela pousse certains poids à devenir exactement zéro, ce qui rend le modèle plus simple et plus “sparse” (moins de connexions utiles).

Le dropout, lui, désactive aléatoirement des neurones pendant l’entraînement. Cela force le réseau à ne pas dépendre trop de certains neurones et à apprendre des représentations plus robustes.

Lien entre les deux : ils empêchent tous les deux le modèle de trop “mémoriser” les données, mais L1 agit sur les poids, tandis que dropout agit sur les neurones pendant l’entraînement.

### L2 vs EarlyStopping

La régularisation L2 et early stopping ont aussi un objectif commun : éviter le surapprentissage, mais leur fonctionnement est différent.

La régularisation L2 ajoute une pénalité sur les grands poids. Cela pousse les poids à rester petits mais sans les annuler complètement, ce qui rend le modèle plus stable et plus général.

EarlyStopping, lui, ne modifie pas directement le modèle. Il arrête simplement l’entraînement lorsque les performances sur les données de validation n’améliorent plus.

Lien entre les deux : L2 agit en continu pendant l’apprentissage en limitant la taille des poids, tandis que early stopping agit au niveau du temps en arrêtant l’entraînement au bon moment.

---
## Opérations de pre-processing

- `RandomFlip` : retourne aléatoirement les images horizontalement ou verticalement pour rendre le modèle plus robuste aux changements d’orientation.

- `RandomRotation` : applique une légère rotation aléatoire aux images afin que le modèle apprenne à reconnaître les objets même s’ils ne sont pas parfaitement droits.

- `RandomZoom` : zoome ou dézoome légèrement les images de façon aléatoire pour aider le modèle à reconnaître les objets à différentes tailles.

---

- `labels='inferred'` : TensorFlow attribue automatiquement les labels (catégories) des images en se basant sur le nom des dossiers dans lesquels elles se trouvent. Par exemple, les images dans le dossier `cats/` auront le label “cat”.

- `subset="both"` : lorsqu’on sépare les données en entraînement et validation avec `validation_split`, cette option permet de récupérer les deux ensembles en même temps : le jeu d’entraînement et le jeu de validation.

- `test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)` : `cache()` garde les données déjà chargées en mémoire pour éviter de relire les fichiers à chaque fois, et `prefetch()` prépare les prochaines données pendant que le modèle travaille sur les données actuelles afin d’accélérer l’exécution. `AUTOTUNE` laisse TensorFlow choisir automatiquement le meilleur réglage.

---
## Explications des différentes couches du CNN

- `Rescaling` : normalise les pixels des images (par exemple de 0–255 vers 0–1) pour faciliter l’apprentissage du modèle.

- `Couches de convolution` : applique des filtres sur l’image pour détecter des caractéristiques importantes comme les contours, formes ou textures.

- `Couches de pooling` : réduit la taille des images/features pour diminuer le nombre de calculs tout en gardant les informations importantes.

- `Dropout` : désactive aléatoirement certains neurones pendant l’entraînement pour éviter que le modèle apprenne “par cœur” (overfitting).

- `Flatten` : transforme les données en un grand vecteur 1D afin de pouvoir les envoyer aux couches finales du réseau.

- `Dense` : couche entièrement connectée qui combine les informations apprises pour effectuer la prédiction finale (classification, régression, etc.).

---

## ReLu vs Sigmoid function vs Hyperbolic tangent

| Fonction                       | Description simple                                   | Valeurs de sortie | Avantages                                                                                     | Inconvénients                                                     |
| ------------------------------ | ---------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `ReLU` (Rectified Linear Unit) | Garde les valeurs positives et met les négatives à 0 | `0 → +∞`          | Très rapide, efficace pour les réseaux profonds, réduit le problème du gradient qui disparaît | Les neurones peuvent devenir inactifs si la sortie reste à 0      |
| `Sigmoid`                      | Transforme une valeur en probabilité                 | `0 → 1`           | Utile pour les sorties binaires (ex : oui/non)                                                | Plus lente, problème de gradient qui disparaît sur grands réseaux |
| `Tanh` (Hyperbolic Tangent)    | Similaire à Sigmoid mais centrée autour de 0         | `-1 → 1`          | Meilleure que Sigmoid pour des données positives/négatives                                    | Peut aussi souffrir du problème de gradient qui disparaît         |

Le **problème du gradient qui disparaît** (vanishing gradient) arrive quand, pendant l’entraînement d’un réseau de neurones, les corrections des poids deviennent de plus en plus petites au fil des couches.

---
### Loss optimization

La loss optimization consiste à ajuster les paramètres d’un modèle pour minimiser une fonction de perte (loss). Cette fonction mesure l’erreur entre les prédictions du modèle et les vraies valeurs. L’objectif est donc de trouver les paramètres qui donnent la plus petite erreur possible.

### Descente de gradient

La descente de gradient est une méthode utilisée pour optimiser la loss. Elle consiste à modifier progressivement les poids du modèle dans la direction qui réduit l’erreur. On calcule la pente (le gradient) de la loss, puis on ajuste les paramètres en sens inverse pour descendre vers un minimum.

### Optimisation convexe

L’optimisation convexe concerne les problèmes où la fonction de perte a une forme “en bol” (convexe). Dans ce cas, il n’existe qu’un seul minimum global, ce qui garantit que la descente de gradient trouvera la meilleure solution. Dans les problèmes non convexes (comme les réseaux de neurones), il peut exister plusieurs minima locaux, ce qui rend l’optimisation plus complexe.

---
## Algorithmes d’optimisation

#### **Stochastic Gradient Descent (SGD)**

C’est la version de base de la descente de gradient. Il met à jour les poids après chaque petit batch de données. Il est simple, mais peut être lent et bruité dans son apprentissage. (base simple)

#### **SGD avec Momentum**

Amélioration de SGD qui ajoute une “mémoire” des gradients passés. Cela permet d’accélérer l’apprentissage dans la bonne direction et de réduire les oscillations. (SGD amélioré avec inertie)

#### **AdaGrad**

Adapte le learning rate pour chaque paramètre en fonction de son historique. Les paramètres rarement mis à jour ont des grands pas, les autres des petits pas. Utile pour les données creuses, mais le learning rate peut devenir trop petit avec le temps. (learning rate adaptatif (ancien))

#### **RMSProp**

Corrige le problème d’AdaGrad en gardant une moyenne glissante des gradients. Il est efficace pour les réseaux de neurones et fonctionne bien sur des problèmes non stationnaires. (version améliorée d’AdaGrad)

### **Adam**

Combine Momentum et RMSProp. Il adapte le learning rate et garde une mémoire des gradients passés. C’est l’un des algorithmes les plus utilisés car il est rapide, stable et performant par défaut. (combinaison moderne la plus utilisée)