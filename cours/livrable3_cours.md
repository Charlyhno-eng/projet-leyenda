# Version Workshop 7

### RNN

Un **RNN** est un réseau de neurones qui traite une information **étape par étape** (comme une phrase mot par mot). À chaque étape, il prend l’entrée actuelle (par exemple un mot) et **garde en mémoire un résumé de ce qu’il a déjà vu** grâce à un “état caché”. Cet état est mis à jour à chaque nouveau mot, ce qui lui permet de comprendre le contexte global. Ensuite, il utilise cette mémoire pour prédire la sortie suivante (par exemple le prochain mot dans une phrase).

### Encodeur et décodeur pour le captioning

L’**encodeur** transforme une image en représentation numérique (features), et le **décodeur** génère une phrase mot par mot à partir de ces features pour décrire l’image.

### Mécanisme d’attention

Le **mécanisme d’attention** permet au modèle de se concentrer sur les parties les plus importantes de l’image ou de la séquence au moment de générer chaque mot.

### BahdanauAttention

La **Bahdanau Attention** est un type d’attention qui calcule des poids pour chaque partie des features afin de créer un contexte pertinent pour générer le mot suivant.

---
# BLIP

### Transformer

Architecture d'IA qui analyse tous les éléments d'une séquence (texte, image, etc.) en même temps pour comprendre leur contexte. C'est la base des LLM modernes.

### Transformer multimodal

Transformer capable de traiter plusieurs types de données à la fois (texte, image, audio, vidéo) et de faire des liens entre elles.

### Contrastive learning

Méthode d'entraînement qui apprend à rapprocher les éléments liés (image ↔ description correcte) et à éloigner ceux qui ne le sont pas.

### Captioning auto-réflexif

Technique où le modèle génère une description, puis la réévalue lui-même pour la corriger ou l'enrichir avant de la produire.

### Q-Former

Module qui sert d’intermédiaire entre le modèle de vision et le modèle de langage. Il extrait les informations importantes des features de l’image et les transforme en vecteurs exploitables par le LLM pour générer du texte.

---
### Bibliothèques : 

* **sentencepiece** : outil de tokenisation pour découper le texte en tokens utilisés par les modèles de NLP.
* **accelerate** : bibliothèque de pour optimiser l’exécution des modèles (CPU/GPU).
* **timm** : collection de modèles de vision pré-entraînés (Computer Vision).
* **IPython** : shell Python interactif utilisé notamment dans les notebooks Jupyter.
