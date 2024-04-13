<div align="center">   
    
## 🎊Reconnaissance des Expressions Faciales avec SNNs et Caméras Événementielles🎊
</div>

</div align="center">  

(https://arxiv.org/pdf/2304.10211.pdf)

</div>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>

- [📍 Introduction](#-overview)
- [👾 Technologies Utilisées](#-demo)
- [🧩 Installation et configuration ](#-features)
- [🗂️ Structure du projet](#️-examples)
- [📦 Utilisation de v2e pour la Conversion Vidéo-en-Événements](#️-configuration)
- [🚀 Modification du code train.py](#-getting-started)
- [🤖 Comment Utiliser ](#-Comment-Utiliser)
- [🧪 Résultat et Interprétation](#-roadmap)
- [🧑‍💻 Contributing](#-Citation)
- [🎗 License](#-license)
</details>

---

#  📍 Introduction 

Ce projet représente une évolution d'un système préexistant de reconnaissance des expressions faciales (FER) en exploitant les réseaux de neurones à impulsions (Spiking Neural Networks, SNNs) et les caméras événementielles. Nous avons contribué à ce champ en optimisant le code source existant, en entraînant le modèle amélioré avec la base de données CKPLUS, et en effectuant des tests et évaluations pour en mesurer les performances.

## Exigence de Configuration pour l'Environnement de Développement

Afin d'assurer la meilleure compatibilité et performance pour l'entraînement de notre modèle Spiking-Fer et pour toute manipulation des données associées, il est impératif d'utiliser **Python 3.8.10**.

## 👾 Technologies Utilisées

- **Python 3.8.10**
- **PyTorch et PyTorch Lightning**
- **h5py** pour la manipulation des fichiers H5
- **Système de caméras événementielles**

## 🧩 Installation et Configuration

1. **Clonez le dépôt** :
```
git clone https://github.com/Boubker10/Projet_P6_Reconnaissance_des_Expressions_Faciales_avec_SNN
```
3. **Installez les dépendances nécessaires** :
 ```
 pip install -r requirements.txt
 ```
5. **Assurez-vous que votre environnement de travail dispose d'un GPU compatible CUDA** pour l'entraînement du modèle.

##  🗂️ Structure du Projet

Le projet est organisé de manière à faciliter l'accès et la manipulation des différents éléments nécessaires pour l'entraînement, l'évaluation, et la mise en œuvre des modèles de reconnaissance des expressions faciales utilisant les réseaux de neurones à impulsions (SNNs) et les caméras événementielles.

### `data/FerDVS/CKPlusDVS/`
Ce dossier contient la base de données CKPlusDVS, utilisée pour l'entraînement et l'évaluation du modèle. Ces données sont spécifiquement optimisées pour une utilisation avec des SNNs et des caméras événementielles, offrant une richesse de détails et de nuances essentielles à la reconnaissance précise des expressions faciales.

### `experiments/`
Destiné à stocker les résultats des différentes expérimentations menées au cours du projet. Il inclut les logs d'entraînement, les graphiques de performances, et des analyses détaillées, permettant une évaluation rigoureuse des modèles développés.

### `myenv/`
Contient les configurations nécessaires à la création d'un environnement virtuel Python. Ceci garantit que le projet peut être reproduit avec les mêmes dépendances et versions de librairies, assurant ainsi la cohérence et la fiabilité des résultats.

### `project/`
Dossier racine qui encapsule les éléments clés du projet, structuré comme suit :

##### `datamodules/`
Héberge les modules de données spécifiques au projet, permettant une manipulation efficace et structurée des ensembles de données.

##### `fer_dvs.py`
Script destiné à la préparation et au chargement des datasets de reconnaissance des expressions faciales (FER), utilisant des données issues de caméras DVS (Dynamic Vision Sensors).

##### `models/`
Répertoire regroupant les différents modèles utilisés dans le projet, incluant :

- `models.py` : Définit les structures de base des modèles pour l'entraînement et l'évaluation.
- `sew_resnet.py` : Implémente une variante du modèle ResNet, adaptée aux spécificités des SNNs.
- `snn_models.py` : Contient les définitions des modèles de réseaux de neurones à impulsions spécifiquement conçus pour la reconnaissance des expressions faciales.

##### `utils/`
Dossier qui comprend divers utilitaires et transformations nécessaires au traitement des données et au fonctionnement optimal du modèle, incluant :

- `drop_event.py` : Utilitaire simulant la perte d'événements dans les données DVS, utilisé comme méthode d'augmentation de données.
- `dvs_noises.py` : Script d'ajout de bruit spécifique aux capteurs DVS dans les données, visant à simuler des conditions réelles de manière plus fidèle.
- `transform_dvs.py` : Propose des fonctions de transformation des données DVS pour leur prétraitement avant l'entraînement ou l'évaluation des modèles.
- `transforms.py` : Implémente des transformations d'image génériques et des augmentations de données utilisées tout au long du processus d'entraînement.

### Fichiers Additionnels

- `.gitignore` : Configure Git pour ignorer les fichiers et dossiers non essentiels lors des commits.
- `LICENSE` : Document important à consulter pour comprendre les modalités d'utilisation et de partage du projet.
- `README.md` : Fournit une vue d'ensemble du projet, incluant des instructions d'installation, d'utilisation et des conseils pour contribuer.
- `output_video.mp4` : Vidéo démonstrative des résultats du modèle ou utilisée comme entrée pour les tests.
- `report_snn_CKPlusDVS.txt` : Rapport détaillé des résultats, analyses et conclusions de l'entraînement et de l'évaluation du modèle sur la base de données CKPlusDVS.
- `requirements.txt` : Liste les dépendances Python requises pour exécuter le projet. Utilisez `pip install -r requirements.txt` pour installer ces dépendances dans votre environnement de travail.

## 🚀 Modifications Apportées au Code de `train.py`

### Nombre de Workers pour le DataLoader

- **Premier Code**: Calcule dynamiquement `train_workers` et `val_workers` en fonction de la taille du jeu de données et de la taille du lot (`batch_size`), utilisant `math.ceil(len(dataset) / batch_size)`.
- **Second Code**: Utilise une valeur fixe de 8 pour `train_workers` et `val_workers`.

### Options de DataLoader

- **Premier Code**: Ne spécifie pas d'options supplémentaires pour `DataLoader`.
- **Second Code**: Ajoute `persistent_workers=True` pour les `DataLoader` de formation et de validation, améliorant ainsi la gestion des données et l'efficacité du chargement.

### Configuration du Trainer de PyTorch Lightning

- **Premier Code**: Utilise `torch.cuda.device_count()` pour définir le nombre de GPUs disponibles.
- **Second Code**: Emploie une condition pour utiliser 1 GPU si disponible (`1 if torch.cuda.is_available() else None`). Cette approche permet une plus grande flexibilité et une meilleure adaptation aux ressources matérielles disponibles.

#### Ajouts Spécifiques dans le Second Code

- Introduit une variable globale `xy` et un bloc de code qui capture les données d'entrée et les étiquettes du premier lot dans `train_loader`, les stocke dans `xy`, puis sort de la boucle. Cette technique est utile pour le débogage et l'inspection des données.
- Ajoute `log_every_n_steps=5` dans la configuration du `Trainer`, facilitant un suivi plus fréquent et plus détaillé de l'entraînement.

Ces différences soulignent une adaptation du code pour des tests ou des démonstrations rapides, une amélioration des performances avec `persistent_workers=True`, et l'intégration d'une logique additionnelle pour la capture préliminaire des données.

## 📦 Utilisation de v2e pour la Conversion Vidéo-en-Événements

#### Utilisation de v2e dans Google Colab
(https://colab.research.google.com/drive/1czx-GJnx-UkhFVBbfoACLVZs8cYlcr_M?usp=sharing).
Google Colab représente une solution pratique pour les utilisateurs cherchant à éviter l'installation locale de v2e ou ceux travaillant sur des systèmes sans les privilèges nécessaires pour les installations logicielles. Offrant un environnement de notebook Jupyter hébergé dans le cloud avec accès à des ressources de calcul gratuites, Google Colab simplifie les conversions vidéo-en-événements. Pour utiliser v2e dans Google Colab, il suffit d'accéder au notebook v2e disponible sur la plateforme.

#### Installation Locale

Pour une intégration plus profonde de v2e dans des pipelines de données personnalisés, une installation locale peut être préférable. Voici les étapes recommandées :

1. **Création d'un Environnement Conda**: Isoler les dépendances de v2e pour éviter les conflits avec d'autres installations.
2. **Installation de PyTorch et d'Autres Paquets via Conda**: Assurer la compatibilité et la stabilité de l'environnement en installant PyTorch et d'autres paquets nécessaires.
3. **Installation des Paquets Restants et de v2e avec pip**: Compléter l'installation en ajoutant les paquets non disponibles via Conda, y compris v2e.

Suivre cet ordre d'installation minimise les problèmes de compatibilité et prépare un environnement solide pour l'exécution de v2e et la conversion des vidéos en formats d'événements adaptés aux SNNs.
#### Démo : 

-  video original 

https://github.com/Boubker10/demo/assets/116897761/bfb852bd-8f0c-4f6e-b667-ea6fbcd8d3f1  

- video event

https://github.com/Boubker10/demo/assets/116897761/bfdcd93e-7c3f-44dc-86ce-545f6e81e294
 








## 🤖 Comment Utiliser 
### Étape 1 : Clonage du Projet
La première étape consiste à obtenir une copie locale du projet. Ceci est réalisé en clonant le dépôt GitHub. Le clonage est une opération cruciale car elle vous permet d'accéder à toutes les ressources du projet, y compris les scripts d'entraînement, les modèles, les utilitaires et les exigences. Utilisez la commande suivante pour cloner le projet :
```
[git clone https://github.com/Boubker10/Projet_P6_Reconnaissance_des_Expressions_Faciales_avec_SNN
```


### Étape 2 : Installation des Dépendances
Une fois le projet cloné, la prochaine étape cruciale est l'installation des dépendances. Ce projet, comme beaucoup dans le domaine de l'intelligence artificielle et du machine learning, dépend de plusieurs bibliothèques externes Python. L'installation de ces dépendances est simplifiée grâce au fichier requirements.txt fourni dans le projet. En exécutant la commande suivante dans le répertoire du projet, toutes les dépendances nécessaires seront installées :
```
pip install -r requirements.txt
```

Cette étape assure que votre environnement Python dispose de tous les outils nécessaires pour l'entraînement et l'évaluation des modèles.


### Étape 3 : Préparation des Datasets
Avant de pouvoir lancer une expérience d'entraînement, il est essentiel de préparer les datasets. Les données doivent être placées dans le dossier data/ du projet. Cette organisation facilite l'accès aux données par les scripts d'entraînement et permet une gestion cohérente des datasets. La préparation des données peut inclure la conversion de vidéos en données d'événements si vous travaillez avec des caméras DVS, un processus pour lequel des instructions détaillées sont fournies dans le projet.




### Étape 4 : Lancement d'une Expérience d'Entraînement
Avec les dépendances installées et les données préparées, vous êtes maintenant prêt à lancer une expérience d'entraînement. Cette étape est cruciale car elle permet d'entraîner le modèle sur le dataset spécifié, en utilisant des configurations définies pour les augmentations de données. Exécutez la commande suivante pour démarrer l'entraînement :

```
python train.py --dataset="CKPlusDVS" --mode="snn" --fold_number=0 --edas="flip,background_activity,crop,reverse,mirror,event_drop"

```

Cette commande configure l'entraînement pour le dataset CKPlusDVS en utilisant un réseau de neurones à impulsions (SNN). Elle spécifie également une série de transformations d'augmentation des données pour améliorer la robustesse et la performance du modèle. Les résultats de cet entraînement, y compris le modèle le mieux performant, seront sauvegardés dans le dossier experiments/, vous permettant d'évaluer l'efficacité du modèle formé.


## 🧪Évaluation et Test 

### Configuration de l'évaluation
Pour évaluer notre modèle, nous utilisons un ensemble de transformations définies pour les données d'entrée, via la classe `DVSTransform`. Cette classe applique une série de transformations spécifiques adaptées à nos données, incluant :

- flip : retournement des images
- background_activity : modification de l'activité de fond
- crop : découpe des images
- reverse : inversion des séquences
- mirror : effet miroir
- event_drop : suppression d'événements

Ces transformations sont adaptées à la taille du capteur des données `FerDVS` et configurées pour concaténer les canaux temporels dans un format `snn`, adapté pour les réseaux de neurones spiking.

### Chargement du modèle
Nous rechargeons le modèle à partir d'un checkpoint sauvegardé (`checkpoint_path`), après un entraînement préalable sur des données spécifiques. Le modèle utilise une architecture de réseau de neurones spiking (SNN), avec plusieurs couches de convolution et des noeuds de type Integrate-and-Fire, simulant le comportement de neurones biologiques.

### Préparation des données pour l'évaluation
Pour tester le modèle, nous chargeons un exemple de données sous forme d'événements depuis un fichier `.h5`. Ces événements sont ensuite filtrés et transformés pour correspondre aux entrées attendues par notre modèle. Les données transformées sont converties en tenseurs PyTorch, adaptées à la dimension attendue par le modèle avant leur soumission pour évaluation.

### Test du modèle
Le modèle est mis en mode évaluation (`model.eval()`), avec le calcul du gradient désactivé pour optimiser les performances pendant les tests. Les données préparées sont soumises au modèle pour obtenir des prédictions. Ces prédictions sont ensuite traitées avec une fonction softmax pour convertir les logits en probabilités. La classe avec la probabilité la plus élevée est sélectionnée comme prédiction finale.

[https://github.com/Boubker10/Projet_P6_Reconnaissance_des_Expressions_Faciales_avec_SNN/assets/116897761/4a080c47-b543-417f-ad92-eead69955210](https://github.com/Boubker10/Projet_P6_Reconnaissance_des_Expressions_Faciales_avec_SNN/issues/1#issue-2241451913)


























##  🧑‍💻 Contributing 
```
@title={Spiking-Fer: Spiking Neural Network for Facial Expression Recognition With Event Cameras},
  author={Boubker BENNANI , Othmane BENZARHOUNI},
  year={2024}
}
