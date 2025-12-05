# Classification d'Images Thoraciques COVID-19

> Pipeline de Deep Learning pour l'aide au diagnostic médical via classification automatique de radiographies pulmonaires

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

---

## Vue d'ensemble

Ce projet développe un système de classification automatique d'images radiographiques thoraciques pour la détection de COVID-19, utilisant des réseaux de neurones convolutifs (CNN) pré-entraînés. Le pipeline intègre des techniques avancées de preprocessing, de transfer learning, et d'optimisation de seuils pour maximiser les performances sur des datasets déséquilibrés.

### Valeur métier

- **Aide au diagnostic** : Support décisionnel pour les professionnels de santé
- **Rapidité** : Classification en temps réel sur images radiographiques
- **Précision** : 96.96% d'accuracy globale, 91.98% de précision COVID
- **Robustesse** : Détection et filtrage automatique d'artefacts (textes, annotations)
- **Interprétabilité** : Visualisations Grad-CAM pour validation clinique

### Cas d'usage

- **Dépistage de masse** : Tri rapide des cas suspects
- **Support diagnostic** : Aide à la décision clinique en complément de l'expertise médicale
- **Recherche médicale** : Analyse de grandes cohortes d'images radiographiques
- **Formation** : Outil pédagogique pour l'interprétation d'images médicales

---

## Performances

### Métriques principales (ResNet18, validation set)

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **Accuracy globale** | **96.96%** | Taux de classification correcte |
| **Précision COVID** | **91.98%** | Fiabilité des prédictions positives |
| **Sensibilité (Recall)** | **90.09%** | Capacité à détecter les cas COVID |
| **Spécificité** | **98.38%** | Capacité à identifier les cas normaux |
| **F1-score COVID** | **91.03%** | Équilibre précision/rappel |
| **AUC-ROC** | **0.9924** | Excellente capacité de discrimination |

### Matrice de confusion (seuil optimal)

```text
                Prédit
              Non-COVID    COVID
Réel
Non-COVID      17,265      284
COVID            358     3,256
```

**Résultats** : Sur 21,163 images de validation

- **Vrais Positifs** : 3,256 cas COVID correctement identifiés
- **Vrais Négatifs** : 17,265 cas normaux correctement identifiés
- **Faux Positifs** : 284 (1.6% des non-COVID)
- **Faux Négatifs** : 358 (9.9% des COVID)

---

## Architecture technique

### Modèles disponibles

| Architecture | Paramètres | Profondeur | Recommandation |
|--------------|------------|------------|----------------|
| ResNet18 | ~11M | 18 couches | Développement rapide |
| ResNet34 | ~21M | 34 couches | Bon compromis |
| **ResNet50** | **~25M** | **50 couches** | **Production recommandée** |
| ResNet152 | ~60M | 152 couches | Performance maximale |
| DenseNet121 | ~8M | 121 couches | Efficace en paramètres |
| DenseNet161 | ~29M | 161 couches | Haute performance |

**Configuration par défaut** : ResNet50 avec fine-tuning sélectif

### Stack technique

- **Framework** : PyTorch 2.0+
- **Modèles pré-entraînés** : torchvision (ImageNet)
- **Preprocessing** : OpenCV, NumPy
- **Métriques** : scikit-learn
- **Visualisation** : Matplotlib, Grad-CAM
- **Notebooks** : Jupyter

### Pipeline de traitement

```text
Images brutes
    ↓
Détection d'artefacts (FFT + spatial)
    ↓
Application de masques pulmonaires
    ↓
Augmentation de données (train)
    ↓
Transfer Learning (ResNet/DenseNet)
    ↓
Entraînement avec early stopping (F1-score)
    ↓
Optimisation du seuil de décision
    ↓
Évaluation et visualisation (Grad-CAM)
```

---

## Démarrage rapide

### Prérequis

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA (optionnel, pour GPU)
```

### Installation

```bash
# Cloner le repository
git clone <repository-url>
cd SEP25_BMLE_Covid19

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration

Éditer `src/features/training_config.py` :

```python
MODEL_CHOICE = 'resnet50'  # Architecture du modèle
TRANSFER_MODE = 'fine_tuning'  # Stratégie de transfer learning
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DATASET_ROOT = Path('/chemin/vers/dataset')
```

### Entraînement

```bash
cd src/features
python train_and_evaluate_f1.py
```

### Évaluation d'un modèle existant

1. Configurer `MODEL_CHECKPOINT_PATH` dans `training_config.py`
2. Mettre `TRAIN_MODEL = False`
3. Lancer `train_and_evaluate_f1.py`

### Visualisation Grad-CAM

```bash
jupyter notebook notebooks/gradcam_visualization.ipynb
```

---

## Structure du projet

```text
SEP25_BMLE_Covid19/
├── src/features/              # Code source principal
│   ├── train_and_evaluate_f1.py    # Script d'entraînement (F1-score)
│   ├── training_config.py          # Configuration centralisée
│   ├── model_utils.py               # Utilitaires modèles
│   ├── training_utils.py           # Utilitaires entraînement
│   └── data_loader_covid.py         # Chargement des données
├── notebooks/                 # Notebooks Jupyter
│   ├── gradcam_visualization.ipynb  # Interprétabilité
│   ├── test_resnet50.ipynb          # Tests et prédictions
│   └── detection_artifacts.ipynb    # Analyse d'artefacts
├── models/                    # Modèles sauvegardés
│   └── best_model_*.pth
├── reports/                   # Rapports et visualisations
│   ├── figures/              # Graphiques (ROC, PR, Grad-CAM)
│   └── rapport_evaluation_*.txt
└── requirements.txt           # Dépendances Python
```

---

## Fonctionnalités clés

### 1. Détection d'artefacts hybride

Combinaison de deux méthodes complémentaires :

- **Analyse fréquentielle (FFT)** : Détection de textes, annotations, bordures
- **Analyse spatiale** : Détection de contrastes et contours anormaux

**Résultat** : 99.99% d'images conservées après filtrage

### 2. Optimisation de seuils

Calcul automatique de trois seuils optimaux :

- **Seuil F1** : Maximise le F1-score
- **Seuil Balanced** : Équilibre précision/rappel (par défaut)
- **Seuil Precision** : Maximise la précision (réduit faux positifs)

### 3. Gestion du déséquilibre

- Pondération automatique des classes
- WeightedRandomSampler pour échantillonnage équilibré
- Métrique F1-score pour early stopping (insensible au déséquilibre)

### 4. Interprétabilité (Grad-CAM)

Visualisation des zones d'intérêt du modèle pour :

- Validation clinique
- Détection de biais
- Aide au diagnostic

---

## Documentation complète

Pour une documentation détaillée de la méthodologie, des résultats, et des visualisations, consultez :

📖 **[README_WORKFLOW.md](README_WORKFLOW.md)** - Documentation scientifique complète

Cette documentation inclut :

- Méthodologie détaillée
- Explication des métriques et seuils
- Résultats complets avec visualisations
- Limitations et perspectives
- Références bibliographiques

---

## Sources de données

Ce projet utilise des images radiographiques thoraciques provenant des sources suivantes :

### 1. COVID-19 Radiography Database (Mendeley)

- **Source** : [Mendeley Data](https://data.mendeley.com/datasets/dvntn9yhd2/1)
- **Description** : Base de données principale contenant des images radiographiques COVID-19, Normal, Lung Opacity et Viral Pneumonia
- **Utilisation** : Dataset principal pour l'entraînement et l'évaluation

### 2. Chest X-Ray COVID19 Pneumonia (Kaggle)

- **Source** : [Kaggle Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)
- **Description** : Images de radiographies thoraciques pour COVID-19 et pneumonie
- **Utilisation** : Enrichissement du dataset avec des cas supplémentaires de COVID-19

### 3. COVID-19 Radiography Database (Kaggle)

- **Source** : [Kaggle Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database?resource=download)
- **Description** : Base de données complète de radiographies COVID-19
- **Utilisation** : Référence uniquement (non utilisée pour l'enrichissement)

**Note** : L'enrichissement du dataset provient uniquement des sources 1 et 2 (Mendeley Data et Kaggle Chest X-Ray COVID19 Pneumonia). Voir `src/features/enrich_dataset_covid.py` pour le script d'enrichissement.

---

## Limitations et considérations

### Limitations actuelles

- **Dataset** : Entraînement sur un seul dataset, pas de validation externe
- **Classes** : Déséquilibre persistant même après enrichissement
- **Validation clinique** : Nécessaire avant utilisation en contexte réel

### Considérations éthiques

- **Biais potentiels** : Le modèle peut être biaisé par la composition du dataset
- **Support décisionnel** : Outil d'aide, ne remplace pas l'expertise médicale
- **Transparence** : Visualisations Grad-CAM pour validation mais ne remplacent pas l'expertise

---

## Contribution

Ce projet est destiné à la recherche et au développement. Pour toute contribution ou question, veuillez ouvrir une issue.

---

## Références

- **ResNet** : He et al. (2016). "Deep Residual Learning for Image Recognition"
- **Grad-CAM** : Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks"
- **Transfer Learning** : Yosinski et al. (2014). "How transferable are features in deep neural networks?"

---

**Dernière mise à jour** : Décembre 2024

**Auteur** : Projet de classification COVID-19 - Deep Learning
