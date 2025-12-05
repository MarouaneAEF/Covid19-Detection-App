"""
Configuration pour l'entraînement du modèle COVID

Modifiez les paramètres ici selon vos besoins.
"""

from pathlib import Path

# ============================================================================
# CONFIGURATION GÉNÉRALE
# ============================================================================

# Mode d'entraînement
TRAIN_MODEL = True  # Mettre à False pour charger un modèle existant
MODEL_CHECKPOINT_PATH = 'best_model_resnet50-marouane.pth'  # Chemin vers le modèle sauvegardé

# ============================================================================
# CONFIGURATION DU MODÈLE
# ============================================================================

BINARY_CLASSIFICATION = True
MODEL_CHOICE = 'resnet50'  # Options: 'resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'efficientnet', 'simple'
TRANSFER_MODE = 'fine_tuning' #'selective'  # ou 'feature_extraction', 
NUM_CLASSES = 2 if BINARY_CLASSIFICATION else 4

# ============================================================================
# CONFIGURATION D'ENTRAÎNEMENT
# ============================================================================

NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # Early stopping patience

# ============================================================================
# CONFIGURATION DU DATASET
# ============================================================================

DATASET_ROOT = Path('/Users/marouane/projet_covid/COVID-19_Radiography_Dataset')
CLASSES = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
THRESHOLD = 0.2  # Seuil de confiance FFT (0-1) - utilisé par la détection hybride
# NOTE: La détection utilise maintenant la méthode HYBRIDE (FFT + spatial)
# qui combine la détection fréquentielle (FFT) et spatiale (contraste/contours)
# Si une des deux détecte des artefacts, l'image est filtrée
SENSITIVITY = 'medium'  # 'low' (strict), 'medium' (équilibré), 'high' (permissif, détecte plus)
USE_MASK = True
NUM_WORKERS = 4  # 0 pour Mac avec MPS
VAL_SPLIT = 0.2  # Proportion du dataset pour la validation (20%)
RANDOM_SEED = 42  # Seed pour la reproductibilité du split train/validation
