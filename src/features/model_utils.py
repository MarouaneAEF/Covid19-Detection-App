"""
Module de modélisation pour la classification COVID

Ce module contient toutes les fonctions nécessaires pour :
- Créer des modèles (ResNet18, EfficientNet, SimpleCNN)
- Entraîner les modèles
- Évaluer les modèles
- Visualiser les résultats

Usage:
    from model_utils import create_model, train_model, evaluate_model
    
    # Créer un modèle
    model = create_model('resnet18', num_classes=2, device=device)
    
    # Entraîner
    history = train_model(model, train_loader, val_loader, ...)
    
    # Évaluer
    results = evaluate_model(model, val_loader, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import numpy as np
import time


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================


def convert_to_binary_class(class_name):
    """
    Convertit une classe originale en classe binaire.

    Paramètres:
    -----------
    class_name : str
        Nom de la classe originale ('COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia')

    Retourne:
    --------
    str
        'COVID' si la classe est COVID, 'Non-COVID' sinon
    """
    if class_name == 'COVID':
        return 'COVID'
    else:
        return 'Non-COVID'


# ============================================================================
# CRÉATION DE MODÈLES
# ============================================================================


def create_resnet18_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle ResNet18 pré-entraîné adapté pour la classification.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de couches à geler (ex: ['conv1', 'bn1', 'layer1'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module
        Modèle ResNet18 adapté
    """
    # Charger ResNet18 pré-entraîné
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Adapter la première couche pour accepter 1 canal (grayscale) au lieu de 3
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False  # 1 canal (grayscale) au lieu de 3
    )
    # Initialiser avec la moyenne des 3 canaux RGB
    with torch.no_grad():
        model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

    # Remplacer la dernière couche pour notre nombre de classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # STRATÉGIE DE TRANSFER LEARNING
    if freeze_backbone:
        # Option 1: Feature Extraction - Geler toutes les couches sauf la dernière
        print("  Mode: Feature Extraction (backbone gelé)")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Garder la dernière couche (fc) entraînable
                param.requires_grad = False
    elif freeze_layers is not None:
        # Option 2: Fine-tuning sélectif - Geler des couches spécifiques
        print(f"  Mode: Fine-tuning sélectif (couches gelées: {freeze_layers})")
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
    else:
        # Option 3: Fine-tuning complet - Toutes les couches entraînables
        print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
        for param in model.parameters():
            param.requires_grad = True

    return model


def create_efficientnet_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle EfficientNet-B0 pré-entraîné adapté pour la classification.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de blocs à geler (ex: ['features.0', 'features.1', 'features.2'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module ou None
        Modèle EfficientNet-B0 adapté, ou None si non disponible
    """
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        # Charger EfficientNet-B0 pré-entraîné
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Adapter la première couche pour accepter 1 canal (grayscale)
        original_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Initialiser avec la moyenne des 3 canaux RGB
        with torch.no_grad():
            model.features[0][0].weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Remplacer la dernière couche
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

        # STRATÉGIE DE TRANSFER LEARNING
        if freeze_backbone:
            print("  Mode: Feature Extraction (backbone gelé)")
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        elif freeze_layers is not None:
            print(f"  Mode: Fine-tuning sélectif (blocs gelés: {freeze_layers})")
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in freeze_layers):
                    param.requires_grad = False
        else:
            print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
            for param in model.parameters():
                param.requires_grad = True

        return model
    except ImportError:
        print("  EfficientNet non disponible dans cette version de torchvision")
        print("   Utilisez ResNet18 ou SimpleCNN à la place")
        return None


def create_resnet152_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle ResNet152 pré-entraîné adapté pour la classification.
    ResNet152 est plus profond et performant que ResNet18, idéal pour l'imagerie médicale.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de couches à geler (ex: ['conv1', 'bn1', 'layer1'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module
        Modèle ResNet152 adapté
    """
    # Charger ResNet152 pré-entraîné
    model = models.resnet152(weights='IMAGENET1K_V1')

    # Adapter la première couche pour accepter 1 canal (grayscale) au lieu de 3
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64,  # 1 canal (grayscale) au lieu de 3
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # Initialiser avec la moyenne des 3 canaux RGB
    with torch.no_grad():
        model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

    # Remplacer la dernière couche pour notre nombre de classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # STRATÉGIE DE TRANSFER LEARNING
    if freeze_backbone:
        # Option 1: Feature Extraction - Geler toutes les couches sauf la dernière
        print("  Mode: Feature Extraction (backbone gelé)")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Garder la dernière couche (fc) entraînable
                param.requires_grad = False
    elif freeze_layers is not None:
        # Option 2: Fine-tuning sélectif - Geler des couches spécifiques
        print(f"  Mode: Fine-tuning sélectif (couches gelées: {freeze_layers})")
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
    else:
        # Option 3: Fine-tuning complet - Toutes les couches entraînables
        print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
        for param in model.parameters():
            param.requires_grad = True

    return model


def create_resnet34_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle ResNet34 pré-entraîné adapté pour la classification.
    ResNet34 est un excellent compromis entre performance et charge d'entraînement.
    Plus performant que ResNet18 tout en restant beaucoup plus léger que ResNet152.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de couches à geler (ex: ['conv1', 'bn1', 'layer1'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module
        Modèle ResNet34 adapté
    """
    # Charger ResNet34 pré-entraîné
    model = models.resnet34(weights='IMAGENET1K_V1')

    # Adapter la première couche pour accepter 1 canal (grayscale) au lieu de 3
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64,  # 1 canal (grayscale) au lieu de 3
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # Initialiser avec la moyenne des 3 canaux RGB
    with torch.no_grad():
        model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

    # Remplacer la dernière couche pour notre nombre de classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # STRATÉGIE DE TRANSFER LEARNING
    if freeze_backbone:
        # Option 1: Feature Extraction - Geler toutes les couches sauf la dernière
        print("  Mode: Feature Extraction (backbone gelé)")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Garder la dernière couche (fc) entraînable
                param.requires_grad = False
    elif freeze_layers is not None:
        # Option 2: Fine-tuning sélectif - Geler des couches spécifiques
        print(f"  Mode: Fine-tuning sélectif (couches gelées: {freeze_layers})")
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
    else:
        # Option 3: Fine-tuning complet - Toutes les couches entraînables
        print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
        for param in model.parameters():
            param.requires_grad = True

    return model


def create_resnet50_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle ResNet50 pré-entraîné adapté pour la classification.
    ResNet50 offre un excellent équilibre performance/charge.
    Plus performant que ResNet34 tout en restant beaucoup plus léger que ResNet152.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de couches à geler (ex: ['conv1', 'bn1', 'layer1'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module
        Modèle ResNet50 adapté
    """
    # Charger ResNet50 pré-entraîné
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Adapter la première couche pour accepter 1 canal (grayscale) au lieu de 3
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(
        1, 64,  # 1 canal (grayscale) au lieu de 3
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # Initialiser avec la moyenne des 3 canaux RGB
    with torch.no_grad():
        model.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

    # Remplacer la dernière couche pour notre nombre de classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # STRATÉGIE DE TRANSFER LEARNING
    if freeze_backbone:
        # Option 1: Feature Extraction - Geler toutes les couches sauf la dernière
        print("  Mode: Feature Extraction (backbone gelé)")
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Garder la dernière couche (fc) entraînable
                param.requires_grad = False
    elif freeze_layers is not None:
        # Option 2: Fine-tuning sélectif - Geler des couches spécifiques
        print(f"  Mode: Fine-tuning sélectif (couches gelées: {freeze_layers})")
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
    else:
        # Option 3: Fine-tuning complet - Toutes les couches entraînables
        print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
        for param in model.parameters():
            param.requires_grad = True

    return model


def create_densenet121_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle DenseNet121 pré-entraîné adapté pour la classification.
    DenseNet121 est très efficace avec seulement ~8M paramètres tout en étant performant.
    Excellent choix pour un bon compromis performance/charge.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de blocs à geler (ex: ['features.conv0', 'features.denseblock1'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module
        Modèle DenseNet121 adapté
    """
    # Charger DenseNet121 pré-entraîné
    model = models.densenet121(weights='IMAGENET1K_V1')

    # Adapter la première couche pour accepter 1 canal (grayscale) au lieu de 3
    original_conv0 = model.features.conv0
    model.features.conv0 = nn.Conv2d(
        1, 64,  # 1 canal (grayscale) au lieu de 3, 64 canaux de sortie pour DenseNet121
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # Initialiser avec la moyenne des 3 canaux RGB
    with torch.no_grad():
        model.features.conv0.weight.data = original_conv0.weight.data.mean(dim=1, keepdim=True)

    # Remplacer la dernière couche
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    # STRATÉGIE DE TRANSFER LEARNING
    if freeze_backbone:
        # Option 1: Feature Extraction - Geler toutes les couches sauf la dernière
        print("  Mode: Feature Extraction (backbone gelé)")
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Garder la dernière couche (classifier) entraînable
                param.requires_grad = False
    elif freeze_layers is not None:
        # Option 2: Fine-tuning sélectif - Geler des couches spécifiques
        print(f"  Mode: Fine-tuning sélectif (blocs gelés: {freeze_layers})")
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
    else:
        # Option 3: Fine-tuning complet - Toutes les couches entraînables
        print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
        for param in model.parameters():
            param.requires_grad = True

    return model


def create_densenet161_model(num_classes=2, freeze_backbone=False, freeze_layers=None):
    """
    Crée un modèle DenseNet161 pré-entraîné adapté pour la classification.
    DenseNet161 est très performant pour la classification d'images et particulièrement
    adapté à l'imagerie médicale grâce à sa capacité à capturer des caractéristiques fines.
    Le modèle est adapté pour les images grayscale (1 canal).

    Paramètres:
    -----------
    num_classes : int
        Nombre de classes de sortie
    freeze_backbone : bool
        Si True, gèle toutes les couches sauf la dernière (feature extraction)
        Si False, toutes les couches sont entraînables (fine-tuning complet)
    freeze_layers : list ou None
        Liste des noms de blocs à geler (ex: ['features.conv0', 'features.denseblock1'])
        Si None, utilise freeze_backbone pour décider

    Retourne:
    --------
    torch.nn.Module
        Modèle DenseNet161 adapté
    """
    # Charger DenseNet161 pré-entraîné
    model = models.densenet161(weights='IMAGENET1K_V1')

    # Adapter la première couche pour accepter 1 canal (grayscale) au lieu de 3
    original_conv0 = model.features.conv0
    model.features.conv0 = nn.Conv2d(
        1, 96,  # 1 canal (grayscale) au lieu de 3, 96 canaux de sortie pour DenseNet161
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # Initialiser avec la moyenne des 3 canaux RGB
    with torch.no_grad():
        model.features.conv0.weight.data = original_conv0.weight.data.mean(dim=1, keepdim=True)

    # Remplacer la dernière couche
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    # STRATÉGIE DE TRANSFER LEARNING
    if freeze_backbone:
        # Option 1: Feature Extraction - Geler toutes les couches sauf la dernière
        print("  Mode: Feature Extraction (backbone gelé)")
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Garder la dernière couche (classifier) entraînable
                param.requires_grad = False
    elif freeze_layers is not None:
        # Option 2: Fine-tuning sélectif - Geler des couches spécifiques
        print(f"  Mode: Fine-tuning sélectif (blocs gelés: {freeze_layers})")
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layers):
                param.requires_grad = False
    else:
        # Option 3: Fine-tuning complet - Toutes les couches entraînables
        print("  Mode: Fine-tuning complet (toutes les couches entraînables)")
        for param in model.parameters():
            param.requires_grad = True

    return model


class SimpleCNN(nn.Module):
    """
    CNN simple personnalisé pour la classification.
    Léger et rapide à entraîner.
    """

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        # Bloc 1: Conv + Pool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Bloc 2: Conv + Pool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bloc 3: Conv + Pool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bloc 4: Conv + Pool
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Calcul de la taille après les convolutions
        # 299 -> 149 -> 74 -> 37 -> 18
        self.fc1 = nn.Linear(256 * 18 * 18, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Bloc 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Bloc 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # Bloc 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # Bloc 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


def create_model(model_choice, num_classes=2, device=None, transfer_mode='fine_tuning'):
    """
    Fonction principale pour créer un modèle selon le choix.

    Paramètres:
    -----------
    model_choice : str
        'resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'efficientnet', ou 'simple'
    num_classes : int
        Nombre de classes
    device : torch.device
        Device sur lequel placer le modèle
    transfer_mode : str
        'feature_extraction', 'selective', ou 'fine_tuning'

    Retourne:
    --------
    torch.nn.Module
        Modèle créé et déplacé sur le device
    """
    if model_choice == 'resnet18':
        if transfer_mode == 'feature_extraction':
            model = create_resnet18_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_resnet18_model(num_classes, freeze_layers=['conv1', 'bn1', 'layer1'])
        else:  # fine_tuning
            model = create_resnet18_model(num_classes, freeze_backbone=False)

    elif model_choice == 'resnet34':
        if transfer_mode == 'feature_extraction':
            model = create_resnet34_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_resnet34_model(num_classes, freeze_layers=['conv1', 'bn1', 'layer1'])
        else:  # fine_tuning
            model = create_resnet34_model(num_classes, freeze_backbone=False)

    elif model_choice == 'resnet50':
        if transfer_mode == 'feature_extraction':
            model = create_resnet50_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_resnet50_model(num_classes, freeze_layers=['conv1', 'bn1', 'layer1'])
        else:  # fine_tuning
            model = create_resnet50_model(num_classes, freeze_backbone=False)

    elif model_choice == 'resnet152':
        if transfer_mode == 'feature_extraction':
            model = create_resnet152_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_resnet152_model(num_classes, freeze_layers=['conv1', 'bn1', 'layer1'])
        else:  # fine_tuning
            model = create_resnet152_model(num_classes, freeze_backbone=False)

    elif model_choice == 'densenet121':
        if transfer_mode == 'feature_extraction':
            model = create_densenet121_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_densenet121_model(num_classes, freeze_layers=['features.conv0', 'features.denseblock1'])
        else:  # fine_tuning
            model = create_densenet121_model(num_classes, freeze_backbone=False)

    elif model_choice == 'densenet161':
        if transfer_mode == 'feature_extraction':
            model = create_densenet161_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_densenet161_model(num_classes, freeze_layers=['features.conv0', 'features.denseblock1'])
        else:  # fine_tuning
            model = create_densenet161_model(num_classes, freeze_backbone=False)

    elif model_choice == 'efficientnet':
        if transfer_mode == 'feature_extraction':
            model = create_efficientnet_model(num_classes, freeze_backbone=True)
        elif transfer_mode == 'selective':
            model = create_efficientnet_model(
                num_classes, freeze_layers=['features.0', 'features.1', 'features.2']
            )
        else:  # fine_tuning
            model = create_efficientnet_model(num_classes, freeze_backbone=False)

        if model is None:
            print("  EfficientNet non disponible, utilisation de ResNet18")
            model = create_resnet18_model(num_classes, freeze_backbone=False)
            model_choice = 'resnet18'

    elif model_choice == 'simple':
        model = SimpleCNN(num_classes)

    else:
        raise ValueError(
            f"Modèle '{model_choice}' non reconnu. Options: 'resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'efficientnet', 'simple'"
        )

    if device is not None:
        model = model.to(device)

    # Afficher les statistiques
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n Modèle {model_choice} créé")
    print(f"   Paramètres totaux: {total_params:,}")
    print(
        f"   Paramètres entraînables: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)"
    )
    print(f"   Paramètres gelés: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")

    return model


def load_model(
    model_choice, num_classes, checkpoint_path, device=None, transfer_mode='fine_tuning'
):
    """
    Charge un modèle sauvegardé depuis un checkpoint.

    Paramètres:
    -----------
    model_choice : str
        'resnet18', 'resnet34', 'resnet50', 'resnet152', 'densenet121', 'densenet161', 'efficientnet', ou 'simple'
    num_classes : int
        Nombre de classes (doit correspondre au modèle sauvegardé)
    checkpoint_path : str
        Chemin vers le fichier de checkpoint (.pth)
    device : torch.device
        Device sur lequel placer le modèle
    transfer_mode : str
        'feature_extraction', 'selective', ou 'fine_tuning'
        (doit correspondre au modèle sauvegardé)

    Retourne:
    --------
    torch.nn.Module
        Modèle chargé avec les poids sauvegardés

    Exemple:
    --------
    >>> model = load_model('efficientnet', num_classes=2, checkpoint_path='best_model.pth', device=device)
    >>> model.eval()
    """
    import os

    # Vérifier que le fichier existe
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Le fichier de checkpoint '{checkpoint_path}' n'existe pas.")

    # Créer le modèle avec la même architecture
    model = create_model(
        model_choice=model_choice,
        num_classes=num_classes,
        device=None,  # On ne déplace pas encore sur le device
        transfer_mode=transfer_mode,
    )

    # Charger les poids sauvegardés
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Gérer différents formats de checkpoint
        if isinstance(checkpoint, dict):
            # Format 1: Checkpoint complet avec 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(
                    f" Format détecté: checkpoint complet (epoch {checkpoint.get('epoch', 'N/A')})"
                )
            # Format 2: Checkpoint avec 'state_dict'
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f" Format détecté: checkpoint avec state_dict")
            # Format 3: Dictionnaire direct (state_dict lui-même)
            else:
                # Vérifier si c'est un state_dict (toutes les clés sont des noms de paramètres)
                # Si ça contient des clés comme 'epoch', 'optimizer', etc., ce n'est pas un state_dict
                if any(key in checkpoint for key in ['epoch', 'optimizer', 'history', 'val_acc']):
                    raise ValueError(
                        f"Format de checkpoint non reconnu. Clés trouvées: {list(checkpoint.keys())[:5]}...\n"
                        f"Le checkpoint doit contenir 'model_state_dict' ou être directement un state_dict."
                    )
                state_dict = checkpoint
                print(f" Format détecté: state_dict direct")
        else:
            # Format 4: State_dict direct (pas un dict)
            state_dict = checkpoint
            print(f" Format détecté: state_dict direct")

        # Nettoyer les clés si elles commencent par 'model.'
        if any(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            print(f"   Clés nettoyées (préfixe 'model.' supprimé)")

        # Charger les poids avec strict=False pour ignorer les clés manquantes
        # (utile si certaines couches ont été gelées différemment)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            print(f"  Clés manquantes (ignorées): {len(missing_keys)} clés")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"   - {key}")
            else:
                print(f"   Premières clés: {missing_keys[:5]}...")

        if unexpected_keys:
            print(f"  Clés inattendues (ignorées): {len(unexpected_keys)} clés")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"   - {key}")
            else:
                print(f"   Premières clés: {unexpected_keys[:5]}...")

        print(f" Modèle chargé depuis '{checkpoint_path}'")

    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du checkpoint: {e}")

    # Déplacer sur le device si spécifié
    if device is not None:
        model = model.to(device)
        print(f" Modèle déplacé sur {device}")

    # Afficher les informations
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Paramètres totaux: {total_params:,}")
    print(f"   Paramètres entraînables: {trainable_params:,}")

    return model


# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================================


def train_epoch(model, dataloader, criterion, optimizer, device, class_to_idx, binary_mode=False):
    """
    Entraîne le modèle pour une époque.

    Paramètres:
    -----------
    model : torch.nn.Module
        Modèle à entraîner
    dataloader : torch.utils.data.DataLoader
        DataLoader pour les données d'entraînement
    criterion : torch.nn.Module
        Fonction de loss
    optimizer : torch.optim.Optimizer
        Optimiseur
    device : torch.device
        Device sur lequel effectuer les calculs
    class_to_idx : dict
        Mapping classe -> index
    binary_mode : bool
        Si True, convertit les classes en binaire (COVID vs Non-COVID)

    Retourne:
    --------
    tuple
        (epoch_loss, epoch_acc)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, masks, metadata_tuples) in enumerate(dataloader):
        # Extraire les classes
        classes, _, _, _ = metadata_tuples

        # Convertir les classes en indices (avec conversion binaire si nécessaire)
        if binary_mode:
            binary_classes = [convert_to_binary_class(cls) for cls in classes]
            labels = torch.tensor(
                [class_to_idx[cls] for cls in binary_classes], dtype=torch.long
            ).to(device)
        else:
            labels = torch.tensor([class_to_idx[cls] for cls in classes], dtype=torch.long).to(
                device
            )

        # Déplacer les images sur le device
        images = images.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistiques
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Afficher la progression
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device, class_to_idx, binary_mode=False):
    """
    Valide le modèle sur le dataset de validation.

    Paramètres:
    -----------
    model : torch.nn.Module
        Modèle à valider
    dataloader : torch.utils.data.DataLoader
        DataLoader pour les données de validation
    criterion : torch.nn.Module
        Fonction de loss
    device : torch.device
        Device sur lequel effectuer les calculs
    class_to_idx : dict
        Mapping classe -> index
    binary_mode : bool
        Si True, convertit les classes en binaire (COVID vs Non-COVID)

    Retourne:
    --------
    tuple
        (epoch_loss, epoch_acc, all_preds, all_labels)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks, metadata_tuples in dataloader:
            # Extraire les classes
            classes, _, _, _ = metadata_tuples

            # Convertir les classes en indices (avec conversion binaire si nécessaire)
            if binary_mode:
                binary_classes = [convert_to_binary_class(cls) for cls in classes]
                labels = torch.tensor(
                    [class_to_idx[cls] for cls in binary_classes], dtype=torch.long
                ).to(device)
            else:
                labels = torch.tensor([class_to_idx[cls] for cls in classes], dtype=torch.long).to(
                    device
                )

            # Déplacer les images sur le device
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Statistiques
            running_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    class_to_idx,
    num_epochs=10,
    binary_mode=False,
    patience=5,
    save_path='best_model.pth',
):
    """
    Fonction principale pour entraîner un modèle.

    Paramètres:
    -----------
    model : torch.nn.Module
        Modèle à entraîner
    train_loader : torch.utils.data.DataLoader
        DataLoader pour l'entraînement
    val_loader : torch.utils.data.DataLoader
        DataLoader pour la validation
    criterion : torch.nn.Module
        Fonction de loss
    optimizer : torch.optim.Optimizer
        Optimiseur
    scheduler : torch.optim.lr_scheduler
        Scheduler pour le learning rate
    device : torch.device
        Device sur lequel effectuer les calculs
    class_to_idx : dict
        Mapping classe -> index
    num_epochs : int
        Nombre d'époques
    binary_mode : bool
        Si True, convertit les classes en binaire
    patience : int
        Patience pour l'early stopping
    save_path : str
        Chemin pour sauvegarder le meilleur modèle

    Retourne:
    --------
    dict
        Historique de l'entraînement avec 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    best_val_acc = 0.0
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f" Démarrage de l'entraînement")
    print(f"   Époques: {num_epochs}")
    print(f"   Device: {device}")
    print(f"   Classification: {'Binaire' if binary_mode else 'Multi-classes'}")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n Époque {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # Entraînement
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, class_to_idx, binary_mode=binary_mode
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device, class_to_idx, binary_mode=binary_mode
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Mettre à jour le scheduler
        scheduler.step(val_loss)

        # Afficher les résultats
        print(f"\n Résultats Époque {epoch + 1}:")
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            print(f"    Meilleure validation accuracy: {best_val_acc:.4f}")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping après {epoch + 1} époques")
                break

    elapsed_time = time.time() - start_time
    print(f"\n Entraînement terminé en {elapsed_time/60:.2f} minutes")
    print(f"   Meilleure validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    return history


# ============================================================================
# FONCTIONS D'ÉVALUATION
# ============================================================================


def evaluate_model(
    model, dataloader, device, class_to_idx, binary_mode=False, class_names=None, threshold=0.5
):
    """
    Évalue le modèle sur un dataset complet.

    Paramètres:
    -----------
    model : torch.nn.Module
        Modèle à évaluer
    dataloader : torch.utils.data.DataLoader
        DataLoader pour les données
    device : torch.device
        Device sur lequel effectuer les calculs
    class_to_idx : dict
        Mapping classe -> index
    binary_mode : bool
        Si True, convertit les classes en binaire
    class_names : list
        Noms des classes pour l'affichage
    threshold : float
        Seuil de décision pour la classification binaire

    Retourne:
    --------
    dict
        Dictionnaire contenant les prédictions, labels, probabilités et métriques
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, masks, metadata_tuples in dataloader:
            classes, _, _, _ = metadata_tuples

            if binary_mode:
                binary_classes = [convert_to_binary_class(cls) for cls in classes]
                labels = torch.tensor(
                    [class_to_idx[cls] for cls in binary_classes], dtype=torch.long
                ).to(device)
            else:
                labels = torch.tensor([class_to_idx[cls] for cls in classes], dtype=torch.long).to(
                    device
                )

            images = images.to(device)
            outputs = model(images)

            # Obtenir les probabilités
            probs = F.softmax(outputs, dim=1)

            if binary_mode:
                # Utiliser le seuil pour la classification binaire
                covid_probs = probs[:, 1].cpu().numpy()
                preds = (covid_probs >= threshold).astype(int)
                all_probs.extend(covid_probs)
            else:
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculer les métriques
    results = {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs) if binary_mode else None,
    }

    # Métriques de classification
    if binary_mode:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        results['support'] = support
        results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)
    else:
        results['classification_report'] = classification_report(
            all_labels, all_preds, target_names=class_names, output_dict=True
        )
        results['confusion_matrix'] = confusion_matrix(all_labels, all_preds)

    return results


def find_optimal_threshold(results, metric='f1', target_precision=None):
    """
    Trouve le seuil optimal basé sur différentes métriques.

    Paramètres:
    -----------
    results : dict
        Résultats de evaluate_model (doit contenir 'probabilities' et 'labels')
    metric : str
        Métrique à optimiser : 'f1', 'balanced' (precision ≈ recall), ou 'precision'
    target_precision : float, optional
        Si metric='precision', seuil de précision cible (ex: 0.7)

    Retourne:
    --------
    dict
        Dictionnaire avec 'threshold', 'f1', 'precision', 'recall' pour le seuil optimal
    """
    if results.get('probabilities') is None:
        raise ValueError(
            "Les probabilités ne sont pas disponibles dans results. Utilisez evaluate_model avec binary_mode=True."
        )

    all_probs = results['probabilities']
    all_labels = results['labels']

    # Calculer la courbe Precision-Recall
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    # Calculer F1-score pour chaque seuil
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = f1_scores[:-1]  # Enlever le dernier élément

    if metric == 'f1':
        # Seuil qui maximise F1-score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        optimal_f1 = f1_scores[best_idx]
        optimal_precision = precision[best_idx]
        optimal_recall = recall[best_idx]

    elif metric == 'balanced':
        # Seuil où precision ≈ recall (équilibre)
        diff = np.abs(precision[:-1] - recall[:-1])
        best_idx = np.argmin(diff)
        optimal_threshold = thresholds[best_idx]
        optimal_f1 = f1_scores[best_idx]
        optimal_precision = precision[best_idx]
        optimal_recall = recall[best_idx]

    elif metric == 'precision' and target_precision is not None:
        # Seuil pour atteindre une précision cible
        precision_above_target = precision[:-1] >= target_precision
        if np.any(precision_above_target):
            # Prendre le seuil le plus bas qui atteint la précision cible
            valid_indices = np.where(precision_above_target)[0]
            best_idx = valid_indices[0]  # Premier seuil qui atteint la précision
            optimal_threshold = thresholds[best_idx]
            optimal_f1 = f1_scores[best_idx]
            optimal_precision = precision[best_idx]
            optimal_recall = recall[best_idx]
        else:
            # Aucun seuil n'atteint la précision cible, prendre le meilleur F1
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx]
            optimal_f1 = f1_scores[best_idx]
            optimal_precision = precision[best_idx]
            optimal_recall = recall[best_idx]
            print(f"  Aucun seuil n'atteint la précision cible {target_precision:.2f}")
            print(f"   Utilisation du seuil avec le meilleur F1-score")
    else:
        # Par défaut, utiliser F1-score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        optimal_f1 = f1_scores[best_idx]
        optimal_precision = precision[best_idx]
        optimal_recall = recall[best_idx]

    return {
        'threshold': optimal_threshold,
        'f1': optimal_f1,
        'precision': optimal_precision,
        'recall': optimal_recall,
        'metric_used': metric,
    }


def analyze_thresholds(results, default_threshold=0.5):
    """
    Analyse les différents seuils de décision et affiche les recommandations.

    Paramètres:
    -----------
    results : dict
        Résultats de evaluate_model (doit contenir 'probabilities' et 'labels')
    default_threshold : float
        Seuil par défaut utilisé (pour comparaison)

    Retourne:
    --------
    dict
        Dictionnaire avec les seuils optimaux pour différentes métriques
    """
    if results.get('probabilities') is None:
        raise ValueError("Les probabilités ne sont pas disponibles dans results.")

    all_probs = results['probabilities']
    all_labels = results['labels']

    # Calculer la courbe Precision-Recall
    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

    # Calculer F1-score pour chaque seuil
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    f1_scores = f1_scores[:-1]

    # 1. Seuil optimal pour F1-score maximum
    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    best_f1_precision = precision[best_f1_idx]
    best_f1_recall = recall[best_f1_idx]

    # 2. Seuil équilibré (precision ≈ recall)
    diff = np.abs(precision[:-1] - recall[:-1])
    balanced_idx = np.argmin(diff)
    balanced_threshold = thresholds[balanced_idx]
    balanced_precision = precision[balanced_idx]
    balanced_recall = recall[balanced_idx]

    # 3. Seuil pour précision >= 0.7
    precision_above_07 = precision[:-1] >= 0.7
    if np.any(precision_above_07):
        target_idx = np.where(precision_above_07)[0][0]
        target_threshold = thresholds[target_idx]
        target_precision = precision[target_idx]
        target_recall = recall[target_idx]
    else:
        target_threshold = None
        target_precision = None
        target_recall = None

    # 4. Métriques avec seuil par défaut
    default_preds = (all_probs >= default_threshold).astype(int)
    default_precision, default_recall, default_f1, _ = precision_recall_fscore_support(
        all_labels, default_preds, average=None, zero_division=0
    )

    # Afficher l'analyse
    print("=" * 70)
    print("ANALYSE DES SEUILS DE DÉCISION")
    print("=" * 70)
    print(f"\n1. Seuil optimal pour F1-score maximum:")
    print(f"   Seuil: {best_f1_threshold:.4f}")
    print(f"   F1-score: {best_f1:.4f}")
    print(f"   Precision COVID: {best_f1_precision:.4f}")
    print(f"   Recall COVID: {best_f1_recall:.4f}")

    print(f"\n2. Seuil équilibré (precision ≈ recall):")
    print(f"   Seuil: {balanced_threshold:.4f}")
    print(f"   Precision COVID: {balanced_precision:.4f}")
    print(f"   Recall COVID: {balanced_recall:.4f}")

    if target_threshold is not None:
        print(f"\n3. Seuil pour précision COVID >= 0.7:")
        print(f"   Seuil: {target_threshold:.4f}")
        print(f"   Precision COVID: {target_precision:.4f}")
        print(f"   Recall COVID: {target_recall:.4f}")

    print(f"\n4. Seuil actuel (par défaut): {default_threshold:.4f}")
    print(f"   Precision COVID: {default_precision[1]:.4f}")
    print(f"   Recall COVID: {default_recall[1]:.4f}")
    print(f"   F1-score: {default_f1[1]:.4f}")

    print("\n" + "=" * 70)
    print("RECOMMANDATION")
    print("=" * 70)
    if target_threshold is not None:
        print(f"Pour réduire les faux positifs (améliorer la précision COVID):")
        print(f"   Utilisez le seuil: {target_threshold:.4f}")
        print(f"   Cela donnera une précision COVID de {target_precision:.4f}")
        print(f"   au prix d'une réduction du recall à {target_recall:.4f}")
    else:
        print(f"Pour un bon équilibre:")
        print(f"   Utilisez le seuil: {balanced_threshold:.4f}")
        print(f"   Precision: {balanced_precision:.4f}, Recall: {balanced_recall:.4f}")

    return {
        'best_f1': {
            'threshold': best_f1_threshold,
            'f1': best_f1,
            'precision': best_f1_precision,
            'recall': best_f1_recall,
        },
        'balanced': {
            'threshold': balanced_threshold,
            'precision': balanced_precision,
            'recall': balanced_recall,
        },
        'target_precision': (
            {'threshold': target_threshold, 'precision': target_precision, 'recall': target_recall}
            if target_threshold is not None
            else None
        ),
        'default': {
            'threshold': default_threshold,
            'precision': default_precision[1],
            'recall': default_recall[1],
            'f1': default_f1[1],
        },
    }


def print_classification_report(results, binary_mode=False, class_names=None, threshold=None):
    """
    Affiche un rapport de classification formaté.

    Paramètres:
    -----------
    results : dict
        Résultats de evaluate_model (doit contenir 'predictions', 'labels', 'probabilities' si binary_mode)
    binary_mode : bool
        Si True, affiche un rapport binaire
    class_names : list
        Noms des classes
    threshold : float, optional
        Seuil de décision pour recalculer les prédictions (uniquement pour binary_mode).
        Si None, utilise les prédictions déjà calculées dans results.
        Si spécifié et que 'probabilities' est disponible, recalcule les prédictions avec ce seuil.
    """
    # Si un seuil personnalisé est fourni et qu'on a les probabilités, recalculer les prédictions
    if binary_mode and threshold is not None and results.get('probabilities') is not None:
        # Recalculer les prédictions avec le nouveau seuil
        all_probs = results['probabilities']
        all_labels = results['labels']
        new_preds = (all_probs >= threshold).astype(int)

        # Recalculer les métriques avec les nouvelles prédictions
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, new_preds, average=None, zero_division=0
        )
        cm = confusion_matrix(all_labels, new_preds)

        # Mettre à jour results temporairement pour l'affichage
        results = results.copy()
        results['predictions'] = new_preds
        results['precision'] = precision
        results['recall'] = recall
        results['f1'] = f1
        results['support'] = support
        results['confusion_matrix'] = cm
        used_threshold = threshold
    else:
        # Utiliser les résultats tels quels
        used_threshold = threshold if threshold is not None else 0.5

    print("=" * 70)
    print("RAPPORT DE CLASSIFICATION")
    if binary_mode and results.get('probabilities') is not None:
        print(f"Seuil de décision: {used_threshold:.4f}")
    print("=" * 70)

    if binary_mode:
        print(f"\n Classification Binaire - COVID vs Non-COVID")
        print(
            f"{'Classe':<15s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s} {'Support':<12s}"
        )
        print("-" * 60)
        for i, cls in enumerate(['Non-COVID', 'COVID']):
            print(
                f"{cls:<15s} {results['precision'][i]:<12.4f} {results['recall'][i]:<12.4f} "
                f"{results['f1'][i]:<12.4f} {results['support'][i]:<12d}"
            )

        # Matrice de confusion
        cm = results['confusion_matrix']
        print(f"\n Matrice de Confusion:")
        print("=" * 60)
        print(f"{'':15s} {'Prédit Non-COVID':<20s} {'Prédit COVID':<20s}")
        print("-" * 60)
        print(f"{'Réel Non-COVID':<15s} {cm[0, 0]:<20d} {cm[0, 1]:<20d}")
        print(f"{'Réel COVID':<15s} {cm[1, 0]:<20d} {cm[1, 1]:<20d}")

        # Interprétation
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        print(f"\n Interprétation:")
        print(f"   Vrais Négatifs (TN): {tn}")
        print(f"   Faux Positifs (FP): {fp}")
        print(f"   Faux Négatifs (FN): {fn}")
        print(f"   Vrais Positifs (TP): {tp}")

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print(f"\n Métriques médicales:")
        print(f"   Sensibilité (Recall COVID): {sensitivity:.4f}")
        print(f"   Spécificité (Recall Non-COVID): {specificity:.4f}")

        # Afficher le seuil utilisé si les probabilités sont disponibles
        if results.get('probabilities') is not None:
            print(f"\n Seuil de décision utilisé: {used_threshold:.4f}")
    else:
        print(
            classification_report(
                results['labels'], results['predictions'], target_names=class_names, digits=4
            )
        )

        cm = results['confusion_matrix']
        print(f"\n Matrice de Confusion:")
        print("=" * 60)
        print(f"{'':15s}", end="")
        for cls in class_names:
            print(f"{cls[:12]:>12s}", end="")
        print()
        for i, cls in enumerate(class_names):
            print(f"{cls[:15]:15s}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:12d}", end="")
            print()


# ============================================================================
# FONCTIONS DE VISUALISATION
# ============================================================================


def plot_training_history(history, save_path=None):
    """
    Visualise l'historique d'entraînement.

    Paramètres:
    -----------
    history : dict
        Historique avec 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    save_path : str
        Chemin pour sauvegarder la figure (optionnel)
    """
    # Vérifier si l'historique est vide (modèle chargé sans entraînement)
    if not history.get('train_loss') or len(history['train_loss']) == 0:
        print("  Historique d'entraînement vide (modèle chargé sans entraînement)")
        print("   Les graphiques d'entraînement ne peuvent pas être affichés.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Évolution de la Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Évolution de l\'Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()

    # Afficher le résumé seulement si l'historique n'est pas vide
    if len(history['train_loss']) > 0:
        print(f"\n Résumé final:")
        print(f"   Train Loss: {history['train_loss'][-1]:.4f} → {history['train_loss'][0]:.4f}")
        print(
            f"   Train Acc:  {history['train_acc'][-1]:.4f} ({history['train_acc'][-1]*100:.2f}%)"
        )
        print(f"   Val Loss:   {history['val_loss'][-1]:.4f} → {history['val_loss'][0]:.4f}")
        print(f"   Val Acc:    {history['val_acc'][-1]:.4f} ({history['val_acc'][-1]*100:.2f}%)")


def plot_precision_recall_curve(results, save_path=None):
    """
    Trace la courbe Precision-Recall.

    Paramètres:
    -----------
    results : dict
        Résultats de evaluate_model (doit contenir 'probabilities')
    save_path : str
        Chemin pour sauvegarder la figure (optionnel)
    """
    if results['probabilities'] is None:
        print("  Probabilités non disponibles")
        return

    precision, recall, thresholds = precision_recall_curve(
        results['labels'], results['probabilities']
    )

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()


def plot_roc_curve(results, save_path=None):
    """
    Trace la courbe ROC.

    Paramètres:
    -----------
    results : dict
        Résultats de evaluate_model (doit contenir 'probabilities')
    save_path : str
        Chemin pour sauvegarder la figure (optionnel)
    """
    if results['probabilities'] is None:
        print("  Probabilités non disponibles")
        return

    fpr, tpr, thresholds = roc_curve(results['labels'], results['probabilities'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Courbe ROC')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)

    plt.show()
    print(f" AUC: {roc_auc:.4f}")
