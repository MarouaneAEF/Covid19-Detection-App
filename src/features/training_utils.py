"""
Fonctions utilitaires pour l'entraînement

Ce module contient les fonctions pour créer les datasets, loaders,
et configurer l'optimiseur.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
from torchvision import transforms
from pathlib import Path

from pytorch_dataset_covid import ArtifactFreeDatasetCovid
from model_utils import convert_to_binary_class


def get_device():
    """Détermine le device à utiliser (MPS, CUDA, ou CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def create_transforms():
    """Crée les transformations pour train et validation."""
    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    val_transform = transforms.Compose([transforms.Normalize(mean=[0.5], std=[0.5])])
    return train_transform, val_transform


def create_datasets(
    dataset_root,
    classes,
    threshold,
    sensitivity,
    use_mask,
    train_transform,
    val_transform,
    val_split=0.2,
    random_seed=42,
):
    """
    Crée les datasets d'entraînement et de validation en divisant le dataset complet.

    Args:
        dataset_root: Chemin vers le dataset
        classes: Liste des classes
        threshold: Seuil pour la détection d'artefacts
        sensitivity: Sensibilité de la détection
        use_mask: Utiliser les masques
        train_transform: Transformations pour l'entraînement
        val_transform: Transformations pour la validation
        val_split: Proportion du dataset pour la validation (0.0 à 1.0)
        random_seed: Seed pour la reproductibilité du split

    Returns:
        train_dataset, val_dataset, full_dataset: Datasets d'entraînement, validation et complet
    """
    print("Création des datasets...")
    # Créer le dataset complet (sans transformation pour calculer les poids)
    full_dataset = ArtifactFreeDatasetCovid(
        dataset_root=dataset_root,
        classes=classes,
        threshold=threshold,
        sensitivity=sensitivity,
        use_mask=use_mask,
        filter_artifacts=True,
        transform=None,  # Pas de transformation pour le dataset complet
    )

    # Utiliser la méthode split() de la classe
    train_dataset, val_dataset = full_dataset.split(
        val_split=val_split,
        train_transform=train_transform,
        val_transform=val_transform,
        random_seed=random_seed,
    )

    total_size = len(full_dataset)
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    print(f"Dataset complet: {total_size} images")
    print(f"Split train/validation: {train_size} / {val_size} ({val_split*100:.1f}% validation)")
    print(f"Train dataset: {train_size} images")
    print(f"Val dataset: {val_size} images")
    return train_dataset, val_dataset, full_dataset


def get_class_mapping(binary_mode, classes):
    """Retourne le mapping des classes et les noms."""
    if binary_mode:
        class_to_idx = {"Non-COVID": 0, "COVID": 1}
        class_names = ["Non-COVID", "COVID"]
    else:
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        class_names = classes
    return class_to_idx, class_names


def calculate_class_weights(full_dataset, binary_mode, classes, device):
    """Calcule les poids de classes depuis le dataset complet."""
    print("\nCalcul des poids de classes...")
    if binary_mode:
        binary_weights = full_dataset.get_binary_class_weights()
        weight_non_covid = binary_weights["Non-COVID"]
        weight_covid = binary_weights["COVID"]
        class_weights_tensor = torch.tensor(
            [weight_non_covid, weight_covid], dtype=torch.float32
        ).to(device)
        # Afficher la distribution
        class_dist = full_dataset.get_class_distribution()
        covid_count = class_dist.get("COVID", 0)
        non_covid_count = sum([class_dist.get(cls, 0) for cls in classes if cls != "COVID"])
        total = covid_count + non_covid_count
        print(f" Distribution binaire:")
        print(f" Non-COVID: {non_covid_count} ({non_covid_count / total * 100:.2f}%)")
        print(f" COVID: {covid_count} ({covid_count / total * 100:.2f}%)")
        print(f" Poids Non-COVID: {weight_non_covid:.4f}")
        print(f" Poids COVID: {weight_covid:.4f}")
    else:
        class_weights = full_dataset.get_class_weights()
        class_weights_tensor = torch.tensor(
            [class_weights[cls] for cls in classes], dtype=torch.float32
        ).to(device)
        print(f" Poids: {dict(zip(classes, class_weights_tensor.cpu().numpy()))}")
    return class_weights_tensor


def create_sampler(train_dataset, binary_mode, classes, class_weights_tensor):
    """Crée le WeightedRandomSampler pour l'entraînement."""
    print("\nCréation du sampler...")
    from torch.utils.data import Subset
    
    # Accéder au dataset sous-jacent et aux indices
    if isinstance(train_dataset, Subset):
        full_dataset = train_dataset.dataset
        train_indices = train_dataset.indices
        train_clean_images_info = [full_dataset.clean_images_info[i] for i in train_indices]
    else:
        train_clean_images_info = train_dataset.clean_images_info

    sample_weights = []
    if binary_mode:
        weight_covid = class_weights_tensor[1].item()
        weight_non_covid = class_weights_tensor[0].item()
        for img_info in train_clean_images_info:
            original_class = img_info["class"]
            binary_class = convert_to_binary_class(original_class)
            sample_weights.append(weight_covid if binary_class == "COVID" else weight_non_covid)
    else:
        class_weights = {cls: class_weights_tensor[i].item() for i, cls in enumerate(classes)}
        for img_info in train_clean_images_info:
            sample_weights.append(class_weights[img_info["class"]])
    weighted_sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(train_dataset),
        replacement=True,
    )
    return weighted_sampler


def create_dataloaders(
    train_dataset, val_dataset, batch_size, sampler, num_workers, pin_memory=False
):
    """Crée les DataLoaders pour train et validation."""
    print("\nCréation des DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,  # Doit être False quand on utilise un sampler
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    return train_loader, val_loader


def setup_optimizer_and_loss(model, learning_rate, weight_decay, class_weights_tensor):
    """Configure l'optimiseur, la loss et le scheduler."""
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    print("Loss, Optimizer et Scheduler configurés")
    return criterion, optimizer, scheduler


def create_empty_history(binary_mode=False):
    """Crée un historique vide pour la visualisation."""
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_weighted": [],
    }
    if binary_mode:
        history["val_auc_roc"] = []
        history["val_auc_pr"] = []
        history["val_specificity"] = []
    return history
