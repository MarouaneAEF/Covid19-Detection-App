"""
Dataset PyTorch pour les images avec masques appliqués et sans artefacts.

Ce module fournit une classe Dataset PyTorch qui utilise ArtifactFreeDataLoader
pour charger et filtrer les images.
"""

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional

from data_loader_covid import ArtifactFreeDataLoaderCovid


class TransformedSubset(Subset):
    """Wrapper pour appliquer des transformations à un Subset."""

    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        img, mask, metadata = super().__getitem__(idx)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, metadata


class ArtifactFreeDatasetCovid(Dataset):
    """
    Dataset PyTorch pour les images avec masques appliqués et sans artefacts.
    """

    def __init__(
        self,
        dataset_root: Path,
        classes: list,
        threshold: float = 0.3,
        sensitivity: str = 'medium',
        use_mask: bool = True,
        filter_artifacts: bool = True,
        target_size: Tuple[int, int] = (299, 299),
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialise le dataset.
        """
        self.target_size = target_size
        self.transform = transform

        # Créer le data loader
        self.loader = ArtifactFreeDataLoaderCovid(
            dataset_root=dataset_root,
            classes=classes,
            threshold=threshold,
            sensitivity=sensitivity,
            use_mask=use_mask,
            filter_artifacts=filter_artifacts,
            cache_artifacts=True,
        )

        # Précharger les métadonnées des images propres
        self.clean_images_info = self.loader._get_clean_images()

        print(f"Dataset PyTorch créé avec {len(self.clean_images_info)} images propres")

    def __len__(self) -> int:
        """Retourne le nombre d'images dans le dataset."""
        return len(self.clean_images_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Récupère un élément du dataset.
        """
        img_info = self.clean_images_info[idx]

        # Charger l'image avec
        # masque appliqué
        img_masked, mask_binary, metadata = self.loader.load_image_with_mask(
            img_info, target_size=self.target_size
        )

        # Convertir en Tensor PyTorch
        # Image: (H, W) -> (1, H, W) pour grayscale
        img_tensor = torch.from_numpy(img_masked).unsqueeze(0).float()

        # Masque: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        # Normaliser l'image entre 0 et 1
        img_tensor = img_tensor / 255.0

        # Appliquer les transformations (data augmentation)
        if self.transform is not None:
            # Pour l'image
            # Les transformations PyTorch fonctionnent avec (C, H, W)
            img_tensor = self.transform(img_tensor)

            # Pour le masque, on ne fait rien car les transformations
            # géométriques ne sont pas appliquées au masque.

        metadata_tuple = (
            metadata['class'],
            metadata['image_name'],
            metadata['image_path'],
            metadata['has_mask'],
        )

        return img_tensor, mask_tensor, metadata_tuple

    def get_statistics(self) -> dict:
        """
        Retourne les statistiques du dataset.
        """
        return self.loader.get_statistics()

    def get_class_distribution(self) -> dict:
        """
        Retourne la distribution des classes dans le dataset.
        """
        from collections import Counter

        classes = [img_info['class'] for img_info in self.clean_images_info]
        return dict(Counter(classes))

    def get_class_weights(self) -> dict:
        """
        Calcule les poids pour chaque classe pour équilibrer le dataset.
        """
        class_dist = self.get_class_distribution()
        total_samples = sum(class_dist.values())
        num_classes = len(class_dist)

        weights = {}
        for class_name, count in class_dist.items():
            if count > 0:
                weights[class_name] = total_samples / (num_classes * count)
            else:
                weights[class_name] = 0.0

        return weights

    def get_binary_class_weights(self) -> dict:
        """
        Calcule les poids pour la classification binaire (COVID vs Non-COVID).

        Retourne:
        --------
        dict : {'Non-COVID': weight_non_covid, 'COVID': weight_covid}
            Poids pour équilibrer le dataset binaire.
        """
        class_dist = self.get_class_distribution()

        # Compter les classes binaires
        covid_count = class_dist.get('COVID', 0)
        non_covid_count = sum(
            [class_dist.get(cls, 0) for cls in class_dist.keys() if cls != 'COVID']
        )

        total = covid_count + non_covid_count

        if total == 0:
            return {'Non-COVID': 1.0, 'COVID': 1.0}

        # Poids inversement proportionnels à la fréquence
        # Formule: weight = total / (num_classes * count)
        # Pour binaire: num_classes = 2
        weight_non_covid = total / (2 * non_covid_count) if non_covid_count > 0 else 1.0
        weight_covid = total / (2 * covid_count) if covid_count > 0 else 1.0

        return {'Non-COVID': weight_non_covid, 'COVID': weight_covid}

    def split(
        self,
        val_split: float = 0.2,
        train_transform: Optional[transforms.Compose] = None,
        val_transform: Optional[transforms.Compose] = None,
        random_seed: int = 42,
    ) -> Tuple[TransformedSubset, TransformedSubset]:
        """
        Divise le dataset en deux subsets séparés (train et validation).

        Utilise un seul dataset et applique des transformations différentes via des wrappers.

        Args:
            val_split: Proportion du dataset pour la validation (0.0 à 1.0)
            train_transform: Transformations pour le dataset d'entraînement
            val_transform: Transformations pour le dataset de validation
            random_seed: Seed pour la reproductibilité du split

        Returns:
            train_subset, val_subset: Deux subsets avec des indices disjoints et transformations différentes
        """
        total_size = len(self)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        # Obtenir les indices du split
        generator = torch.Generator().manual_seed(random_seed)
        train_indices, val_indices = random_split(
            range(total_size), [train_size, val_size], generator=generator
        )

        # Créer deux subsets qui pointent vers le même dataset mais avec des transformations différentes
        train_subset = Subset(self, train_indices.indices)
        val_subset = Subset(self, val_indices.indices)

        # Envelopper avec les transformations
        train_subset = TransformedSubset(train_subset, transform=train_transform)
        val_subset = TransformedSubset(val_subset, transform=val_transform)

        return train_subset, val_subset
