"""
Script pour enrichir le dataset actuel avec plus d'images COVID.

Ce script permet de:
1. Télécharger des images COVID depuis un autre dataset
2. Les copier dans la structure du dataset actuel
3. Générer des masques de poumons automatiquement
"""

from pathlib import Path
from PIL import Image
import shutil
import numpy as np
import cv2
from typing import List, Optional


def generate_lung_mask(img_array: np.ndarray) -> np.ndarray:
    """
    Génère automatiquement un masque de poumons à partir d'une image de radiographie.

    Args:
        img_array: Image en niveaux de gris (numpy array)

    Returns:
        Masque binaire (0/255) des poumons
    """
    # Normalisation
    if img_array.max() > 255:
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    # Seuillage adaptatif (Otsu) pour séparer les poumons du fond
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphologie pour nettoyer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Ouverture pour supprimer les petits bruits
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fermeture pour combler les petits trous
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Trouver les composantes connexes
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Créer un masque vide
    lung_mask = np.zeros_like(binary)

    # Filtrer les composantes par taille et position
    # Les poumons sont généralement les deux plus grandes composantes au centre
    if num_labels > 1:
        # Trier par aire (en excluant le fond)
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        areas.sort(key=lambda x: x[1], reverse=True)

        # Prendre les 2 plus grandes composantes (poumons gauche et droit)
        # ou la plus grande si elle est assez grande
        height, width = img_array.shape
        min_area = (height * width) * 0.05  # Au moins 5% de l'image
        max_area = (height * width) * 0.6   # Au plus 60% de l'image

        for idx, area in areas[:2]:  # Prendre les 2 plus grandes
            if min_area <= area <= max_area:
                # Vérifier que la composante est dans la zone centrale (pas les bords)
                x, y = centroids[idx]
                margin = 0.1
                if (margin * width <= x <= (1 - margin) * width and
                    margin * height <= y <= (1 - margin) * height):
                    lung_mask[labels == idx] = 255

    # Si aucun poumon trouvé, utiliser un masque basé sur le seuillage simple
    if np.sum(lung_mask) == 0:
        # Seuillage simple basé sur l'histogramme
        hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
        # Trouver le pic principal (poumons sont généralement sombres)
        peak_idx = np.argmax(hist[50:200]) + 50  # Éviter les valeurs extrêmes
        _, binary_simple = cv2.threshold(img_array, peak_idx - 30, 255, cv2.THRESH_BINARY_INV)
        
        # Nettoyage morphologique
        binary_simple = cv2.morphologyEx(binary_simple, cv2.MORPH_OPEN, kernel, iterations=2)
        binary_simple = cv2.morphologyEx(binary_simple, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Prendre la plus grande composante
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_simple, connectivity=8)
        if num_labels > 1:
            areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
            areas.sort(key=lambda x: x[1], reverse=True)
            if areas:
                lung_mask[labels == areas[0][0]] = 255

    return lung_mask


def enrich_covid_images(
    source_dirs: List[Path],
    target_dir: Path,
    image_extensions: List[str] = ['.png', '.jpg', '.jpeg'],
    generate_masks: bool = True,
    prefix: Optional[str] = None,
):
    """
    Enrichit le dataset avec des images COVID depuis un ou plusieurs dossiers sources.

    Args:
        source_dirs: Liste de dossiers sources contenant les images COVID à ajouter
        target_dir: Dossier cible (COVID/images/ du dataset actuel)
        image_extensions: Extensions d'images à chercher
        generate_masks: Si True, génère des masques de poumons automatiquement
        prefix: Préfixe optionnel pour les noms de fichiers (ex: 'train_', 'test_')
    """
    target_images = target_dir / 'images'
    target_masks = target_dir / 'masks'

    # Créer les dossiers si nécessaire
    target_images.mkdir(parents=True, exist_ok=True)
    if generate_masks:
        target_masks.mkdir(parents=True, exist_ok=True)

    # Compter les images existantes
    existing_count = len(list(target_images.glob('*.png')))
    print(f"Images COVID existantes: {existing_count}")

    # Trouver toutes les images dans les dossiers sources
    new_images = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"  Attention: Dossier source non trouve: {source_dir}")
            continue
        
        for ext in image_extensions:
            new_images.extend(list(source_dir.glob(f'*{ext}')))
            new_images.extend(list(source_dir.glob(f'*{ext.upper()}')))

    print(f"Images trouvees dans les sources: {len(new_images)}")

    # Copier les images
    copied = 0
    skipped = 0
    existing_names = {f.name for f in target_images.glob('*.png')}

    for img_path in new_images:
        # Générer un nom de fichier unique
        if prefix:
            new_name = f"{prefix}{img_path.stem}.png"
        else:
            new_name = f"{img_path.stem}.png"
        
        target_path = target_images / new_name

        # Si l'image existe déjà, skip
        if new_name in existing_names:
            skipped += 1
            continue

        try:
            # Charger l'image
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)

            # Sauvegarder l'image en PNG
            img.save(target_path, 'PNG')
            existing_names.add(new_name)

            # Générer un masque de poumons automatiquement
            if generate_masks:
                mask_path = target_masks / target_path.name
                if not mask_path.exists():
                    # Générer le masque de poumons
                    lung_mask = generate_lung_mask(img_array)
                    
                    # Sauvegarder le masque
                    mask_img = Image.fromarray(lung_mask, mode='L')
                    mask_img.save(mask_path, 'PNG')

            copied += 1
            if copied % 100 == 0:
                print(f"  Copie: {copied} images...")

        except Exception as e:
            print(f"  Erreur avec {img_path.name}: {e}")
            skipped += 1

    print(f"\nResultat:")
    print(f"  Images copiees: {copied}")
    print(f"  Images ignorees: {skipped}")
    print(f"  Total images COVID maintenant: {existing_count + copied}")


def main():
    """Fonction principale."""
    print("="*70)
    print("ENRICHISSEMENT DU DATASET COVID")
    print("="*70)

    # Configuration
    dataset_root = Path('/Users/marouane/projet_covid/COVID-19_Radiography_Dataset')
    covid_target = dataset_root / 'COVID'

    # Dossiers sources avec images COVID
    # Dataset 1: /Users/marouane/Downloads/Data (train/test)
    data_root = Path('/Users/marouane/Downloads/Data')
    train_covid = data_root / 'train' / 'COVID19'
    test_covid = data_root / 'test' / 'COVID19'

    # Dataset 2: Covid19-Pneumonia-Normal Chest X-Ray Images Dataset
    dataset2_root = Path('/Users/marouane/Downloads/Covid19-Pneumonia-Normal Chest X-Ray Images Dataset')
    dataset2_covid = dataset2_root / 'COVID'

    # Liste de tous les dossiers sources à traiter
    all_sources = [
        ('train/COVID19', train_covid, 'train_'),
        ('test/COVID19', test_covid, 'test_'),
        ('Covid19-Pneumonia Dataset/COVID', dataset2_covid, 'ds2_'),
    ]

    # Vérifier que les dossiers existent
    existing_sources = []
    for name, path, prefix in all_sources:
        if path.exists():
            existing_sources.append((name, path, prefix))
        else:
            print(f"  Attention: Dossier non trouve (sera ignore): {path}")

    if not existing_sources:
        print(f"\n  Aucun dossier source trouve!")
        return

    print(f"\nSources trouvees:")
    for name, path, _ in existing_sources:
        count = len(list(path.glob('*.jpg'))) + len(list(path.glob('*.jpeg'))) + len(list(path.glob('*.png')))
        print(f"  - {name}: {count} images")
    print(f"Cible: {covid_target}")

    # Enrichir depuis chaque source
    for name, source_dir, prefix in existing_sources:
        print("\n" + "-"*70)
        print(f"Extraction depuis {name}...")
        print("-"*70)
        enrich_covid_images(
            source_dirs=[source_dir],
            target_dir=covid_target,
            image_extensions=['.png', '.jpg', '.jpeg'],
            generate_masks=True,
            prefix=prefix,
        )

    # Statistiques finales
    final_count = len(list((covid_target / 'images').glob('*.png')))
    print("\n" + "="*70)
    print(f"Total images COVID apres enrichissement: {final_count}")
    print("="*70)


if __name__ == '__main__':
    main()

