"""
Module de détection d'artefacts dans les images de radiographies.

Ce module contient les fonctions pour détecter automatiquement les artefacts
dans les images en utilisant la transformée de Fourier et l'analyse d'image.
Analyse uniquement les zones masquées (poumons).
"""

import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

TARGET_SIZE = (299, 299)


def load_and_apply_mask(img_path: Path, dataset_root: Path = None) -> tuple:
    """
    Charge l'image et son masque, puis applique le masque.

    Paramètres:
    -----------
    img_path : Path
        Chemin vers l'image
    dataset_root : Path or None
        Racine du dataset (pour trouver le masque)

    Retourne:
    --------
    tuple: (img_masked, mask_binary, img_original)
        - img_masked: Image masquée (zones non masquées = 0)
        - mask_binary: Masque binaire (True = zone masquée)
        - img_original: Image originale
    """
    # Charger l'image
    with Image.open(img_path) as im:
        if im.mode == 'RGB':
            img_np = np.array(im)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = np.array(im.convert('L'))

    # Trouver le masque correspondant
    # Convertir en Path pour garantir la cohérence
    img_path = Path(img_path)

    # Liste des classes possibles
    possible_classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']

    if dataset_root is None:
        # Essayer de déduire depuis le chemin de l'image
        parts = img_path.parts
        if 'images' in parts:
            idx = parts.index('images')
            dataset_root = Path(*parts[:idx])
            class_name = parts[idx - 1]
        else:
            raise ValueError("Impossible de trouver le dataset_root depuis le chemin")
    else:
        # Convertir dataset_root en Path et résoudre en absolu
        dataset_root = Path(dataset_root).resolve()
        img_path = img_path.resolve()  # Résoudre aussi l'image en absolu

        # Méthode robuste: essayer de trouver la classe en testant chaque classe possible
        class_name = None
        img_path_str = str(img_path)
        dataset_root_str = str(dataset_root)

        # Essayer chaque classe possible
        for cls in possible_classes:
            test_images_dir = (dataset_root / cls / 'images').resolve()
            test_images_dir_str = str(test_images_dir)

            # Vérifier si l'image est dans ce répertoire
            # Utiliser une comparaison plus robuste
            if test_images_dir.exists():
                # Vérifier que le chemin de l'image commence par le chemin du répertoire images
                if img_path_str.startswith(test_images_dir_str):
                    class_name = cls
                    break
                # Aussi vérifier avec parent pour être sûr
                elif img_path.parent.resolve() == test_images_dir:
                    class_name = cls
                    break

        # Si pas trouvé, essayer la méthode par parsing du chemin
        if class_name is None:
            parts = img_path.parts
            if 'images' in parts:
                idx = parts.index('images')
                if idx > 0:
                    potential_class = parts[idx - 1]
                    # Vérifier que cette classe existe dans le dataset
                    test_path = (dataset_root / potential_class / 'images').resolve()
                    if test_path.exists():
                        class_name = potential_class
                    else:
                        # Chercher dans les parties du chemin
                        for cls in possible_classes:
                            if cls in parts:
                                test_path = (dataset_root / cls / 'images').resolve()
                                if test_path.exists():
                                    class_name = cls
                                    break

        if class_name is None:
            # Debug: afficher les informations
            print(f"DEBUG - Classe non trouvée:")
            print(f"  Image path: {img_path}")
            print(f"  Image path (str): {img_path_str}")
            print(f"  Dataset root: {dataset_root}")
            print(f"  Dataset root (str): {dataset_root_str}")
            for cls in possible_classes:
                test_dir = (dataset_root / cls / 'images').resolve()
                print(
                    f"  Test {cls}: {test_dir} exists={test_dir.exists()}, starts_with={img_path_str.startswith(str(test_dir))}"
                )
            raise ValueError(
                f"Classe non trouvée dans le chemin: {img_path}\n"
                f"Dataset root: {dataset_root}\n"
                f"Vérifiez que l'image est dans: {dataset_root}/CLASS/images/"
            )

    # Chemin du masque - utiliser resolve() pour les chemins absolus
    mask_path = (dataset_root / class_name / 'masks' / img_path.name).resolve()

    if not mask_path.exists():
        # Vérifier si le répertoire masks existe
        masks_dir = (dataset_root / class_name / 'masks').resolve()
        if not masks_dir.exists():
            print(f"  Répertoire masks non trouvé: {masks_dir}")
        else:
            # Lister quelques masques disponibles pour debug
            available_masks = list(masks_dir.glob('*.png'))[:5]
            print(f"  Masque non trouvé: {mask_path}")
            print(f"   Classe détectée: {class_name}")
            print(f"   Nom de l'image: {img_path.name}")
            print(f"   Répertoire masks existe: {masks_dir.exists()}")
            if available_masks:
                print(f"   Exemples de masques disponibles: {[m.name for m in available_masks]}")
        print(f"   Utilisation de l'image complète (sans masque)")
        # Retourner l'image complète si pas de masque
        mask_binary = np.ones_like(img_gray, dtype=bool)
        return img_gray, mask_binary, img_gray

    # Charger et redimensionner le masque (KNN)
    with Image.open(mask_path) as m:
        m = m.convert('L')
        # Redimensionner à 299x299 avec KNN
        m_resized = m.resize(TARGET_SIZE, resample=Image.NEAREST)
        m_np = np.asarray(m_resized)

    # Redimensionner l'image si nécessaire
    if img_gray.shape != TARGET_SIZE:
        img_pil = Image.fromarray(img_gray)
        img_gray = np.array(img_pil.resize(TARGET_SIZE, resample=Image.BILINEAR))

    # Binariser le masque
    mask_binary = m_np > 0

    # Appliquer le masque (zones non masquées = 0)
    img_masked = img_gray.copy()
    img_masked[~mask_binary] = 0

    return img_masked, mask_binary, img_gray


def detect_artifacts_auto(
    img_path: Path,
    threshold: float = 0.3,
    visualize: bool = False,
    use_mask: bool = True,
    dataset_root: Path = None,
    sensitivity: str = 'medium',
) -> dict:
    """
    Détecte automatiquement la présence d'artefacts dans une image.

    Utilise plusieurs techniques:
    - Transformée de Fourier pour détecter patterns répétitifs
    - Analyse de variance locale pour détecter textes/annotations
    - Détection de bordures uniformes
    - Analyse d'énergie haute fréquence

    Paramètres:
    -----------
    img_path : Path
        Chemin vers l'image à analyser
    threshold : float
        Seuil de confiance global (0-1) pour considérer qu'il y a des artefacts
    visualize : bool
        Affiche les visualisations si True
    use_mask : bool
        Si True, analyse uniquement les zones masquées (poumons)
    dataset_root : Path or None
        Racine du dataset (pour trouver les masques)
    sensitivity : str
        Niveau de sensibilité: 'low' (strict), 'medium' (défaut), 'high' (sensible)
        - 'low': Seuils stricts, moins de faux positifs, peut manquer des artefacts
        - 'medium': Équilibre entre précision et rappel
        - 'high': Seuils plus permissifs, détecte plus d'artefacts mais plus de faux positifs

    Retourne:
    --------
    dict avec:
        - 'has_artifacts' : bool
        - 'confidence' : float (0-1)
        - 'artifact_types' : list
        - 'scores' : dict (scores détaillés par type)
    """
    # Configuration des seuils selon la sensibilité
    thresholds_config = _get_thresholds_config(sensitivity)
    # Chargement de l'image et application du masque si demandé
    if use_mask:
        img_gray, mask_binary, img_original = load_and_apply_mask(img_path, dataset_root)
        # Ne pas afficher si pas de visualisation (pour éviter trop de messages lors du scan)
        if visualize:
            print(f" Masque appliqué: {mask_binary.sum()} pixels masqués sur {mask_binary.size}")
    else:
        # Chargement de l'image sans masque
        with Image.open(img_path) as im:
            if im.mode == 'RGB':
                img_np = np.array(im)
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = np.array(im.convert('L'))
        mask_binary = np.ones_like(img_gray, dtype=bool)
        img_original = img_gray.copy()

    h, w = img_gray.shape
    scores = []
    artifact_types = []

    # 1. DÉTECTION PAR FFT (Patterns répétitifs) - 40% du score
    # Utiliser uniquement les zones masquées
    score_fft, magnitude_norm, peaks = _detect_fft_patterns(
        img_gray,
        mask_binary if use_mask else None,
        percentile_threshold=thresholds_config['fft_percentile'],
    )
    # Seuil ajustable selon la sensibilité
    if score_fft > thresholds_config['fft_threshold']:
        artifact_types.append('Pattern répétitif (FFT)')
    # Réduire le poids si le score est modéré
    fft_weight = 0.4 if score_fft > thresholds_config['fft_threshold'] else 0.2
    scores.append(score_fft * fft_weight)

    # 2. DÉTECTION DE BORDURES UNIFORMES - 20% du score
    # Sur les zones masquées uniquement
    border_score = _detect_uniform_borders(img_gray, mask_binary if use_mask else None)
    # Seuil ajustable selon la sensibilité
    if border_score > thresholds_config['border_threshold']:
        artifact_types.append('Bordures suspectes')
    scores.append(border_score * 0.2)

    # 3. DÉTECTION DE TEXTES/ANNOTATIONS - 20% du score
    # Sur les zones masquées uniquement
    score_text, high_var_mask = _detect_text_annotations(
        img_gray,
        mask_binary if use_mask else None,
        percentile_threshold=thresholds_config['text_percentile'],
    )
    # Seuil ajustable selon la sensibilité
    if score_text > thresholds_config['text_threshold']:
        artifact_types.append('Textes/Annotations')
    scores.append(score_text * 0.2)

    # 4. DÉTECTION D'ÉNERGIE HAUTE FRÉQUENCE - 20% du score
    # Sur les zones masquées uniquement
    score_hf = _detect_high_frequency_artifacts(
        img_gray, magnitude_norm, mask_binary if use_mask else None
    )
    # Seuil ajustable selon la sensibilité
    if score_hf > thresholds_config['high_freq_threshold']:
        artifact_types.append('Artefacts haute fréquence')
    # Réduire le poids si le score est modéré
    hf_weight = 0.2 if score_hf > thresholds_config['high_freq_threshold'] else 0.1
    scores.append(score_hf * hf_weight)

    # Score global avec validation stricte
    confidence = sum(scores)

    # Validation supplémentaire : nécessite au moins N indicateurs significatifs OU un indicateur très fort
    strong_indicators = sum(
        [
            score_fft > thresholds_config['fft_strong'],
            border_score > thresholds_config['border_strong'],
            score_text > thresholds_config['text_strong'],
            score_hf > thresholds_config['high_freq_strong'],
        ]
    )

    # Détection seulement si :
    # - Score global >= threshold ET
    # - (Au moins N indicateurs forts OU un indicateur très fort OU score global très élevé)
    has_artifacts = confidence >= threshold and (
        strong_indicators >= thresholds_config['min_strong_indicators']
        or confidence > thresholds_config['very_high_confidence']
        or score_fft > thresholds_config['fft_very_strong']
        or border_score > thresholds_config['border_very_strong']
        or score_text > thresholds_config['text_very_strong']
    )

    # Visualisation
    if visualize:
        _visualize_detection(
            img_original if use_mask else img_gray,
            img_gray,  # Image masquée
            mask_binary if use_mask else None,
            magnitude_norm,
            peaks,
            high_var_mask,
            confidence,
            has_artifacts,
            artifact_types,
            img_path.name,
            use_mask,
        )

    return {
        'has_artifacts': has_artifacts,
        'confidence': confidence,
        'artifact_types': artifact_types,
        'scores': {
            'fft': score_fft,
            'borders': border_score,
            'text': score_text,
            'high_freq': score_hf,
        },
    }


def _get_thresholds_config(sensitivity: str = 'medium') -> dict:
    """
    Retourne la configuration des seuils selon le niveau de sensibilité.

    Paramètres:
    -----------
    sensitivity : str
        'low', 'medium', ou 'high'

    Retourne:
    --------
    dict avec tous les seuils configurés
    """
    configs = {
        'low': {
            # Seuils stricts (moins de faux positifs, peut manquer des artefacts)
            'fft_threshold': 0.6,
            'fft_strong': 0.7,
            'fft_very_strong': 0.8,
            'fft_percentile': 99,  # 99ème percentile
            'border_threshold': 0.6,
            'border_strong': 0.8,
            'border_very_strong': 0.85,
            'text_threshold': 0.5,
            'text_strong': 0.6,
            'text_very_strong': 0.7,
            'text_percentile': 98,  # 98ème percentile
            'high_freq_threshold': 0.6,
            'high_freq_strong': 0.7,
            'min_strong_indicators': 2,
            'very_high_confidence': 0.75,
        },
        'medium': {
            # Équilibre (défaut)
            'fft_threshold': 0.5,
            'fft_strong': 0.6,
            'fft_very_strong': 0.7,
            'fft_percentile': 98,  # 98ème percentile
            'border_threshold': 0.5,
            'border_strong': 0.75,
            'border_very_strong': 0.75,
            'text_threshold': 0.4,
            'text_strong': 0.5,
            'text_very_strong': 0.6,
            'text_percentile': 97,  # 97ème percentile
            'high_freq_threshold': 0.5,
            'high_freq_strong': 0.6,
            'min_strong_indicators': 2,
            'very_high_confidence': 0.7,
        },
        'high': {
            # Seuils permissifs (détecte plus d'artefacts, plus de faux positifs)
            'fft_threshold': 0.3,
            'fft_strong': 0.4,
            'fft_very_strong': 0.5,
            'fft_percentile': 95,  # 95ème percentile
            'border_threshold': 0.3,
            'border_strong': 0.5,
            'border_very_strong': 0.6,
            'text_threshold': 0.25,
            'text_strong': 0.35,
            'text_very_strong': 0.45,
            'text_percentile': 95,  # 95ème percentile
            'high_freq_threshold': 0.3,
            'high_freq_strong': 0.4,
            'min_strong_indicators': 1,  # Un seul indicateur fort suffit
            'very_high_confidence': 0.5,
        },
    }

    if sensitivity not in configs:
        print(f"  Sensibilité '{sensitivity}' non reconnue, utilisation de 'medium'")
        sensitivity = 'medium'

    return configs[sensitivity]


def _detect_fft_patterns(
    img_gray: np.ndarray, mask_binary: np.ndarray = None, percentile_threshold: int = 98
) -> tuple:
    """Détecte les patterns répétitifs via FFT."""
    h, w = img_gray.shape

    # Si un masque est fourni, ne travailler que sur les zones masquées
    if mask_binary is not None:
        # Créer une image avec seulement les zones masquées (zones non masquées = 0)
        img_for_fft = img_gray.copy()
        img_for_fft[~mask_binary] = 0
    else:
        img_for_fft = img_gray

    # Transformée de Fourier
    f = np.fft.fft2(img_for_fft.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log1p(magnitude)
    magnitude_norm = magnitude_log / (magnitude_log.max() + 1e-10)

    # Masquer le centre (DC component)
    center_h, center_w = h // 2, w // 2
    center_radius = min(h, w) // 20
    y, x = np.ogrid[:h, :w]
    mask_center = ((x - center_w) ** 2 + (y - center_h) ** 2) > center_radius**2

    # Si un masque est fourni, ne considérer que les zones masquées dans l'analyse
    if mask_binary is not None:
        # Combiner le masque de centre avec le masque binaire
        analysis_mask = mask_center & mask_binary
    else:
        analysis_mask = mask_center

    # Détection de pics significatifs - seuil ajustable selon percentile_threshold
    magnitude_analyzed = magnitude_norm.copy()
    magnitude_analyzed[~mask_center] = 0
    threshold_peak = (
        np.percentile(magnitude_analyzed[analysis_mask], percentile_threshold)
        if np.sum(analysis_mask) > 0
        else 0
    )
    peaks = magnitude_analyzed > threshold_peak

    # Vérifier que les pics forment des patterns (pas juste du bruit)
    peak_density = (
        np.sum(peaks[analysis_mask]) / np.sum(analysis_mask) if np.sum(analysis_mask) > 0 else 0
    )

    # Score ajusté : nécessite une densité plus élevée pour être significatif
    score_fft = min(1.0, max(0, (peak_density - 0.02) * 20))  # Seuil de base à 0.02

    return score_fft, magnitude_norm, peaks


def _detect_uniform_borders(
    img_gray: np.ndarray, mask_binary: np.ndarray = None, border_thickness: int = 15
) -> float:
    """Détecte les bordures uniformes suspectes dans les zones masquées."""
    # Si un masque est fourni, ne considérer que les bordures dans les zones masquées
    if mask_binary is not None:
        # Extraire les bordures masquées uniquement
        borders = [
            img_gray[:border_thickness, :][mask_binary[:border_thickness, :]],  # Haut
            img_gray[-border_thickness:, :][mask_binary[-border_thickness:, :]],  # Bas
            img_gray[:, :border_thickness][mask_binary[:, :border_thickness]],  # Gauche
            img_gray[:, -border_thickness:][mask_binary[:, -border_thickness:]],  # Droite
        ]
    else:
        borders = [
            img_gray[:border_thickness, :],  # Haut
            img_gray[-border_thickness:, :],  # Bas
            img_gray[:, :border_thickness],  # Gauche
            img_gray[:, -border_thickness:],  # Droite
        ]

    border_score = 0
    for border in borders:
        if len(border) > 0 and border.std() < 15:  # Bordure très uniforme
            border_score += 0.25

    return border_score


def _detect_text_annotations(
    img_gray: np.ndarray, mask_binary: np.ndarray = None, percentile_threshold: int = 97
) -> tuple:
    """Détecte les zones de texte/annotations via variance locale dans les zones masquées."""
    h, w = img_gray.shape

    # Variance locale
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    mean_local = cv2.filter2D(img_gray.astype(np.float32), -1, kernel)
    sqr_mean = cv2.filter2D((img_gray.astype(np.float32)) ** 2, -1, kernel)
    variance_local = sqr_mean - mean_local**2

    # Si un masque est fourni, ne considérer que les zones masquées
    if mask_binary is not None:
        variance_local[~mask_binary] = 0
        # Zones de haute variance - seuil ajustable selon percentile_threshold sur zones masquées
        masked_variance = variance_local[mask_binary]
        if len(masked_variance) > 0:
            threshold_var = np.percentile(masked_variance, percentile_threshold)
            high_var_mask = (variance_local > threshold_var) & mask_binary
            text_ratio = np.sum(high_var_mask) / np.sum(
                mask_binary
            )  # Ratio par rapport aux zones masquées
        else:
            high_var_mask = np.zeros_like(mask_binary, dtype=bool)
            text_ratio = 0
    else:
        # Zones de haute variance - seuil ajustable selon percentile_threshold
        threshold_var = np.percentile(variance_local, percentile_threshold)
        high_var_mask = variance_local > threshold_var
        text_ratio = np.sum(high_var_mask) / (h * w)

    # Score ajusté : nécessite un ratio plus élevé
    score_text = min(1.0, max(0, (text_ratio - 0.02) * 25))  # Seuil de base à 0.02

    return score_text, high_var_mask


def _detect_high_frequency_artifacts(
    img_gray: np.ndarray, magnitude_norm: np.ndarray, mask_binary: np.ndarray = None
) -> float:
    """Détecte l'énergie haute fréquence anormale dans les zones masquées."""
    h, w = img_gray.shape
    center_h, center_w = h // 2, w // 2

    # Distance au centre
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
    max_dist = np.sqrt(center_h**2 + center_w**2)
    distances_norm = distances / max_dist

    # Énergie haute vs basse fréquence
    # Zone haute fréquence plus restrictive (80% au lieu de 70%)
    high_freq_mask = distances_norm > 0.8
    low_freq_mask = ~high_freq_mask

    # Si un masque est fourni, ne considérer que les zones masquées
    if mask_binary is not None:
        high_freq_mask = high_freq_mask & mask_binary
        low_freq_mask = low_freq_mask & mask_binary

    if np.sum(high_freq_mask) > 0 and np.sum(low_freq_mask) > 0:
        high_freq_energy = np.mean(magnitude_norm[high_freq_mask])
        low_freq_energy = np.mean(magnitude_norm[low_freq_mask])
        energy_ratio = high_freq_energy / (low_freq_energy + 1e-10)
    else:
        energy_ratio = 0

    # Score ajusté : nécessite un ratio plus élevé
    score_hf = min(1.0, max(0, (energy_ratio - 0.2) * 2))  # Seuil de base à 0.2

    return score_hf


def _visualize_detection(
    img_original: np.ndarray,
    img_masked: np.ndarray,
    mask_binary: np.ndarray,
    magnitude_norm: np.ndarray,
    peaks: np.ndarray,
    high_var_mask: np.ndarray,
    confidence: float,
    has_artifacts: bool,
    artifact_types: list,
    img_name: str,
    use_mask: bool = False,
):
    """Visualise les résultats de détection."""
    if use_mask:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = np.array([[axes[0, 0], axes[0, 1]], [axes[1, 0], axes[1, 1]]])

    # Image originale
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title(f'Image originale\n{img_name}')
    axes[0, 0].axis('off')

    # Image masquée (si masque utilisé)
    if use_mask and mask_binary is not None:
        axes[0, 1].imshow(img_original, cmap='gray', alpha=0.7)
        axes[0, 1].imshow(mask_binary, alpha=0.3, cmap='Greens')
        axes[0, 1].set_title('Image avec masque (zones vertes)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(img_masked, cmap='gray')
        axes[0, 2].set_title('Image masquée (zones analysées)')
        axes[0, 2].axis('off')

        col_idx = 0
    else:
        col_idx = 1

    # Spectre FFT
    magnitude_log = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img_masked.astype(np.float32)))))
    im1 = axes[1, col_idx].imshow(magnitude_log, cmap='hot')
    axes[1, col_idx].set_title('Spectre de Fourier')
    axes[1, col_idx].axis('off')
    plt.colorbar(im1, ax=axes[1, col_idx])
    col_idx += 1

    # Pics détectés
    if col_idx < axes.shape[1]:
        magnitude_peaks = magnitude_norm.copy()
        magnitude_peaks[~peaks] = 0
        axes[1, col_idx].imshow(magnitude_norm, cmap='hot', alpha=0.7)
        axes[1, col_idx].imshow(magnitude_peaks, cmap='cool', alpha=0.6)
        axes[1, col_idx].set_title('Pics détectés (bleu)')
        axes[1, col_idx].axis('off')
        col_idx += 1

    # Variance locale (textes)
    if col_idx < axes.shape[1]:
        axes[1, col_idx].imshow(img_masked, cmap='gray', alpha=0.7)
        axes[1, col_idx].imshow(high_var_mask, alpha=0.4, cmap='Reds')
        axes[1, col_idx].set_title('Zones haute variance (textes)')
        axes[1, col_idx].axis('off')

    mask_status = " (zones masquées uniquement)" if use_mask else ""
    plt.suptitle(
        f'Détection automatique d\'artefacts{mask_status}\n'
        f'Confiance: {confidence:.3f} | '
        f'Artefacts: {"OUI ✓" if has_artifacts else "NON ✗"}\n'
        f'Types: {", ".join(artifact_types) if artifact_types else "Aucun"}',
        fontsize=12,
        y=0.98,
    )
    plt.tight_layout()
    plt.show()
