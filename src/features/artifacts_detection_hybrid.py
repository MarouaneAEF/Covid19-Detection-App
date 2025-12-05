"""
Détection d'artefacts hybride (Fourier + contraste/contours).

Ce module fournit une fonction unique :

    detect_artifacts_hybrid(...)

qui combine :
  - la détection fréquentielle (FFT, variance locale, bords, HF)
    via `detect_artifacts_auto` ;
  - la détection spatiale par contraste et contours
    via `detect_artefacts_pure`.

Aucune visualisation n'est réalisée ici : uniquement de la détection.
"""

from pathlib import Path
from typing import Dict, Any

import numpy as np

# Imports compatibles avec les deux cas : script direct et import depuis
# package
try:
    # Import relatif (pour import depuis package features)
    from .artifacts_detection_update import (
        detect_artifacts_auto,
        load_and_apply_mask,
    )
    from .adetection_pure import detect_artefacts_pure
except ImportError:
    # Import absolu (pour script lancé directement depuis le répertoire)
    from artifacts_detection_update import (
        detect_artifacts_auto,
        load_and_apply_mask,
    )
    from adetection_pure import detect_artefacts_pure


def detect_artifacts_hybrid(
    img_path: Path,
    dataset_root: Path,
    threshold_fft: float = 0.3,
    sensitivity: str = "medium",
    use_mask: bool = True,
    area_ratio_threshold: float = 0.005,
) -> Dict[str, Any]:
    """
    Détection hybride d'artefacts sur une image de radiographie.

    Combine :
      - la détection fréquentielle (FFT + heuristiques globales)
      - la détection spatiale (contraste + contours dans les poumons)

    Paramètres
    ----------
    img_path : Path
        Chemin vers l'image à analyser (dans le dataset COVID-19_Radiography_Dataset).
    dataset_root : Path
        Racine du dataset (ex: .../COVID-19_Radiography_Dataset).
    threshold_fft : float
        Seuil global pour la détection FFT (utilisé par detect_artifacts_auto).
    sensitivity : str
        Sensibilité pour detect_artifacts_auto ('low', 'medium', 'high').
    use_mask : bool
        Si True, utilise les masques pulmonaires du dataset pour la FFT.
    area_ratio_threshold : float
        Seuil sur le ratio (surface artefacts / surface poumon) pour la
        détection spatiale par contraste/contours.

    Retour
    ------
    dict
        {
          "has_artifacts": bool,          # décision globale (FFT OU spatial)
          "fft": { ... },                 # résultat complet de detect_artifacts_auto
          "spatial": { ... },             # résultat complet de detect_artefacts_pure
        }
    """
    img_path = Path(img_path)
    dataset_root = Path(dataset_root)

    # --- 1) Détection fréquentielle via detect_artifacts_auto (FFT, variance, etc.) ---
    fft_result = detect_artifacts_auto(
        img_path=img_path,
        threshold=threshold_fft,
        visualize=False,
        use_mask=use_mask,
        dataset_root=dataset_root,
        sensitivity=sensitivity,
    )

    # --- 2) Préparation image + masque pour la détection spatiale ---
    # On réutilise load_and_apply_mask pour obtenir :
    #   - img_masked : image avec masque poumon appliqué
    #   - mask_binary : masque booléen
    #   - img_original : image d'origine redimensionnée
    img_masked, mask_binary, img_original = load_and_apply_mask(
        img_path=img_path,
        dataset_root=dataset_root,
    )

    # detect_artefacts_pure attend un masque binaire 0/255 de type uint8
    resized_mask_uint8 = mask_binary.astype(np.uint8) * 255

    # --- 3) Détection spatiale (contraste + contours) ---
    spatial_result = detect_artefacts_pure(
        img=img_original,
        resized_mask=resized_mask_uint8,
        debug_mode=False,
        area_ratio_threshold=area_ratio_threshold,
    )

    # --- 4) Décision globale ---
    # Combiner les deux méthodes : FFT OU spatial
    # Si l'une des deux détecte des artefacts, on considère qu'il y en a
    has_artifacts_global = bool(
        fft_result.get("has_artifacts", False)
        or spatial_result.get("has_artifacts", False)
    )

    return {
        "has_artifacts": has_artifacts_global,
        "fft": fft_result,
        "spatial": spatial_result,
    }
