"""
Détection pure d'artefacts sur les radiographies pulmonaires.

Ce module fournit une fonction principale :
    - detect_artefacts_pure(img, resized_mask, ...)

qui retourne :
    - un booléen has_artifacts
    - un score area_ratio (surface artefacts / surface poumon)
    - les masques d'artefacts intermédiaires.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(title: str, img: np.ndarray) -> None:
    """
    Affiche une image avec Matplotlib (utile en debug dans les notebooks).
    """
    plt.figure(figsize=(4, 4))
    cmap = "gray" if img.ndim == 2 else None
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


def estimate_canny_minmax(img: np.ndarray) -> tuple[int, int]:
    """
    Estime automatiquement les seuils min / max pour Canny à partir de l'histogramme.

    Code adapté depuis le notebook d'exploration des artefacts.
    """
    # Histogramme (256 niveaux)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

    # Normalisation
    hist = hist / (hist.sum() + 1e-12)

    # Calcul du "spread" : distance entre les percentiles 5% et 95%
    cdf = np.cumsum(hist)
    p5 = np.searchsorted(cdf, 0.05)
    p95 = np.searchsorted(cdf, 0.95)

    spread = (p95 - p5) / 255.0  # ~ 0.1 à 1.0

    # Mapping : plus l'image est contrastée → sigma petit
    sigma = 0.7 - 0.5 * spread
    sigma = float(np.clip(sigma, 0.3, 0.7))

    # Médiane des pixels
    v = float(np.median(img))

    # Seuils Canny
    canny_min = int(max(0, (1.0 - sigma) * v))
    canny_max = int(min(255, (1.0 + sigma) * v))

    if v == 0:
        canny_min = -1
        canny_max = -1

    return canny_min, canny_max


def mask_generator_by_contours(
    img: np.ndarray,
    resized_mask: np.ndarray,
    debug_mode: bool = False,
) -> np.ndarray:
    """
    Génère un masque binaire des artefacts en exploitant la détection de contours de Canny.

    Paramètres
    ----------
    img : np.ndarray
        Image en niveaux de gris.
    resized_mask : np.ndarray
        Masque binaire (0/255) des poumons, même taille que img.
    debug_mode : bool
        Si True, affiche les images étape par étape.

    Retour
    ------
    np.ndarray
        Masque binaire 8-bit (255 = artefact).
    """
    if debug_mode:
        show("img", img)

    # Normalisation de l'image
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug_mode:
        show("Grayscale Input", gray)

    # Seuils Canny automatiques
    canny_min, canny_max = estimate_canny_minmax(gray)
    if debug_mode:
        print(f"{canny_min = }, {canny_max = }")

    # Détection des bords (filtre Canny)
    if 0 <= canny_min <= 255:
        edges = cv2.Canny(gray, canny_min, canny_max)
    else:
        edges = np.zeros_like(gray)
    if debug_mode:
        show("edges", edges)

    # Dilatation pour renforcer les bords
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    if debug_mode:
        show("edges_dilated", edges_dilated)

    # Remasquage par le masque poumon
    # On utilise un poumon "érosé" pour ignorer une bande près du bord externe,
    # afin de ne pas détecter systématiquement le contour des poumons comme artefact.
    lung_binary = (resized_mask > 0).astype(np.uint8)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner_lung = cv2.erode(lung_binary, erosion_kernel, iterations=1)
    inner_lung_mask = (inner_lung * 255).astype(np.uint8)

    remask_mask = cv2.bitwise_and(edges_dilated, inner_lung_mask)
    if debug_mode:
        show("remask", remask_mask)

    # Suppression des composantes trop grandes ou trop petites
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(remask_mask)
    mask_filtered = np.zeros_like(remask_mask)
    max_area = 2000
    min_area = 50
    if debug_mode:
        print(f"{n = }")
        print(f"{stats_cc = }")
    for i in range(1, n):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if debug_mode:
            print(f"{area = }")
        if min_area <= area <= max_area:
            mask_filtered[labels == i] = 255
    if debug_mode:
        show("mask_filtered", mask_filtered)

    # Découpage du masque par poumon
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(resized_mask)
    mask_poumon_1 = np.zeros_like(resized_mask)
    mask_poumon_2 = np.zeros_like(resized_mask)

    for i in range(1, n):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if debug_mode:
            print(f"{area = }")
            print(f"{stats_cc = }")
        if i == 1:
            mask_poumon_1[labels == i] = 255
        elif i == 2:
            mask_poumon_2[labels == i] = 255

    # On découpe le masque filtré par poumon
    remask_mask_poumon_1 = cv2.bitwise_and(mask_filtered, mask_poumon_1)
    remask_mask_poumon_2 = cv2.bitwise_and(mask_filtered, mask_poumon_2)

    # Filtrage sur le poumon 1
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(remask_mask_poumon_1)
    mask_filtered_1 = np.zeros_like(remask_mask_poumon_1)
    max_area = 2000
    min_area = 20
    if debug_mode:
        print("n 1 = ", n)
        print(f"{stats_cc = }")
    if n <= 5:
        for i in range(1, n):
            area = stats_cc[i, cv2.CC_STAT_AREA]
            if debug_mode:
                print(f"{area = }")
            if min_area <= area <= max_area:
                mask_filtered_1[labels == i] = 255
    if debug_mode:
        show("mask_filtered_1", mask_filtered_1)

    # Filtrage sur le poumon 2
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(remask_mask_poumon_2)
    mask_filtered_2 = np.zeros_like(remask_mask_poumon_2)
    max_area = 2000
    min_area = 20
    if debug_mode:
        print("n 2 = ", n)
        print(f"{stats_cc = }")
    if n <= 5:
        for i in range(1, n):
            area = stats_cc[i, cv2.CC_STAT_AREA]
            if debug_mode:
                print(f"{area = }")
            if min_area <= area <= max_area:
                mask_filtered_2[labels == i] = 255
    if debug_mode:
        show("mask_filtered_2", mask_filtered_2)

    # Assemblage des deux poumons
    mask_filtered_3 = cv2.bitwise_or(mask_filtered_1, mask_filtered_2)
    if debug_mode:
        show("mask_filtered_3", mask_filtered_3)

    return mask_filtered_3


def mask_generator_by_contrast(
    img: np.ndarray,
    resized_mask: np.ndarray,
    debug_mode: bool = False,
) -> np.ndarray:
    """
    Génère un masque binaire des artefacts en exploitant la propriété contrastante
    d'un colormap.

    Paramètres
    ----------
    img : np.ndarray
        Image en niveaux de gris.
    resized_mask : np.ndarray
        Masque binaire (0/255) des poumons, même taille que img.
    debug_mode : bool
        Si True, affiche les images étape par étape.

    Retour
    ------
    np.ndarray
        Masque binaire 8-bit (255 = artefact).
    """
    # Normalisation de l'image
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if debug_mode:
        show("Grayscale Input", gray)

    # Application d'un color_map (COLORMAP.BONE)
    colored = cv2.applyColorMap(gray, 1)
    if debug_mode:
        show("Colormap", colored)

    # Conversion en HSV pour segmentation couleur
    hsv = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV)

    # Détection des zones non contrastées (non artefacts)
    lower = np.array([0, 15, 0])
    upper = np.array([125, 255, 230])
    mask_hsv = cv2.inRange(hsv, lower, upper)
    if debug_mode:
        show("Raw hsv Mask", mask_hsv)

    # Morphologie légère pour stabiliser le masque
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    mask_morph = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, k, iterations=1)
    mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, k, iterations=1)
    if debug_mode:
        show("Morphology Light", mask_morph)

    # Inversion : les zones restantes correspondent aux artefacts
    mask_inverted = cv2.bitwise_not(mask_morph)
    if debug_mode:
        show("Inverted Mask", mask_inverted)

    # Remasquage pour ne garder que les zones pulmonaires
    remask_mask = cv2.bitwise_and(mask_inverted, resized_mask)
    if debug_mode:
        show("remask", remask_mask)

    # Filtrage des zones par taille (supprimer les zones trop grandes)
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(remask_mask)
    mask_filtered = np.zeros_like(remask_mask)
    max_area = 600
    for i in range(1, n):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if debug_mode:
            print(area)
        if area <= max_area:
            mask_filtered[labels == i] = 255
    if debug_mode:
        show("mask_filtered", mask_filtered)

    # Dilatation pour renforcer les bords
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (3, 3))
    mask_dilated = cv2.dilate(mask_filtered, kernel, iterations=3)
    if debug_mode:
        show("Dilated mask", mask_dilated)

    # Second filtrage pour supprimer les très petites zones
    n, labels, stats_cc, _ = cv2.connectedComponentsWithStats(mask_dilated)
    mask_refiltered = np.zeros_like(mask_dilated)
    min_area = 50
    for i in range(1, n):
        area = stats_cc[i, cv2.CC_STAT_AREA]
        if debug_mode:
            print(area)
        if area >= min_area:
            mask_refiltered[labels == i] = 255
    if debug_mode:
        show("mask_refiltered", mask_refiltered)

    return mask_refiltered


def detect_artefacts_pure(
    img: np.ndarray,
    resized_mask: np.ndarray,
    debug_mode: bool = False,
    area_ratio_threshold: float = 0.005,
) -> dict:
    """
    Détection pure d'artefacts à partir d'une image et de son masque poumon.

    Cette fonction ne fait PAS de nettoyage (inpainting), uniquement de la
    détection et retourne un masque d'artefacts + un score.

    Paramètres
    ----------
    img : np.ndarray
        Image en niveaux de gris (uint8 ou convertible en uint8).
    resized_mask : np.ndarray
        Masque binaire (0/255) des poumons, même taille que img.
    debug_mode : bool
        Si True, affiche les masques intermédiaires.
    area_ratio_threshold : float
        Seuil sur le ratio (aire artefacts / aire poumon) pour décider has_artifacts.

    Retour
    ------
    dict :
        {
            "has_artifacts": bool,
            "area_ratio": float,
            "artifact_mask": np.ndarray (uint8, 0/255),
            "mask_contrast": np.ndarray,
            "mask_contours": np.ndarray,
        }
    """
    # Normalisation / typage doux
    gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lung_mask = (resized_mask > 0).astype(np.uint8) * 255

    # 1) Masque par contraste
    mask_contrast = mask_generator_by_contrast(
        img=gray,
        resized_mask=lung_mask,
        debug_mode=debug_mode,
    )

    # 2) Masque par contours
    mask_contours = mask_generator_by_contours(
        img=gray,
        resized_mask=lung_mask,
        debug_mode=debug_mode,
    )

    # 3) Masque final = intersection des deux
    # On ne garde que les zones qui sont à la fois:
    #   - anormales en contraste (mask_contrast)
    #   - supportées par des contours (mask_contours)
    # Cela réduit fortement les faux positifs le long des bords pulmonaires.
    artifact_mask = cv2.bitwise_and(mask_contrast, mask_contours)

    # 4) Calcul du ratio de surface d'artefacts dans la zone poumon
    lung_area = int(np.count_nonzero(lung_mask))
    artifact_area = int(np.count_nonzero(artifact_mask))
    area_ratio = float(artifact_area / lung_area) if lung_area > 0 else 0.0

    has_artifacts = area_ratio >= area_ratio_threshold

    if debug_mode:
        print(
            f"lung_area = {lung_area}, "
            f"artifact_area = {artifact_area}, "
            f"area_ratio = {area_ratio:.6f}"
        )
        show("mask_contrast", mask_contrast)
        show("mask_contours", mask_contours)
        show("artifact_mask", artifact_mask)

    return {
        "has_artifacts": bool(has_artifacts),
        "area_ratio": area_ratio,
        "artifact_mask": artifact_mask,
        "mask_contrast": mask_contrast,
        "mask_contours": mask_contours,
    }
