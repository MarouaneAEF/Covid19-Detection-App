from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import pandas as pd

from artifacts_detection_hybrid import detect_artifacts_hybrid


class ArtifactFreeDataLoaderCovid:
    """
    Data Loader pour charger les images avec masques appliqués et filtrer les artefacts.
    """

    def __init__(
        self,
        dataset_root: Path,
        classes: List[str],
        threshold: float = 0.3,
        sensitivity: str = 'medium',
        use_mask: bool = True,
        filter_artifacts: bool = True,
        cache_artifacts: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.classes = classes
        self.threshold = threshold
        self.sensitivity = sensitivity
        self.use_mask = use_mask
        self.filter_artifacts = filter_artifacts
        self.cache_artifacts = cache_artifacts

        # Cache pour les résultats de détection
        self._artifacts_cache = {}

        # Liste des images propres (sans artefacts)
        self._clean_images = None

        # Liste de toutes les images
        self._all_images = None

    def _get_all_images(self) -> List[dict]:
        """
        Récupère la liste de toutes les images du dataset.
        """
        if self._all_images is not None:
            return self._all_images

        all_images = []

        for cls in self.classes:
            img_dir = self.dataset_root / cls / 'images'
            mask_dir = self.dataset_root / cls / 'masks'

            if not img_dir.exists():
                print(f"Répertoire non trouvé: {img_dir}")
                continue

            # Lister toutes les images
            for img_path in sorted(img_dir.glob('*.png')):
                mask_path = mask_dir / img_path.name

                all_images.append(
                    {
                        'class': cls,
                        'image_path': img_path,
                        'mask_path': mask_path if mask_path.exists() else None,
                        'image_name': img_path.name,
                    }
                )

        self._all_images = all_images
        return all_images

    def _has_artifacts(self, img_path: Path, verbose: bool = False) -> bool:
        """
        Vérifie si une image a des artefacts.
        """
        if self.cache_artifacts and str(img_path) in self._artifacts_cache:
            return self._artifacts_cache[str(img_path)]

        # Détecter les artefacts avec la méthode hybride (FFT + spatial)
        try:
            result = detect_artifacts_hybrid(
                img_path=img_path,
                dataset_root=self.dataset_root,
                threshold_fft=self.threshold,
                sensitivity=self.sensitivity,
                use_mask=self.use_mask,
                area_ratio_threshold=0.005,  # Seuil pour la détection spatiale
            )

            has_artifacts = result['has_artifacts']

            if verbose:
                # Afficher les résultats de la détection hybride
                fft_confidence = result.get('fft', {}).get('confidence', 0.0)
                spatial_area_ratio = result.get('spatial', {}).get('area_ratio', 0.0)
                print(
                    f"  {img_path.name}: has_artifacts={has_artifacts}, "
                    f"FFT_confidence={fft_confidence:.4f}, spatial_area_ratio={spatial_area_ratio:.6f}"
                )

            # Mettre en cache
            if self.cache_artifacts:
                self._artifacts_cache[str(img_path)] = has_artifacts

            return has_artifacts

        except Exception as e:
            print(f"Erreur lors de la détection d'artefacts pour {img_path.name}: {e}")
            # En cas d'erreur, considérer comme ayant des artefacts (sécurité)
            return True

    def _get_clean_images(self) -> List[dict]:
        """
        Récupère la liste des images sans artefacts.
        """
        if self._clean_images is not None:
            return self._clean_images

        all_images = self._get_all_images()

        if not self.filter_artifacts:
            self._clean_images = all_images
            return self._clean_images

        print(f"   Filtrage des images avec artefacts...")
        print(f"   Total d'images: {len(all_images)}")
        print(f"   Seuil: {self.threshold}, Sensibilité: {self.sensitivity}\n")

        clean_images = []
        total = len(all_images)
        artifacts_count = 0

        for idx, img_info in enumerate(all_images, 1):
            # Vérifier les artefacts
            if not self._has_artifacts(img_info['image_path']):
                clean_images.append(img_info)
            else:
                artifacts_count += 1

        if total > 0:
            print(
                f"\n {len(clean_images)} images propres sur {total} ({100*len(clean_images)/total:.1f}%)"
            )
        else:
            print(f"\n {len(clean_images)} images propres sur {total} (aucune image trouvée)")

        if len(clean_images) == 0:
            print(f"\n  ATTENTION: Aucune image propre trouvée!")
            print(f"   Cela peut signifier que:")
            print(f"   - Le seuil ({self.threshold}) est trop bas")
            print(f"   - La sensibilité ({self.sensitivity}) est trop élevée")
            print(f"   - Toutes les images ont vraiment des artefacts")
            print(f"\n   Suggestions:")
            print(f"   - Augmentez le threshold (ex: 0.4 ou 0.5)")
            print(f"   - Utilisez sensitivity='low' pour être moins strict")
            print(f"   - Ou utilisez filter_artifacts=False pour charger toutes les images")
            return []

        self._clean_images = clean_images
        return clean_images

    def load_image_with_mask(
        self, img_info: dict, target_size: Tuple[int, int] = (299, 299)
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Charge une image avec son masque appliqué.
        """
        img_path = img_info['image_path']
        mask_path = img_info['mask_path']

        # Charger l'image
        with Image.open(img_path) as im:
            if im.mode == 'RGB':
                img_np = np.array(im.convert('L'))
            else:
                img_np = np.array(im.convert('L'))

        # Redimensionner l'image si nécessaire
        if img_np.shape[:2] != target_size:
            img_pil = Image.fromarray(img_np)
            img_np = np.array(img_pil.resize(target_size, resample=Image.BILINEAR))

        # Charger le masque
        if mask_path is not None:
            with Image.open(mask_path) as im:
                # Convertir en grayscale puis redimensionner avec KNN
                mask_pil = im.convert('L').resize(target_size, resample=Image.NEAREST)
                mask_np = np.array(mask_pil)
                # Binariser le masque
                mask_binary = (mask_np > 0).astype(np.float32)
                # Appliquer le masque
                img_masked = img_np.astype(np.float32) * mask_binary
        else:
            # Pas de masque, utiliser l'image complète
            mask_binary = np.ones_like(img_np, dtype=np.float32)
            img_masked = img_np.astype(np.float32)

        metadata = {
            'class': img_info['class'],
            'image_name': img_info['image_name'],
            'image_path': str(img_path),
            'has_mask': self.use_mask and mask_path and mask_path.exists(),
        }

        return img_masked, mask_binary, metadata

    def load_clean_images(
        self, target_size: Tuple[int, int] = (299, 299)
    ) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
        """
        Charge toutes les images propres (sans artefacts) avec leurs masques.

        Paramètres:
        -----------
        target_size : Tuple[int, int]
            Taille cible pour les images

        Retourne:
        --------
        List[Tuple[np.ndarray, np.ndarray, dict]]
            Liste de (image_masked, mask_binary, metadata)
        """
        clean_images_info = self._get_clean_images()

        if len(clean_images_info) == 0:
            print(f"\n  ATTENTION: Aucune image propre trouvée!")
            print(f"   Cela peut signifier que:")
            print(f"   - Le seuil ({self.threshold}) est trop bas")
            print(f"   - La sensibilité ({self.sensitivity}) est trop élevée")
            print(f"   - Toutes les images ont vraiment des artefacts")
            print(f"\n   Suggestions:")
            print(f"   - Augmentez le threshold (ex: 0.4 ou 0.5)")
            print(f"   - Utilisez sensitivity='low' pour être moins strict")
            print(f"   - Ou utilisez filter_artifacts=False pour charger toutes les images")
            return []

        print(f" Chargement de {len(clean_images_info)} images propres...")

        loaded_images = []
        for idx, img_info in enumerate(clean_images_info, 1):

            try:
                img_masked, mask_binary, metadata = self.load_image_with_mask(
                    img_info, target_size=target_size
                )
                loaded_images.append((img_masked, mask_binary, metadata))
            except Exception as e:
                print(f"Erreur lors du chargement de {img_info['image_name']}: {e}")
                continue

        print(f" {len(loaded_images)} images chargées avec succès")

        return loaded_images

    def get_statistics(self) -> dict:
        """
        Retourne les statistiques sur le dataset.
        """
        all_images = self._get_all_images()
        clean_images = self._get_clean_images()

        stats = {
            'total_images': len(all_images),
            'clean_images': len(clean_images),
            'images_with_artifacts': len(all_images) - len(clean_images),
            'filter_rate': (
                (len(all_images) - len(clean_images)) / len(all_images) if all_images else 0
            ),
        }

        # Par classe
        stats_by_class = {}
        for cls in self.classes:
            all_cls = [img for img in all_images if img['class'] == cls]
            clean_cls = [img for img in clean_images if img['class'] == cls]

            stats_by_class[cls] = {
                'total': len(all_cls),
                'clean': len(clean_cls),
                'with_artifacts': len(all_cls) - len(clean_cls),
            }

        stats['by_class'] = stats_by_class

        return stats

    def __iter__(self):
        """Permet d'itérer sur les images propres."""
        clean_images = self._get_clean_images()
        self._iterator_index = 0
        self._iterator_images = clean_images
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Itération suivante."""
        if self._iterator_index >= len(self._iterator_images):
            raise StopIteration
        img_info = self._iterator_images[self._iterator_index]
        self._iterator_index += 1
        img_masked, mask_binary, metadata = self.load_image_with_mask(img_info)
        return img_masked, mask_binary, metadata

    def __len__(self) -> int:
        """Retourne le nombre d'images propres."""
        return len(self._get_clean_images())
