"""
Version modifiée de train_model qui utilise F1-score pour l'early stopping
au lieu de accuracy (adapté aux datasets déséquilibrés).
"""

import torch
import time
from model_utils import train_epoch  # Utiliser la fonction existante
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from model_utils import convert_to_binary_class


def validate_epoch_f1(model, dataloader, criterion, device, class_to_idx, binary_mode=False):
    """
    Valide le modèle sur le dataset de validation.

    Utilise des métriques adaptées aux datasets déséquilibrés :
    - F1-score (métrique principale pour early stopping)
    - Accuracy (pour référence)

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
        (epoch_loss, epoch_acc, val_f1, all_preds, all_labels)
        où val_f1 est le F1-score (adapté aux datasets déséquilibrés)
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

    # Calculer F1-score (métrique adaptée aux datasets déséquilibrés)
    if binary_mode:
        # Pour classification binaire, utiliser F1-score de la classe positive (COVID)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        val_f1 = f1[1]  # F1-score pour la classe COVID (classe positive)
    else:
        # Pour multi-classes, utiliser F1-score weighted
        _, _, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        val_f1 = f1_weighted

    return epoch_loss, epoch_acc, val_f1, all_preds, all_labels


def train_model_f1(
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
    early_stopping_metric='f1',
):
    """
    Fonction principale pour entraîner un modèle.

    Utilise F1-score pour l'early stopping (adapté aux datasets déséquilibrés).

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
    early_stopping_metric : str
        Métrique pour early stopping : 'f1' (recommandé) ou 'accuracy'

    Retourne:
    --------
    dict
        Historique de l'entraînement avec toutes les métriques
    """
    # Utiliser F1-score comme métrique principale pour l'early stopping
    best_val_metric = 0.0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],  # Ajouter F1-score à l'historique
    }

    print(f" Démarrage de l'entraînement")
    print(f"   Époques: {num_epochs}")
    print(f"   Device: {device}")
    print(f"   Classification: {'Binaire' if binary_mode else 'Multi-classes'}")
    print(
        f"   Métrique early stopping: {early_stopping_metric.upper()} (adapté aux datasets déséquilibrés)"
    )
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

        # Validation (retourne maintenant F1-score)
        val_loss, val_acc, val_f1, val_preds, val_labels = validate_epoch_f1(
            model, val_loader, criterion, device, class_to_idx, binary_mode=binary_mode
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        # Mettre à jour le scheduler (utiliser la loss)
        scheduler.step(val_loss)

        # Afficher les résultats avec F1-score
        print(f"\n Résultats Époque {epoch + 1}:")
        print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"   Val   - F1-score: {val_f1:.4f} (métrique principale)")

        # Early stopping basé sur F1-score (ou accuracy selon le choix)
        if early_stopping_metric == 'f1':
            current_metric = val_f1
            metric_name = "F1-score"
        else:
            current_metric = val_acc
            metric_name = "Accuracy"

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
            print(f"    Meilleure validation {metric_name}: {best_val_metric:.4f}")
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping après {epoch + 1} époques")
                print(f"   (Pas d'amélioration de {metric_name} depuis {patience} époques)")
                break

    elapsed_time = time.time() - start_time
    print(f"\n Entraînement terminé en {elapsed_time/60:.2f} minutes")
    print(f"   Meilleure validation {metric_name}: {best_val_metric:.4f}")
    if early_stopping_metric == 'f1':
        print(f"   (Accuracy finale: {history['val_acc'][-1]:.4f})")

    return history
