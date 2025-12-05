"""
Script principal pour entraîner et évaluer le modèle COVID
VERSION AVEC F1-SCORE POUR EARLY STOPPING

Cette version utilise F1-score au lieu d'accuracy pour l'early stopping,
ce qui est adapté aux datasets déséquilibrés.

Ce script est identique à train_and_evaluate.py mais utilise :
- train_model_f1 au lieu de train_model
- validate_epoch_f1 au lieu de validate_epoch
"""

from pathlib import Path

# Imports
import torch
from model_utils import (
    create_model,
    evaluate_model,
    load_model,
    print_classification_report,
    plot_training_history,
    plot_precision_recall_curve,
    plot_roc_curve,
    analyze_thresholds,
    find_optimal_threshold,
)

# Imports des versions F1 (early stopping basé sur F1-score)
from train_model_f1 import train_model_f1 as train_model

# Note: validate_epoch_f1 est utilisé par train_model_f1, pas besoin de l'importer ici

# Imports locaux
from training_config import *
from training_utils import (
    get_device,
    create_transforms,
    create_datasets,
    get_class_mapping,
    calculate_class_weights,
    create_sampler,
    create_dataloaders,
    setup_optimizer_and_loss,
    create_empty_history,
)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================


def main():
    """Fonction principale pour entraîner et évaluer le modèle."""

    # ============================================================================
    # ÉTAPE 1: CONFIGURATION ET CRÉATION DU MODÈLE
    # ============================================================================

    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(
        "  VERSION F1-SCORE : Early stopping basé sur F1-score (adapté aux datasets déséquilibrés)"
    )
    print("=" * 70)

    device = get_device()
    print(f" Device: {device}")

    print("\n" + "=" * 70)
    print("ÉTAPE 1: Création du modèle")
    print("=" * 70)
    print(f" Configuration: MODEL_CHOICE = '{MODEL_CHOICE}', TRANSFER_MODE = '{TRANSFER_MODE}'")

    model = create_model(
        model_choice=MODEL_CHOICE,
        num_classes=NUM_CLASSES,
        device=device,
        transfer_mode=TRANSFER_MODE,
    )

    # ============================================================================
    # ÉTAPE 2: PRÉPARATION DES DONNÉES
    # ============================================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 2: Préparation des données")
    print("=" * 70)

    train_transform, val_transform = create_transforms()
    train_dataset, val_dataset, full_dataset = create_datasets(
        DATASET_ROOT,
        CLASSES,
        THRESHOLD,
        SENSITIVITY,
        USE_MASK,
        train_transform,
        val_transform,
        val_split=VAL_SPLIT,
        random_seed=RANDOM_SEED,
    )

    class_to_idx, class_names = get_class_mapping(BINARY_CLASSIFICATION, CLASSES)
    class_weights_tensor = calculate_class_weights(
        full_dataset, BINARY_CLASSIFICATION, CLASSES, device
    )

    sampler = create_sampler(train_dataset, BINARY_CLASSIFICATION, CLASSES, class_weights_tensor)

    pin_memory = False if device.type == 'mps' else True
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, BATCH_SIZE, sampler, NUM_WORKERS, pin_memory
    )

    # ============================================================================
    # ÉTAPE 3: CONFIGURATION OPTIMISEUR
    # ============================================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 3: Configuration optimiseur")
    print("=" * 70)

    criterion, optimizer, scheduler = setup_optimizer_and_loss(
        model, LEARNING_RATE, WEIGHT_DECAY, class_weights_tensor
    )

    # ============================================================================
    # ÉTAPE 4: ENTRÂINEMENT OU CHARGEMENT
    # ============================================================================

    if TRAIN_MODEL:
        print("\n" + "=" * 70)
        print("ÉTAPE 4: Entraînement (avec F1-score pour early stopping)")
        print("=" * 70)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_to_idx=class_to_idx,
            num_epochs=NUM_EPOCHS,
            binary_mode=BINARY_CLASSIFICATION,
            patience=PATIENCE,
            save_path=MODEL_CHECKPOINT_PATH,
            early_stopping_metric='f1',  # Utiliser F1-score pour early stopping
        )
    else:
        print("\n" + "=" * 70)
        print("ÉTAPE 4: Chargement du modèle sauvegardé")
        print("=" * 70)

        checkpoint_path = Path(MODEL_CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Le fichier de checkpoint n'existe pas: {checkpoint_path}\n"
                f"Veuillez entraîner le modèle d'abord (TRAIN_MODEL = True dans training_config.py)"
            )

        model = load_model(
            model_choice=MODEL_CHOICE,
            num_classes=NUM_CLASSES,
            checkpoint_path=checkpoint_path,
            device=device,
            transfer_mode=TRANSFER_MODE,
        )

        history = create_empty_history(BINARY_CLASSIFICATION)
        # Ajouter val_f1 à l'historique vide pour compatibilité
        if BINARY_CLASSIFICATION:
            history['val_f1'] = []
        print(f" Modèle chargé depuis: {checkpoint_path}")

    # ============================================================================
    # ÉTAPE 5: ÉVALUATION INITIALE
    # ============================================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 5: Évaluation (seuil 0.5)")
    print("=" * 70)

    results = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        class_to_idx=class_to_idx,
        binary_mode=BINARY_CLASSIFICATION,
        class_names=class_names,
        threshold=0.5,
    )

    print_classification_report(results, binary_mode=BINARY_CLASSIFICATION, class_names=class_names)

    # ============================================================================
    # ÉTAPE 6: ANALYSE DES SEUILS (Binaire uniquement)
    # ============================================================================

    if BINARY_CLASSIFICATION:
        print("\n" + "=" * 70)
        print("ÉTAPE 6: Analyse des seuils")
        print("=" * 70)

        threshold_analysis = analyze_thresholds(results, default_threshold=0.5)

        # Trouver le seuil optimal
        optimal_f1 = find_optimal_threshold(results, metric='f1')
        optimal_balanced = find_optimal_threshold(results, metric='balanced')
        optimal_precision = find_optimal_threshold(
            results, metric='precision', target_precision=0.7
        )

        # Choisir le seuil (par défaut: F1 optimal)
        DECISION_THRESHOLD = optimal_f1['threshold']
        print(f"\n Seuil choisi: {DECISION_THRESHOLD:.4f} (F1-score optimal)")

        # Réévaluer avec le seuil optimal
        print("\n" + "=" * 70)
        print("ÉTAPE 7: Évaluation avec seuil optimal")
        print("=" * 70)

        results_optimal = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            class_to_idx=class_to_idx,
            binary_mode=BINARY_CLASSIFICATION,
            class_names=class_names,
            threshold=DECISION_THRESHOLD,
        )

        print_classification_report(
            results_optimal,
            binary_mode=BINARY_CLASSIFICATION,
            class_names=class_names,
            threshold=DECISION_THRESHOLD,
        )

        # Comparaison
        print("\n" + "=" * 70)
        print("COMPARAISON: Seuil 0.5 vs Seuil Optimal")
        print("=" * 70)

        results_default = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            class_to_idx=class_to_idx,
            binary_mode=BINARY_CLASSIFICATION,
            class_names=class_names,
            threshold=0.5,
        )

        print(
            f"\n{'Métrique':<30s} {'Seuil 0.5':<15s} {'Seuil Optimal':<15s} {'Amélioration':<15s}"
        )
        print("-" * 75)

        metrics = ['precision', 'recall', 'f1']
        for metric in metrics:
            val_default = results_default[metric][1]
            val_optimal = results_optimal[metric][1]
            print(
                f"{metric.capitalize() + ' COVID':<30s} {val_default:<15.4f} "
                f"{val_optimal:<15.4f} {val_optimal - val_default:>+15.4f}"
            )

        cm_default = results_default['confusion_matrix']
        cm_optimal = results_optimal['confusion_matrix']
        fp_default, fp_optimal = cm_default[0, 1], cm_optimal[0, 1]
        fn_default, fn_optimal = cm_default[1, 0], cm_optimal[1, 0]

        print(
            f"{'Faux Positifs (FP)':<30s} {fp_default:<15d} {fp_optimal:<15d} "
            f"{fp_optimal - fp_default:>+15d}"
        )
        print(
            f"{'Faux Négatifs (FN)':<30s} {fn_default:<15d} {fn_optimal:<15d} "
            f"{fn_optimal - fn_default:>+15d}"
        )

    # ============================================================================
    # ÉTAPE 8: VISUALISATION
    # ============================================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 8: Visualisation")
    print("=" * 70)

    plot_training_history(history)

    if BINARY_CLASSIFICATION:
        results_to_plot = results_optimal if 'results_optimal' in locals() else results
        if results_to_plot.get('probabilities') is not None:
            plot_precision_recall_curve(results_to_plot)
            plot_roc_curve(results_to_plot)

    # ============================================================================
    # RÉSUMÉ
    # ============================================================================

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f" Modèle: {MODEL_CHOICE}")
    if 'val_f1' in history and len(history['val_f1']) > 0:
        print(
            f" Meilleure validation F1-score: {max(history['val_f1']):.4f} (métrique d'early stopping)"
        )
        print(f"   Accuracy finale: {history['val_acc'][-1]:.4f}")
    elif 'val_acc' in history and len(history['val_acc']) > 0:
        print(f" Meilleure validation accuracy: {max(history['val_acc']):.4f}")
    if BINARY_CLASSIFICATION and 'DECISION_THRESHOLD' in locals():
        print(f" Seuil optimal: {DECISION_THRESHOLD:.4f}")
        if 'results_optimal' in locals():
            print(f"   Precision: {results_optimal['precision'][1]:.4f}")
            print(f"   Recall: {results_optimal['recall'][1]:.4f}")
            print(f"   F1-score: {results_optimal['f1'][1]:.4f}")

    print("\n Script terminé (version F1-score)")


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

if __name__ == '__main__':
    main()
