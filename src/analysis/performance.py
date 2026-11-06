import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, 
    precision_recall_curve, confusion_matrix, classification_report
)
from scipy.special import expit # Sigmoid function
from pathlib import Path
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def find_threshold_for_recall(y_true, y_pred_logits, target_recall=0.999):
    y_pred_proba = expit(y_pred_logits)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    valid_indices = np.where(tpr >= target_recall)[0]
    
    if len(valid_indices) == 0:
        logging.info(f"Warning: Cannot achieve {target_recall*100:.1f}% recall. Using lowest threshold.")
        threshold_idx = len(thresholds) - 1
    else:
        threshold_idx = valid_indices[0]
    
    threshold = thresholds[threshold_idx]
    actual_recall = tpr[threshold_idx]
    actual_fpr = fpr[threshold_idx]
    
    return threshold, actual_recall


def compute_metrics(y_true, y_pred_logits, output_dir, threshold=None, target_recall=0.999):
    y_pred_proba = expit(y_pred_logits)
    
    if threshold is None:
        threshold, actual_recall = find_threshold_for_recall(y_true, y_pred_logits, target_recall)
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC (AP)': pr_auc,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
    }
    
    output_path = Path(output_dir)
    with open(output_path / 'performance_metrics.txt', 'w') as f:
        f.write("BINARY CLASSIFICATION METRICS\n")
        f.write(f"Classification Threshold: {threshold:.6f}\n")
        if threshold is not None:
            f.write(f"(Selected to preserve {target_recall*100:.1f}% recall)\n\n")
        else:
            f.write("\n")
        for metric, value in metrics.items():
            if isinstance(value, int):
                f.write(f"{metric:25s}: {value}\n")
            else:
                f.write(f"{metric:25s}: {value:.6f}\n")
        
        f.write(classification_report(y_true, y_pred_binary, 
                                     target_names=['Fake Track', 'Real Track']))
    
    logging.info(f"\nSaved: {output_path / 'performance_metrics.txt'}")
    return metrics, threshold


def plot_roc_curve(y_true, y_pred_logits, output_dir, highlight_threshold=None, target_recall=0.999):
    logging.info("\nGenerating ROC curve...")
    
    y_pred_proba = expit(y_pred_logits)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    if highlight_threshold is None:
        highlight_threshold, _ = find_threshold_for_recall(y_true, y_pred_logits, target_recall)
    
    threshold_idx = np.argmin(np.abs(thresholds - highlight_threshold))
    
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(fpr, 1 - tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    
    fpr_random = np.linspace(0.001, 1, 100)  # Avoid log(0)
    fnr_random = 1 - fpr_random
    ax.plot(fpr_random, fnr_random, color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.plot(fpr[threshold_idx], 1 - tpr[threshold_idx], 'ro', markersize=12,
            label=f'Target {target_recall*100:.1f}% TPR (Threshold = {highlight_threshold:.4f})')
    
    ax.plot(fpr[optimal_idx], 1 - tpr[optimal_idx], 'gs', markersize=10,
            label=f'Youden Optimal (Threshold = {optimal_threshold:.4f})', alpha=0.6)
    
    ax.set_xlim([0.0, 1.0])
    # ax.set_ylim([0.0, 1.05])
    ax.set_yscale('log')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('log(1 - True Positive Rate)', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'roc_curve.png'}")
    logging.info(f"Selected threshold for {target_recall*100:.1f}% recall: {highlight_threshold:.6f}")
    logging.info(f"Youden's optimal threshold: {optimal_threshold:.4f}")
    
    return roc_auc, highlight_threshold


def plot_precision_recall_curve(y_true, y_pred_logits, output_dir, highlight_threshold=None, target_recall=0.999):
    logging.info("\nGenerating Precision-Recall curve...")
    
    y_pred_proba = expit(y_pred_logits)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    if highlight_threshold is None:
        highlight_threshold, _ = find_threshold_for_recall(y_true, y_pred_logits, target_recall)
    
    threshold_idx = np.argmin(np.abs(thresholds - highlight_threshold))
    
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    baseline = y_true.sum() / len(y_true)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {pr_auc:.4f})')
    ax.axhline(y=baseline, color='navy', lw=2, linestyle='--',
              label=f'Baseline (Prevalence = {baseline:.4f})')
    
    ax.plot(recall[threshold_idx], precision[threshold_idx], 'ro', markersize=12,
            label=f'Target {target_recall*100:.1f}% Recall (Threshold = {highlight_threshold:.4f})')
    
    ax.plot(recall[optimal_idx], precision[optimal_idx], 'gs', markersize=10,
            label=f'Max F1 (Threshold = {optimal_threshold:.4f})', alpha=0.6)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'precision_recall_curve.png'}")
    logging.info(f"Selected threshold for {target_recall*100:.1f}% recall: {highlight_threshold:.6f}")
    logging.info(f"Optimal F1 threshold: {optimal_threshold:.4f}")
    
    return pr_auc, highlight_threshold


def plot_confusion_matrix(y_true, y_pred_logits, output_dir, threshold=None, target_recall=0.999):
    logging.info("\nGenerating confusion matrix...")
    
    y_pred_proba = expit(y_pred_logits)
    
    if threshold is None:
        threshold, _ = find_threshold_for_recall(y_true, y_pred_logits, target_recall)
    
    y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred_binary)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['Fake Track (0)', 'Real Track (1)']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=16)
    
    ax.set_title(f'Normalized Confusion Matrix (Threshold = {threshold:.6f}, Target {target_recall*100:.1f}% Recall)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'confusion_matrix.png'}")


def plot_prediction_distribution(y_true, y_pred_logits, output_dir, threshold=None, target_recall=0.999):
    logging.info("\nGenerating prediction distribution plot...")
    
    y_pred_proba = expit(y_pred_logits)
    
    if threshold is None:
        threshold, _ = find_threshold_for_recall(y_true, y_pred_logits, target_recall)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    fake_tracks = y_pred_proba[y_true == 0]
    real_tracks = y_pred_proba[y_true == 1]
    
    axes[0].hist(fake_tracks, bins=50, alpha=0.7, label='Fake Tracks', 
                 color='red', edgecolor='black')
    axes[0].hist(real_tracks, bins=50, alpha=0.7, label='Real Tracks', 
                 color='green', edgecolor='black')
    axes[0].axvline(x=threshold, color='black', linestyle='--', linewidth=2, 
                    label=f'Threshold = {threshold:.4f} ({target_recall*100:.1f}% recall)')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Predicted Probabilities', 
                      fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].hist([fake_tracks, real_tracks], bins=50, alpha=0.7, 
                 label=['Fake Tracks', 'Real Tracks'],
                 color=['red', 'green'], edgecolor='black', 
                 stacked=True, density=True)
    axes[1].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                    label=f'Threshold = {threshold:.4f} ({target_recall*100:.1f}% recall)')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Stacked Probability Distribution (Normalized)', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'prediction_distribution.png'}")


def plot_calibration(y_true, y_pred_logits, output_dir, n_bins=20):
    logging.info("\nGenerating calibration curve...")
    
    y_pred_proba = expit(y_pred_logits)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    mean_pred = []
    mean_true = []
    counts = []
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include the last edge
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
        
        if mask.sum() > 0:
            mean_pred.append(y_pred_proba[mask].mean())
            mean_true.append(y_true[mask].mean())
            counts.append(mask.sum())
        else:
            mean_pred.append(bin_centers[i])
            mean_true.append(bin_centers[i])
            counts.append(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
    axes[0].plot(mean_pred, mean_true, 'bo-', linewidth=2, markersize=8, label='Model Calibration')
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Fraction of Positives (True)', fontsize=12)
    axes[0].set_title('Calibration Curve', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    axes[1].bar(range(n_bins), counts, alpha=0.7, edgecolor='black', label='Sample Count')
    axes[1].set_xlabel('Bin Index', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Samples per Bin', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir)
    plt.savefig(output_path / 'calibration_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path / 'calibration_plot.png'}")

def plot_loss(logdir, outdir):
    event_files = [
        os.path.join(logdir, f)
        for f in os.listdir(logdir)
        if "events.out.tfevents" in f
    ]

    if len(event_files) == 0:
        raise RuntimeError(f"No TensorBoard event files found in {logdir}")

    print("Using:")
    for f in event_files:
        print("  ", f)

    # Load TensorBoard data
    ea = EventAccumulator(event_files[0])
    ea.Reload()

    print("\nAvailable scalar metrics:")
    print(ea.Tags()["scalars"])

    # Get epoch-level training loss
    train_events = ea.Scalars("train_loss_epoch")

    # Get validation loss
    val_events = ea.Scalars("val_loss")

    print("Train:")
    for event in train_events[:10]:
        print(event.step, event.value)

    print("Validation:")
    for event in val_events[:10]:
        print(event.step, event.value)

    # Extract values
    train_epochs = range(len(train_events))
    val_epochs = range(len(val_events))
    # train_epochs = [event.step for event in train_events]
    # train_loss = [event.value for event in train_events]
    train_loss = [event.value for event in train_events]

    # val_epochs = [event.step for event in val_events]
    val_loss = [event.value for event in val_events]

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        train_epochs,
        train_loss,
        label="Train",
    )

    ax.plot(
        val_epochs,
        val_loss,
        label="Validation",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "loss.png"))

    ax.set_yscale("log")
    fig.savefig(os.path.join(outdir, "loss_log.png"))


# def plot_prediction_distribution_vertex_cut(
#     y_true,
#     y_pred_logits,
#     X_test,
#     output_dir,
#     feature_names,
#     vertex_cut,
#     threshold=None,
#     target_recall=0.999,
# ):
#     logging.info(
#         f"\nGenerating prediction distribution plot for dxyClosestPV > {vertex_cut} "
#     )

#     # Convert logits → probabilities
#     y_pred_proba = expit(y_pred_logits)

#     # Apply vertex cut
#     vertex_index = feature_names.index("trk_dxyClosestPV")
#     vertex_values = X_test[:, vertex_index]
#     mask = vertex_values > vertex_cut

#     if mask.sum() == 0:
#         logging.warning("No samples pass the vertex cut — skipping plot.")
#         return

#     y_true_cut = y_true[mask]
#     y_pred_proba_cut = y_pred_proba[mask]
#     y_pred_logits_cut = y_pred_logits[mask]

#     # Determine threshold if not provided
#     if threshold is None:
#         threshold, _ = find_threshold_for_recall(
#             y_true_cut, y_pred_logits_cut, target_recall
#         )

#     # Separate classes
#     fake_tracks = y_pred_proba_cut[y_true_cut == 0]
#     real_tracks = y_pred_proba_cut[y_true_cut == 1]

#     fig, axes = plt.subplots(2, 1, figsize=(12, 10))

#     # --- Top panel: unstacked ---
#     axes[0].hist(
#         fake_tracks,
#         bins=50,
#         alpha=0.7,
#         label="Fake Tracks",
#         color="red",
#         edgecolor="black",
#     )
#     axes[0].hist(
#         real_tracks,
#         bins=50,
#         alpha=0.7,
#         label="Real Tracks",
#         color="green",
#         edgecolor="black",
#     )
#     axes[0].axvline(
#         x=threshold,
#         color="black",
#         linestyle="--",
#         linewidth=2,
#         label=f"Threshold = {threshold:.4f} ({target_recall*100:.1f}% recall)",
#     )
#     axes[0].set_xlabel("Predicted Probability", fontsize=12)
#     axes[0].set_ylabel("Frequency", fontsize=12)
#     axes[0].set_title(
#         f"Distribution of Predicted Probabilities (dxyClosestPV > {vertex_cut})",
#         fontsize=14,
#         fontweight="bold",
#     )
#     axes[0].legend(fontsize=10)
#     axes[0].grid(True, alpha=0.3, axis="y")

#     # --- Bottom panel: stacked & normalized ---
#     axes[1].hist(
#         [fake_tracks, real_tracks],
#         bins=50,
#         alpha=0.7,
#         label=["Fake Tracks", "Real Tracks"],
#         color=["red", "green"],
#         edgecolor="black",
#         stacked=True,
#         density=True,
#     )
#     axes[1].axvline(
#         x=threshold,
#         color="black",
#         linestyle="--",
#         linewidth=2,
#         label=f"Threshold = {threshold:.4f} ({target_recall*100:.1f}% recall)",
#     )
#     axes[1].set_xlabel("Predicted Probability", fontsize=12)
#     axes[1].set_ylabel("Density", fontsize=12)
#     axes[1].set_title(
#         f"Stacked Probability Distribution (dxyClosestPV > {vertex_cut}) (Normalized)",
#         fontsize=14,
#         fontweight="bold",
#     )
#     axes[1].legend(fontsize=10)
#     axes[1].grid(True, alpha=0.3, axis="y")

#     plt.tight_layout()
#     output_path = Path(output_dir)
#     out_file = output_path / f"prediction_distribution_dxyClosestPV_gt_{vertex_cut}.png"
#     plt.savefig(out_file, dpi=300, bbox_inches="tight")
#     plt.close()

#     logging.info(f"Saved: {output_path / 'prediction_distribution_vertex_gt_{vertex_cut}.png'}")