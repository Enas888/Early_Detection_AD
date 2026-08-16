#!/usr/bin/env python3
"""
experiment_utils.py

Shared utilities for classical ML experiments.

Responsibilities:
    1. Task label selection
    2. Classification metrics
    3. Confusion matrix plotting
    4. CV result aggregation
    5. Result saving

IMPORTANT:
    Test data is NOT handled here.
    These utilities are usable for CV and, later, final test
    evaluation.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# TASK DEFINITIONS
# ============================================================

TASK_CLASSES = {

    "cn_vs_ad": [
        "CN",
        "AD",
    ],

    "cn_vs_emci": [
        "CN",
        "EMCI",
    ],

    "cn_vs_lmci": [
        "CN",
        "LMCI",
    ],

    "four_class": [
        "CN",
        "EMCI",
        "LMCI",
        "AD",
    ],
}


def get_task_classes(task_name: str):
    """
    Return ordered class names for a task.
    """

    if task_name not in TASK_CLASSES:
        raise ValueError(
            f"Unknown task: {task_name}"
        )

    return TASK_CLASSES[task_name]


# ============================================================
# FILTER TASK
# ============================================================

# Full class list — must match build_dataset.py label_map order
ALL_CLASSES = ['CN', 'EMCI', 'LMCI', 'AD']

def filter_task(X, y, task_name: str):
    """
    Filter X and y to only include samples from the task classes.
    y contains integers mapped from ALL_CLASSES order.
    Remaps labels to 0, 1, ... for the task.

    Example — cn_vs_ad:
        CN=0, AD=3 in full label_map
        After filter: CN→0, AD→1
    """
    task_classes = get_task_classes(task_name)

    # Convert class names to their integer labels
    task_labels = [ALL_CLASSES.index(cls) for cls in task_classes]

    # Filter to task samples only
    y = np.asarray(y)
    mask = np.isin(y, task_labels)
    X_task = X[mask]
    y_task = y[mask]

    # Remap to 0, 1, ... for this task
    # WHY: classifiers and metrics expect contiguous 0-based labels
    label_remap = {old: new for new, old in enumerate(task_labels)}
    y_task = np.array([label_remap[lbl] for lbl in y_task])

    return X_task, y_task

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob=None,
    task_name: Optional[str] = None,
):
    """
    Calculate classification metrics.

    Metrics:
        - accuracy
        - balanced accuracy
        - F1
        - ROC-AUC

    Binary:
        F1 = binary F1
        ROC-AUC = positive-class ROC-AUC

    Multiclass:
        F1 = macro F1
        ROC-AUC = macro one-vs-rest ROC-AUC
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = len(np.unique(y_true))

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # Balanced accuracy
    # --------------------------------------------------------

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if n_classes == 2:

        f1 = f1_score(
            y_true,
            y_pred,
            average="binary",
        )

    else:

        f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
        )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = np.nan

    if y_prob is not None:

        try:

            if n_classes == 2:

                # y_prob can be:
                #
                #   shape (N,)
                #
                # or:
                #
                #   shape (N, 2)

                if y_prob.ndim == 2:

                    positive_prob = y_prob[:, 1]

                else:

                    positive_prob = y_prob

                roc_auc = roc_auc_score(
                    y_true,
                    positive_prob,
                )

            else:

                roc_auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )

        except ValueError:

            # This can happen if a fold contains only one class.
            # Stratified K-Fold should normally prevent this when
            # sufficient samples are available.
            roc_auc = np.nan

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(y_true, y_pred, class_names,
                           output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # WHY integer labels: y_true/y_pred are 0-based integers after remap
    int_labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=int_labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,   # show string names on axes
    )
    display.plot(ax=ax, values_format='d', cmap='Blues', colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# SAVE CV RESULTS
# ============================================================

def save_cv_results(fold_metrics, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold_idx, metrics in enumerate(fold_metrics, start=1):
        rows.append({"fold": fold_idx, **metrics})

    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    mean_row = {"fold": "mean"}
    std_row  = {"fold": "std"}

    for metric in metric_names:
        values = np.array([m[metric] for m in fold_metrics], dtype=float)
        mean_row[metric] = np.nanmean(values)
        std_row[metric]  = np.nanstd(values)

    rows.append(mean_row)
    rows.append(std_row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cv_results.csv", index=False)

    # ── Print mean to terminal ────────────────────────────────
    print(f"      CV Results (mean ± std):")
    for metric in metric_names:
        values = np.array([m[metric] for m in fold_metrics], dtype=float)
        print(f"        {metric:20s}: "
              f"{np.nanmean(values):.4f} ± {np.nanstd(values):.4f}")

    return df


# ============================================================
# SAVE METADATA
# ============================================================

def save_experiment_summary(
    output_dir,
    experiment_id,
    task_name,
    classifier_name,
    fusion_type,
):
    """
    Save basic experiment metadata.
    """

    output_dir = Path(output_dir)

    summary = {
        "experiment_id": experiment_id,
        "task": task_name,
        "classifier": classifier_name,
        "fusion_type": fusion_type,
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_dir / "experiment_summary.csv",
        index=False,
    )