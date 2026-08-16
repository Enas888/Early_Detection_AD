#!/usr/bin/env python3
"""
metrics.py

Central evaluation definitions.

This file provides a single consistent definition of
classification metrics for all experiments.
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
):
    """
    Calculate the four required metrics.

    Required:
        accuracy
        balanced_accuracy
        f1
        roc_auc
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_classes = np.unique(
        y_true
    )

    n_classes = len(
        unique_classes
    )

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

                if y_prob.ndim == 2:

                    positive_probability = (
                        y_prob[:, 1]
                    )

                else:

                    positive_probability = y_prob

                roc_auc = roc_auc_score(
                    y_true,
                    positive_probability,
                )

            else:

                roc_auc = roc_auc_score(
                    y_true,
                    y_prob,
                    multi_class="ovr",
                    average="macro",
                )

        except ValueError:

            roc_auc = np.nan

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }