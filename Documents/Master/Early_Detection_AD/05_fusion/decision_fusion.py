#!/usr/bin/env python3
"""
decision_fusion.py

Decision-level fusion for multi-scale features.

A separate classifier is trained for each scale.

Example:

    narrow   → RF
    original → RF
    wide     → RF

Each classifier produces probabilities.

The probabilities are averaged:

    P_final =
        (P_narrow +
         P_original +
         P_wide) / 3

The final class is selected from P_final.

IMPORTANT:
    This module does not perform CV itself.
    The experiment runner controls the CV folds.
"""

import numpy as np


def average_probabilities(
    probabilities,
):
    """
    Average predicted probabilities from multiple scales.

    Parameters
    ----------
    probabilities:
        List of arrays.

        Binary:
            each array shape = (N, 2)

        Multiclass:
            each array shape = (N, C)

    Returns
    -------
    averaged:
        Same shape as an individual probability array.
    """

    if len(probabilities) == 0:

        raise ValueError(
            "No probability arrays supplied."
        )

    arrays = [
        np.asarray(p)
        for p in probabilities
    ]

    reference_shape = arrays[0].shape

    for idx, arr in enumerate(arrays):

        if arr.shape != reference_shape:

            raise ValueError(
                "Probability shapes do not match. "
                f"Array 0: {reference_shape}, "
                f"array {idx}: {arr.shape}"
            )

    return np.mean(
        np.stack(
            arrays,
            axis=0,
        ),
        axis=0,
    )


def probabilities_to_predictions(probabilities, classes):
    """
    Convert averaged probabilities into integer class predictions.
    Returns 0-based integers matching remapped y labels.
    WHY integers: calculate_metrics and balanced_accuracy_score
    expect the same type as y_true (integers after filter_task remap).
    """
    probabilities = np.asarray(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected probability matrix, got {probabilities.shape}")

    # Return integer indices directly — not class name strings
    return np.argmax(probabilities, axis=1).astype(int)


def decision_fusion_predict(models, scale_features, scales, classes):
    probabilities = []

    for scale in scales:
        probability = models[scale].predict_proba(scale_features[scale])
        probabilities.append(probability)

    averaged = average_probabilities(probabilities)

    # Returns integers, not strings
    predictions = probabilities_to_predictions(averaged, classes)

    return predictions, averaged