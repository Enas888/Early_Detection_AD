#!/usr/bin/env python3
"""
feature_fusion.py

Feature-level fusion for multi-scale GLCM features.

Each scale currently contains 6 GLCM features:

    narrow   → 6
    original → 6
    wide     → 6

Feature-level fusion concatenates them:

    [narrow | original | wide]

Result:

    18-dimensional vector
"""

import numpy as np


def concatenate_scales(
    scale_features,
    scales=("narrow", "original", "wide"),
):
    """
    Concatenate feature vectors from multiple scales.

    Parameters
    ----------
    scale_features:
        Dictionary:

            {
                "narrow": X_narrow,
                "original": X_original,
                "wide": X_wide
            }

        Each X has shape:

            (N_samples, N_features)

    scales:
        Order in which scales are concatenated.

    Returns
    -------
    X_fused:
        Shape:

            (N_samples, N_features * N_scales)
    """

    arrays = []

    for scale in scales:

        if scale not in scale_features:
            raise KeyError(
                f"Missing scale: {scale}"
            )

        X = np.asarray(
            scale_features[scale]
        )

        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D feature matrix for "
                f"{scale}, got {X.shape}"
            )

        arrays.append(X)

    # --------------------------------------------------------
    # Check sample alignment.
    # --------------------------------------------------------

    n_samples = arrays[0].shape[0]

    for scale, X in zip(scales, arrays):

        if X.shape[0] != n_samples:

            raise ValueError(
                "Scale sample counts do not match. "
                f"{scale} has {X.shape[0]}, "
                f"expected {n_samples}."
            )

    # --------------------------------------------------------
    # Concatenate along feature axis.
    # --------------------------------------------------------

    return np.concatenate(
        arrays,
        axis=1,
    )


def feature_fusion(
    X_narrow,
    X_original,
    X_wide,
):
    """
    Convenience wrapper for the three GLCM scales.
    """

    return concatenate_scales(
        {
            "narrow": X_narrow,
            "original": X_original,
            "wide": X_wide,
        }
    )