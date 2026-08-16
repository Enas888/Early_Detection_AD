#!/usr/bin/env python3
"""
extract_glcm.py

============================================================
2D GLCM TEXTURE FEATURE EXTRACTION
============================================================

PURPOSE
-------
Extract 2D GLCM texture features from the complete rectangular
VOIs generated during VOI extraction.

IMPORTANT
---------
GLCM is calculated from the ENTIRE rectangular VOI/BBOX.

The ROI mask is NOT used to filter the voxels.

The atlas ROI is used earlier during VOI extraction to determine
the anatomical bounding box. The bounding box is then scaled:

    narrow   = 0.8 × original ROI bounding box
    original = 1.0 × original ROI bounding box
    wide     = 1.2 × original ROI bounding box

Each scale is therefore a complete rectangular 3D VOI.

For every VOI:

    3D VOI
       ↓
    axial slices
       ↓
    2D GLCM per slice
       ↓
    5 texture features per slice
       ↓
    mean pooling across slices
       ↓
    final 5-dimensional feature vector


============================================================
GLCM CONFIGURATION
============================================================

Gray levels : 32
Distance    : 1 voxel
Angle       : 0 degrees

The same configuration is used for all experiments so that
classical and quantum models receive exactly the same
handcrafted GLCM representation.


============================================================
FEATURES
============================================================

Five GLCM features are retained:

    1. Contrast
    2. Dissimilarity
    3. Homogeneity
    4. Energy
    5. Correlation

ASM is excluded because it is mathematically related to
Energy:

    Energy = sqrt(ASM)

Entropy is also excluded to keep the representation compact
and to maintain exactly five consistent GLCM features.


============================================================
OUTPUT
============================================================

Features are saved as:

    01_data/processed/features_affine/glcm/

        narrow/
            CN/
            EMCI/
            LMCI/
            AD/

        original/
            CN/
            EMCI/
            LMCI/
            AD/

        wide/
            CN/
            EMCI/
            LMCI/
            AD/

Each file contains:

    shape = (5,)

Example:

    153_S_4077_257401.npy
"""


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


# ============================================================
# GLCM CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Five selected GLCM properties.
#
# These are the ONLY features extracted.
# ------------------------------------------------------------

GLCM_PROPERTIES = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "correlation",
]


# ------------------------------------------------------------
# One spatial distance.
#
# Distance = 1 means each voxel is compared with its
# immediate neighboring voxel.
# ------------------------------------------------------------

GLCM_DISTANCES = [1]


# ------------------------------------------------------------
# One spatial direction.
#
# 0 radians = 0 degrees.
#
# This corresponds to the horizontal direction inside
# each axial slice.
# ------------------------------------------------------------

GLCM_ANGLES = [0.0]


# ------------------------------------------------------------
# Number of gray levels.
#
# 32 levels provide a reasonable methodological balance
# between:
#
#   - preserving intensity/texture information
#   - avoiding excessive GLCM sparsity
#   - computational efficiency
#   - feature stability
#
# The same value is used for all experiments.
# ------------------------------------------------------------

GLCM_LEVELS = 32


# ============================================================
# CONFIGURATION
# ============================================================

def load_config():
    """
    Load the central project configuration.

    Project structure:

        Early_Detection_AD/
        │
        ├── config/
        │   └── data_config.yaml
        │
        └── 03_feature_extraction/
            └── glcm/
                └── extract_glcm.py

    parents[2] points to the project root.
    """

    project_root = Path(__file__).resolve().parents[2]

    config_path = (
        project_root
        / "config"
        / "data_config.yaml"
    )

    if not config_path.exists():

        raise FileNotFoundError(
            f"\nConfiguration file not found:\n"
            f"{config_path}\n"
        )

    with open(config_path, "r") as f:

        return yaml.safe_load(f)


# ============================================================
# QUANTIZATION
# ============================================================

def quantize_slice(
    slice_2d,
    n_levels=GLCM_LEVELS
):
    """
    Convert a normalized MRI slice from [0, 1] into
    discrete integer gray levels.

    GLCM requires discrete integer values.

    Input
    -----
    slice_2d : normalized 2D MRI slice

    Output
    ------
    uint8 array with values:

        0 ... n_levels - 1
    """

    # Ensure all values are inside the expected range.
    slice_2d = np.clip(
        slice_2d,
        0.0,
        1.0
    )

    # Convert continuous intensity to discrete levels.
    slice_q = (
        slice_2d * (n_levels - 1)
    ).astype(np.uint8)

    return slice_q


# ============================================================
# GLCM FROM ONE AXIAL SLICE
# ============================================================

def extract_glcm_from_slice(slice_2d):
    """
    Calculate GLCM features from the ENTIRE rectangular
    axial VOI slice.

    IMPORTANT
    ---------
    No ROI mask is used.

    Therefore every voxel/pixel inside the rectangular
    VOI contributes to the GLCM.

    Returns
    -------
    np.ndarray
        Five-dimensional feature vector:

        [contrast,
         dissimilarity,
         homogeneity,
         energy,
         correlation]
    """

    # --------------------------------------------------------
    # Quantize complete VOI slice.
    # --------------------------------------------------------

    slice_q = quantize_slice(
        slice_2d
    )

    # --------------------------------------------------------
    # Skip completely empty slices.
    #
    # This prevents meaningless GLCM computation on a slice
    # containing only zeros.
    # --------------------------------------------------------

    if np.all(slice_q == 0):

        return None

    # --------------------------------------------------------
    # Build 2D GLCM.
    #
    # distance = 1
    # angle    = 0°
    #
    # symmetric=True:
    # treats paired gray-level relationships symmetrically.
    #
    # normed=True:
    # converts GLCM counts into probabilities.
    # --------------------------------------------------------

    glcm = graycomatrix(
        slice_q,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True
    )

    # --------------------------------------------------------
    # Extract the five properties.
    #
    # Because there is only:
    #
    #     1 distance × 1 angle
    #
    # each property produces exactly one scalar.
    # --------------------------------------------------------

    features = []

    # Five standard GLCM properties
    for property_name in GLCM_PROPERTIES:

        value = graycoprops(
            glcm,
            property_name
        )[0, 0]

        features.append(value)

    # --------------------------------------------------------
    # GLCM ENTROPY
    # --------------------------------------------------------
    #
    # Entropy measures the randomness/complexity of the
    # gray-level co-occurrence distribution.
    #
    # Higher entropy → more complex/random texture
    # Lower entropy  → more uniform/regular texture
    #
    # --------------------------------------------------------

    p = glcm[:, :, 0, 0].astype(np.float64)

    entropy = -np.sum(
        p * np.log2(p + 1e-12)
    )

    features.append(entropy)

    return np.asarray(
        features,
        dtype=np.float32
    )

# ============================================================
# GLCM FROM COMPLETE 3D VOI
# ============================================================

def extract_glcm_from_voi(voi):
    """
    Extract 2D GLCM features from every axial slice of
    the complete rectangular 3D VOI.

    VOI shape:

        W × H × D

    Therefore:

        voi[:, :, z]

    is one axial slice.

    Each valid slice produces:

        5 features

    The features are then averaged across all valid axial
    slices.

    Final output:

        shape = (5,)
    """

    slice_features = []

    # --------------------------------------------------------
    # Traverse the complete depth of the VOI.
    # --------------------------------------------------------

    for z in range(voi.shape[2]):

        # Extract one complete rectangular axial slice.
        slice_2d = voi[:, :, z]

        # GLCM is calculated over the ENTIRE slice.
        features = extract_glcm_from_slice(
            slice_2d
        )

        if features is not None:

            slice_features.append(
                features
            )

    # --------------------------------------------------------
    # No valid slices.
    # --------------------------------------------------------

    if len(slice_features) == 0:

        return None

    # --------------------------------------------------------
    # Convert list to matrix:
    #
    #     number_of_slices × 5
    # --------------------------------------------------------

    slice_features = np.stack(
        slice_features,
        axis=0
    )

    # --------------------------------------------------------
    # Mean pooling across axial slices.
    #
    # Example:
    #
    # Slice 1 → [C1, D1, H1, E1, R1]
    # Slice 2 → [C2, D2, H2, E2, R2]
    # ...
    #
    # Final:
    #
    # [mean(C),
    #  mean(D),
    #  mean(H),
    #  mean(E),
    #  mean(R)]
    # --------------------------------------------------------

    pooled_features = np.mean(
        slice_features,
        axis=0
    )

    return pooled_features.astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("2D GLCM TEXTURE FEATURE EXTRACTION")
    print("=" * 75)

    # ========================================================
    # 1. LOAD CONFIG
    # ========================================================

    cfg = load_config()

    csv_dir = Path(
        cfg["paths"]["csv_dir"]
    )

    processed_dir = Path(
        cfg["paths"]["processed_dir"]
    )

    classes = cfg[
        "classes"
    ][
        "target"
    ]

    scales = cfg[
        "voi"
    ][
        "scales"
    ]

    # ========================================================
    # 2. INPUT VOI DIRECTORY
    # ========================================================

    # With the new data_config:
    #
    # processed_dir =
    #
    # 01_data/processed/classes_affine
    #
    # therefore VOIs are:
    #
    # 01_data/processed/classes_affine/voi
    # ========================================================

    voi_dir = (
        processed_dir
        / "voi"
    )

    # ========================================================
    # 3. OUTPUT FEATURE DIRECTORY
    # ========================================================

    # processed_dir.parent =
    #
    # 01_data/processed
    #
    # Therefore output becomes:
    #
    # 01_data/processed/features_affine/glcm
    # ========================================================

    feature_dir = (
        processed_dir.parent
        / "features_affine"
        / "glcm"
    )

    # ========================================================
    # 4. LOAD METADATA
    # ========================================================

    metadata_path = (
        csv_dir
        / cfg["paths"]["metadata_csv"]
    )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"\nMetadata CSV not found:\n"
            f"{metadata_path}\n"
        )

    meta = pd.read_csv(
        metadata_path
    )

    # ========================================================
    # 5. PRINT CONFIGURATION
    # ========================================================

    print("\nConfiguration")
    print("-" * 75)

    print(
        f"Input MRI visits : {len(meta)}"
    )

    print(
        f"Unique subjects  : "
        f"{meta['subject_id'].nunique()}"
    )

    print(
        f"VOI directory    : {voi_dir}"
    )

    print(
        f"Feature output    : {feature_dir}"
    )

    print(
        f"Scales            : "
        f"{list(scales.keys())}"
    )

    print(
        f"GLCM levels       : {GLCM_LEVELS}"
    )

    print(
        f"GLCM distance     : 1"
    )

    print(
        f"GLCM angle        : 0 degrees"
    )

    print(
        f"Features / slice  : {len(GLCM_PROPERTIES)}"
    )

    print(
        f"Features / VOI    : {len(GLCM_PROPERTIES)}"
    )

    # ========================================================
    # 6. CREATE OUTPUT DIRECTORIES
    # ========================================================

    for scale_name in scales:

        for class_name in classes:

            output_dir = (
                feature_dir
                / scale_name
                / class_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

    # ========================================================
    # 7. PROCESS ALL MRI VISITS
    # ========================================================

    failed = []

    saved = 0

    for _, row in tqdm(
        meta.iterrows(),
        total=len(meta),
        desc="2D GLCM extraction"
    ):

        subject_id = str(
            row["subject_id"]
        )

        image_id = str(
            row["image_id"]
        )

        class_name = str(
            row["class"]
        )

        filename = (
            f"{subject_id}_{image_id}.npy"
        )

        all_scales_ok = True

        # ====================================================
        # PROCESS ALL THREE VOI SCALES
        # ====================================================

        for scale_name in scales:

            # ------------------------------------------------
            # Input rectangular VOI
            # ------------------------------------------------

            voi_path = (
                voi_dir
                / scale_name
                / class_name
                / filename
            )

            if not voi_path.exists():

                failed.append({

                    "subject_id":
                        subject_id,

                    "image_id":
                        image_id,

                    "class":
                        class_name,

                    "scale":
                        scale_name,

                    "error":
                        "VOI file missing",

                })

                all_scales_ok = False

                continue

            try:

                # ============================================
                # LOAD VOI
                # ============================================

                voi = np.load(
                    str(voi_path)
                )

                # ============================================
                # VALIDATE VOI
                # ============================================

                if voi.ndim != 3:

                    raise ValueError(
                        f"Expected 3D VOI, "
                        f"got {voi.shape}"
                    )

                if min(voi.shape) <= 1:

                    raise ValueError(
                        f"Invalid VOI shape: "
                        f"{voi.shape}"
                    )

                # ============================================
                # EXTRACT GLCM
                #
                # Complete rectangular VOI.
                # ROI mask is NOT used.
                # ============================================

                features = extract_glcm_from_voi(
                    voi
                )

                if features is None:

                    raise ValueError(
                        "No valid axial slices"
                    )

                # ============================================
                # FINAL VALIDATION
                #
                # Must contain exactly 5 features.
                # ============================================

                if features.shape != (6,):

                    raise ValueError(
                        f"Expected feature shape "
                        f"(5,), got {features.shape}"
                    )

                # ============================================
                # OUTPUT PATH
                # ============================================

                output_path = (
                    feature_dir
                    / scale_name
                    / class_name
                    / filename
                )

                # ============================================
                # SAVE FEATURE VECTOR
                # ============================================

                np.save(
                    str(output_path),
                    features
                )

            except Exception as error:

                failed.append({

                    "subject_id":
                        subject_id,

                    "image_id":
                        image_id,

                    "class":
                        class_name,

                    "scale":
                        scale_name,

                    "error":
                        str(error),

                })

                all_scales_ok = False

        # ----------------------------------------------------
        # Count the visit as complete only when all three
        # scales were successfully processed.
        # ----------------------------------------------------

        if all_scales_ok:

            saved += 1

    # ========================================================
    # 8. SAVE FAILURE REPORT
    # ========================================================

    if failed:

        failure_path = (
            csv_dir
            / "glcm_2d_extraction_failed_affine_3t.csv"
        )

        pd.DataFrame(
            failed
        ).to_csv(
            failure_path,
            index=False
        )

    # ========================================================
    # 9. FINAL REPORT
    # ========================================================

    print("\n" + "=" * 75)
    print("2D GLCM EXTRACTION COMPLETE")
    print("=" * 75)

    print(
        f"Input MRI visits : {len(meta)}"
    )

    print(
        f"Complete visits  : {saved}"
    )

    print(
        f"Failed entries   : {len(failed)}"
    )

    print(
        f"GLCM levels      : {GLCM_LEVELS}"
    )

    print(
        f"Distance         : 1"
    )

    print(
        f"Angle            : 0 degrees"
    )

    print(
        f"Features / slice : 5"
    )

    print(
        f"Features / VOI   : 5"
    )

    print(
        f"Output directory : {feature_dir}"
    )

    print(
        "\nFeature vector:"
    )

    print(
        "  [contrast, dissimilarity, "
        "homogeneity, energy, correlation]"
    )

    print(
        "  shape = (5,)"
    )

    if failed:

        print(
            "\nFailure report:"
        )

        print(
            f"  {failure_path}"
        )

    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()