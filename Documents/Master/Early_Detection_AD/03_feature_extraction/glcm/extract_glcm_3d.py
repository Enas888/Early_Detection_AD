#!/usr/bin/env python3
"""
extract_glcm_3d.py

WHY 3D GLCM over 2D:
  Hippocampal atrophy changes tissue texture across slices (z-direction),
  not just within a single slice. 2D GLCM applied per-slice then averaged
  loses inter-slice texture gradients. 3D GLCM captures co-occurrence
  along all 13 spatial directions simultaneously — including diagonals
  that cross slice boundaries — giving a richer texture representation.

Reference:
  Tan et al. "3D-GLCM CNN: A 3-dimensional gray-level co-occurrence
  matrix-based CNN model for polyp classification via CT colonography"
  IEEE Trans Med Imaging, 2020.

Method:
  - Quantize VOI to 32 gray levels (WHY 32: best trade-off between
    GLCM sparsity and information preservation per Tan et al.)
  - Compute 13 direction GLCMs (half of 26 neighbors — opposite
    directions are redundant due to symmetry)
  - Extract 6 Haralick properties per direction
  - Aggregate: mean + std across 13 directions
  - Final feature vector: 6 properties × 2 stats = 12 dimensions

Output:
  Per subject per scale → 12-dim feature vector
  Saved to 01_data/features/glcm_3d/{scale}/{class}/{subj}_{img}.npy
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from itertools import product
from tqdm import tqdm


# ── 13 unique 3D directions (half of 26-neighbor directions) ─────────────────
# WHY these 13: they represent all unique spatial directions in 3D space
# from a voxel center to its 26 neighbors, excluding opposite duplicates.
# Each direction (dx, dy, dz) where we take only those where the first
# non-zero component is positive to avoid redundancy.

DIRECTIONS_3D = [
    # Face neighbors (6 total → 3 unique)
    (1, 0, 0),   # x-axis
    (0, 1, 0),   # y-axis
    (0, 0, 1),   # z-axis (inter-slice — new vs 2D!)

    # Edge neighbors (12 total → 6 unique)
    (1, 1, 0),
    (1,-1, 0),
    (1, 0, 1),
    (1, 0,-1),
    (0, 1, 1),
    (0, 1,-1),

    # Corner neighbors (8 total → 4 unique)
    (1, 1, 1),
    (1, 1,-1),
    (1,-1, 1),
    (1,-1,-1),
]
# Total: 3 + 6 + 4 = 13 directions ✓

N_LEVELS = 32   # WHY 32: Tan et al. experimental validation
                # Less than 32 → too sparse, loses texture detail
                # More than 32 → GLCM becomes very sparse, unreliable stats

HARALICK_PROPS = [
    'contrast',
    'dissimilarity',
    'homogeneity',
    'energy',
    'correlation',
    'ASM',
]


def load_config():
    cfg_path = Path(__file__).parent.parent.parent / "config" / "data_config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def quantize_voi(voi, n_levels=N_LEVELS):
    """
    WHY: 3D GLCM requires integer voxel values in [0, n_levels-1].
    VOI is float [0,1] after normalization.
    We clip first to handle any floating point edge cases.
    """
    voi = np.clip(voi, 0.0, 1.0)
    return (voi * (n_levels - 1)).astype(np.int32)

# Fast alternative using numpy vectorization — add after testing correctness:
def compute_3d_glcm_fast(voi_q, direction, n_levels=N_LEVELS):
    dx, dy, dz = direction
    # Slice the volume to get voxel pairs
    src = voi_q[
        max(0,-dx):min(voi_q.shape[0], voi_q.shape[0]-dx),
        max(0,-dy):min(voi_q.shape[1], voi_q.shape[1]-dy),
        max(0,-dz):min(voi_q.shape[2], voi_q.shape[2]-dz),
    ]
    tgt = voi_q[
        max(0, dx):min(voi_q.shape[0], voi_q.shape[0]+dx),
        max(0, dy):min(voi_q.shape[1], voi_q.shape[1]+dy),
        max(0, dz):min(voi_q.shape[2], voi_q.shape[2]+dz),
    ]
    glcm = np.zeros((n_levels, n_levels), dtype=np.float64)
    np.add.at(glcm, (src.ravel(), tgt.ravel()), 1)
    np.add.at(glcm, (tgt.ravel(), src.ravel()), 1)  # symmetric
    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm

def haralick_from_glcm(glcm):
    """
    WHY same Haralick properties as 2D:
    The mathematical definitions are identical — they operate on
    any normalized co-occurrence matrix regardless of whether it
    was built from 2D or 3D data.

    Returns dict of 6 scalar properties.
    """
    n = glcm.shape[0]
    i_idx, j_idx = np.mgrid[0:n, 0:n]

    # Normalize (should already be, but safe)
    p = glcm / (glcm.sum() + 1e-10)

    # Mean and std of marginal distributions
    mu_i = np.sum(i_idx * p)
    mu_j = np.sum(j_idx * p)
    std_i = np.sqrt(np.sum(p * (i_idx - mu_i)**2) + 1e-10)
    std_j = np.sqrt(np.sum(p * (j_idx - mu_j)**2) + 1e-10)

    diff = np.abs(i_idx - j_idx)

    contrast      = np.sum(p * diff**2)
    dissimilarity = np.sum(p * diff)
    homogeneity   = np.sum(p / (1 + diff**2))
    energy        = np.sqrt(np.sum(p**2))
    asm           = np.sum(p**2)
    correlation   = np.sum(
        p * (i_idx - mu_i) * (j_idx - mu_j)
    ) / (std_i * std_j + 1e-10)

    return {
        'contrast'     : float(contrast),
        'dissimilarity': float(dissimilarity),
        'homogeneity'  : float(homogeneity),
        'energy'       : float(energy),
        'correlation'  : float(correlation),
        'ASM'          : float(asm),
    }


def extract_glcm_3d(voi):
    """
    Full 3D GLCM feature extraction pipeline.

    WHY aggregate with mean + std across 13 directions:
      - Mean: overall texture summary across all spatial directions
      - Std: texture anisotropy — how much texture varies by direction
             (e.g. atrophied hippocampus may be more anisotropic)

    Returns feature vector of shape (12,):
      [contrast_mean, contrast_std,
       dissimilarity_mean, dissimilarity_std,
       homogeneity_mean, homogeneity_std,
       energy_mean, energy_std,
       correlation_mean, correlation_std,
       ASM_mean, ASM_std]
    """
    voi_q = quantize_voi(voi)

    # Collect Haralick values per direction
    dir_features = {prop: [] for prop in HARALICK_PROPS}

    for direction in DIRECTIONS_3D:
        glcm = compute_3d_glcm_fast(voi_q, direction)
        props = haralick_from_glcm(glcm)
        for prop in HARALICK_PROPS:
            dir_features[prop].append(props[prop])

    # Aggregate: mean + std across 13 directions
    feature_vector = []
    for prop in HARALICK_PROPS:
        vals = np.array(dir_features[prop])
        feature_vector.append(vals.mean())
        feature_vector.append(vals.std())

    return np.array(feature_vector, dtype=np.float32)
    # Shape: (12,) = 6 properties × 2 stats


def main():
    cfg = load_config()

    CSV_DIR  = Path(cfg['paths']['csv_dir'])
    VOI_DIR  = Path(cfg['paths']['processed_dir']).parent / "voi"
    FEAT_DIR = Path(cfg['paths']['processed_dir']).parent / "features_affine" / "glcm_3d"
    SCALES   = cfg['voi']['scales']

    meta = pd.read_csv(CSV_DIR / cfg['paths']['metadata_csv'])

    # Create output dirs
    for scale_name in SCALES:
        for cls in cfg['classes']['target']:
            (FEAT_DIR / scale_name / cls).mkdir(parents=True, exist_ok=True)

    failed = []
    saved  = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="3D GLCM"):
        subj_id  = row['subject_id']
        img_id   = str(row['image_id'])
        cls      = row['class']
        filename = f"{subj_id}_{img_id}.npy"

        all_scales_ok = True

        for scale_name in SCALES:
            voi_path = VOI_DIR / scale_name / cls / filename

            if not voi_path.exists():
                failed.append({'subject_id': subj_id, 'image_id': img_id,
                               'scale': scale_name, 'error': 'VOI missing'})
                all_scales_ok = False
                continue

            voi  = np.load(str(voi_path))
            feat = extract_glcm_3d(voi)

            out_path = FEAT_DIR / scale_name / cls / filename
            np.save(str(out_path), feat)

        if all_scales_ok:
            saved += 1

    print(f"\nDone. Saved: {saved}  Failed: {len(failed)}")
    print(f"\nFeature vector dimensions:")
    print(f"  Per scale   : 12  (6 Haralick props × mean+std across 13 directions)")
    print(f"  All 3 scales: 36  (after feature-level fusion: concatenate 3 × 12)")
    print(f"  Decision fusion: 3 separate 12-dim vectors → 3 classifiers → vote")

    if failed:
        pd.DataFrame(failed).to_csv(
            CSV_DIR / "glcm_3d_failed.csv", index=False)


if __name__ == "__main__":
    main()