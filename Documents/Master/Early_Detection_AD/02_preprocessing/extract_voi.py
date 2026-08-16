#!/usr/bin/env python3
"""
extract_voi.py

Subject/visit-specific multi-scale VOI extraction using the Juelich atlas.

Method:
  1. Load configuration and metadata.
  2. Load Juelich atlas and build binary ROI mask (HC + AMY + EC).
  3. Resample atlas ROI into MRI voxel grid ONCE (all MRIs share
     same MNI grid 182×218×182 after affine registration).
  4. For each subject/visit:
     a. Load registered MRI.
     b. Apply shared ROI mask → subject bbox.
     c. Generate 3 scaled bboxes (narrow ×0.8, original ×1.0, wide ×1.2).
     d. Normalize MRI volume-level.
     e. Extract and save 3 VOIs (.npy).
     f. Extract and save 3 ROI mask crops (.npy) — used later to
        restrict GLCM to actual ROI voxels, not full rectangle.
  5. Save all metadata CSVs to 01_data/csv/.

WHY save the ROI mask crop:
  The VOI is a rectangular box — it contains both ROI tissue and
  surrounding background. Saving the mask crop lets GLCM extraction
  later compute texture ONLY from atlas-defined ROI voxels, avoiding
  noise from non-ROI tissue inside the rectangle.
"""

import numpy as np
import nibabel as nib
import pandas as pd
import yaml

from pathlib import Path
from tqdm import tqdm
from nibabel.processing import resample_from_to


# ============================================================
# CONFIGURATION
# ============================================================

def load_config():
    project_root = Path(__file__).resolve().parent.parent
    config_path  = project_root / "config" / "data_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found:\n{config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# ATLAS
# ============================================================

def load_atlas(atlas_path: Path) -> nib.Nifti1Image:
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas not found:\n{atlas_path}")

    print(f"\nLoading Juelich atlas: {atlas_path}")
    atlas_img = nib.load(str(atlas_path))
    print(f"  Shape: {atlas_img.shape}")
    return atlas_img


def build_roi_mask_img(atlas_img: nib.Nifti1Image,
                       roi_labels: list) -> nib.Nifti1Image:
    """
    Build binary ROI mask in atlas space.
    Returns NIfTI image so affine is preserved for resampling.
    """
    atlas_data = atlas_img.get_fdata()
    roi_labels = [int(x) for x in roi_labels]
    roi_mask   = np.isin(atlas_data, roi_labels)
    n_voxels   = int(roi_mask.sum())

    if n_voxels == 0:
        raise ValueError(
            "ROI mask is empty — check roi_labels in data_config.yaml")

    print(f"\nROI mask: {n_voxels} voxels  |  labels: {roi_labels}")

    return nib.Nifti1Image(
        roi_mask.astype(np.uint8),
        atlas_img.affine,
        atlas_img.header
    )


def resample_roi_to_mri(roi_img: nib.Nifti1Image,
                        reference_mri: nib.Nifti1Image) -> np.ndarray:
    """
    Resample atlas ROI mask into MRI voxel grid using nearest-neighbor.
    WHY nearest-neighbor: preserves binary mask — no fractional labels.
    WHY once: all MRIs share identical MNI grid after registration,
    so the result is the same for every subject.
    """
    resampled = resample_from_to(roi_img,
                                  (reference_mri.shape,
                                   reference_mri.affine),
                                  order=0)
    mask = resampled.get_fdata() > 0.5

    if not np.any(mask):
        raise ValueError("Resampled ROI mask is empty in MRI grid.")

    n_voxels = int(mask.sum())
    print(f"  Resampled ROI: {n_voxels} voxels in MRI grid "
          f"{reference_mri.shape}")
    return mask


def verify_shared_affine(meta: pd.DataFrame,
                         n_check: int = 20) -> bool:
    """
    Verify that all MRIs share the same affine and shape.
    WHY: confirms the shared ROI mask is valid for all subjects.
    """
    print(f"\nVerifying affine consistency (first {n_check} volumes)...")
    paths   = meta['file_path'].head(n_check).tolist()
    ref_img = nib.load(str(paths[0]))
    ref_aff = ref_img.affine
    ref_shp = ref_img.shape

    all_ok = True
    for p in paths[1:]:
        img = nib.load(str(p))
        if not np.allclose(img.affine, ref_aff, atol=1e-4):
            print(f"  WARNING: Different affine: {p}")
            all_ok = False
        if img.shape != ref_shp:
            print(f"  WARNING: Different shape {img.shape}: {p}")
            all_ok = False

    if all_ok:
        print(f"  All {n_check} volumes share identical affine "
              f"and shape {ref_shp} ✓")
    return all_ok, ref_img


# ============================================================
# BOUNDING BOX
# ============================================================

def calculate_bbox(roi_mask: np.ndarray) -> dict:
    """
    Bounding box of non-zero voxels in the ROI mask.
    max coordinates are exclusive (NumPy slicing convention).
    """
    x, y, z = np.where(roi_mask)

    if len(x) == 0:
        raise ValueError("ROI mask is empty — cannot compute bbox.")

    return {
        "x_min": int(x.min()), "x_max": int(x.max()) + 1,
        "y_min": int(y.min()), "y_max": int(y.max()) + 1,
        "z_min": int(z.min()), "z_max": int(z.max()) + 1,
    }


def scale_bbox(bbox: dict, scale: float,
               volume_shape: tuple) -> dict:
    """
    Scale bbox around its center.
    scale < 1.0 → narrower (for atrophied tissue, less background)
    scale = 1.0 → exact atlas bbox
    scale > 1.0 → wider (registration offset safety margin)

    Shifts bbox if it exceeds volume boundaries rather than clipping,
    so VOI size stays constant. Final clip ensures safety.
    """
    orig_min  = np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"]],
                          dtype=float)
    orig_max  = np.array([bbox["x_max"], bbox["y_max"], bbox["z_max"]],
                          dtype=float)
    orig_size = orig_max - orig_min
    center    = (orig_min + orig_max) / 2.0

    new_size = np.maximum(np.round(orig_size * scale).astype(int), 1)
    new_min  = np.floor(center - new_size / 2.0).astype(int)
    new_max  = new_min + new_size

    for ax in range(3):
        if new_min[ax] < 0:
            shift = -new_min[ax]
            new_min[ax] += shift
            new_max[ax] += shift
        if new_max[ax] > volume_shape[ax]:
            shift = new_max[ax] - volume_shape[ax]
            new_min[ax] -= shift
            new_max[ax] -= shift
        new_min[ax] = max(int(new_min[ax]), 0)
        new_max[ax] = min(int(new_max[ax]), volume_shape[ax])

    return {
        "x_min": int(new_min[0]), "x_max": int(new_max[0]),
        "y_min": int(new_min[1]), "y_max": int(new_max[1]),
        "z_min": int(new_min[2]), "z_max": int(new_max[2]),
    }


# ============================================================
# MRI UTILITIES
# ============================================================

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """
    Volume-level min-max normalization → [0, 1].
    WHY volume-level (not slice-level): preserves relative
    intensity differences across slices within the same brain.
    """
    vol = volume.astype(np.float32)
    mn, mx = vol.min(), vol.max()
    if mx - mn < 1e-8:
        return np.zeros_like(vol)
    return (vol - mn) / (mx - mn)


def extract_crop(volume: np.ndarray, bbox: dict) -> np.ndarray:
    """Crop a 3D array using a bbox dict."""
    return volume[
        bbox["x_min"]:bbox["x_max"],
        bbox["y_min"]:bbox["y_max"],
        bbox["z_min"]:bbox["z_max"],
    ]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 75)
    print("MULTI-SCALE VOI EXTRACTION")
    print("=" * 75)

    # ── Config ───────────────────────────────────────────────
    cfg           = load_config()
    csv_dir       = Path(cfg["paths"]["csv_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    atlas_path    = Path(cfg["paths"]["atlas_path"])
    scales        = cfg["voi"]["scales"]
    roi_labels    = cfg["voi"]["roi_labels"]
    classes       = cfg["classes"]["target"]

    print(f"\nScales       : {scales}")
    print(f"Classes      : {classes}")
    print(f"Atlas        : {atlas_path}")

    # ── Metadata ─────────────────────────────────────────────
    meta_path = csv_dir / cfg["paths"]["metadata_csv"]
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found:\n{meta_path}")

    meta = pd.read_csv(meta_path)
    print(f"\nMetadata: {len(meta)} visits | "
          f"{meta['subject_id'].nunique()} subjects | "
          f"classes: {sorted(meta['class'].unique())}")

    # ── Atlas → ROI mask image ────────────────────────────────
    atlas_img = load_atlas(atlas_path)
    roi_img   = build_roi_mask_img(atlas_img, roi_labels)

    # ── Verify all MRIs share same grid, resample ROI once ───
    affines_ok, ref_mri = verify_shared_affine(meta, n_check=20)
    if not affines_ok:
        print("  WARNING: affines differ — per-subject resampling "
              "would be more accurate. Proceeding with shared mask.")

    shared_roi_mask = resample_roi_to_mri(roi_img, ref_mri)
    shared_bbox     = calculate_bbox(shared_roi_mask)

    print(f"\nShared atlas bbox:")
    print(f"  X: {shared_bbox['x_min']} → {shared_bbox['x_max']}  "
          f"(W={shared_bbox['x_max']-shared_bbox['x_min']})")
    print(f"  Y: {shared_bbox['y_min']} → {shared_bbox['y_max']}  "
          f"(H={shared_bbox['y_max']-shared_bbox['y_min']})")
    print(f"  Z: {shared_bbox['z_min']} → {shared_bbox['z_max']}  "
          f"(D={shared_bbox['z_max']-shared_bbox['z_min']})")

    # Precompute all scaled bboxes (identical for every subject)
    scaled_bboxes = {
        name: scale_bbox(shared_bbox, float(val), ref_mri.shape)
        for name, val in scales.items()
    }
    print("\nScaled bboxes:")
    for name, bb in scaled_bboxes.items():
        W = bb['x_max']-bb['x_min']
        H = bb['y_max']-bb['y_min']
        D = bb['z_max']-bb['z_min']
        print(f"  {name:10s}: W={W:3d} H={H:3d} D={D:3d}")

    # ── Output directories ────────────────────────────────────
    voi_dir  = processed_dir / "voi"    # VOI crops
    mask_dir = processed_dir / "masks"  # ROI mask crops

    for scale_name in scales:
        for cls in classes:
            (voi_dir  / scale_name / cls).mkdir(parents=True, exist_ok=True)
            (mask_dir / scale_name / cls).mkdir(parents=True, exist_ok=True)

    print(f"\nVOI output  : {voi_dir}")
    print(f"Mask output : {mask_dir}")

    # ── Per-subject extraction ────────────────────────────────
    voi_records  = []   # metadata per saved VOI
    mask_records = []   # metadata per saved mask crop
    failed       = []   # failed subjects

    for _, row in tqdm(meta.iterrows(), total=len(meta),
                       desc="VOI extraction"):

        subj_id   = str(row["subject_id"])
        img_id    = str(row["image_id"])
        cls       = str(row["class"])
        file_path = Path(str(row["file_path"]))
        filename  = f"{subj_id}_{img_id}.npy"

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"MRI not found: {file_path}")

            mri_img = nib.load(str(file_path))
            volume  = mri_img.get_fdata()

            if volume.ndim != 3:
                raise ValueError(f"MRI is not 3D: {volume.shape}")

            vol_norm = normalize_volume(volume)

            for scale_name, bbox in scaled_bboxes.items():

                # VOI crop (normalized MRI)
                voi      = extract_crop(vol_norm, bbox)
                voi_path = voi_dir / scale_name / cls / filename
                np.save(str(voi_path), voi.astype(np.float32))

                # Mask crop (binary ROI mask within bbox)
                # WHY: lets GLCM later compute texture from ROI
                # voxels only, not full rectangle background
                mask_crop = extract_crop(
                    shared_roi_mask.astype(np.uint8), bbox)
                mask_path = mask_dir / scale_name / cls / filename
                np.save(str(mask_path), mask_crop.astype(np.uint8))

                W = bbox['x_max']-bbox['x_min']
                H = bbox['y_max']-bbox['y_min']
                D = bbox['z_max']-bbox['z_min']

                voi_records.append({
                    "subject_id": subj_id,
                    "image_id"  : img_id,
                    "class"     : cls,
                    "scale"     : scale_name,
                    "x_min": bbox["x_min"], "x_max": bbox["x_max"],
                    "y_min": bbox["y_min"], "y_max": bbox["y_max"],
                    "z_min": bbox["z_min"], "z_max": bbox["z_max"],
                    "voi_shape" : f"{W}x{H}x{D}",
                    "roi_voxels": int(mask_crop.sum()),
                })

                mask_records.append({
                    "subject_id": subj_id,
                    "image_id"  : img_id,
                    "class"     : cls,
                    "scale"     : scale_name,
                    "mask_path" : str(mask_path),
                    "roi_voxels": int(mask_crop.sum()),
                })

        except Exception as e:
            failed.append({
                "subject_id": subj_id,
                "image_id"  : img_id,
                "class"     : cls,
                "file_path" : str(file_path),
                "error"     : str(e),
            })

    # ── Save all CSVs to 01_data/csv/ ────────────────────────
    suffix = "_affine_3t"

    voi_meta_path  = csv_dir / f"voi_metadata{suffix}.csv"
    mask_meta_path = csv_dir / f"subject_roi_mask_metadata{suffix}.csv"
    failed_path    = csv_dir / f"voi_extraction_failed{suffix}.csv"

    pd.DataFrame(voi_records).to_csv(voi_meta_path, index=False)
    pd.DataFrame(mask_records).to_csv(mask_meta_path, index=False)

    if failed:
        pd.DataFrame(failed).to_csv(failed_path, index=False)

    # ── Final report ──────────────────────────────────────────
    n_saved    = len(meta) - len(failed)
    n_expected = len(meta) * len(scales)

    print("\n" + "=" * 75)
    print("EXTRACTION COMPLETE")
    print("=" * 75)
    print(f"  MRI visits    : {len(meta)}")
    print(f"  Successful    : {n_saved}")
    print(f"  Failed        : {len(failed)}")
    print(f"  VOIs saved    : {len(voi_records)}  (expected {n_expected})")
    print(f"\nCSVs saved to {csv_dir}:")
    print(f"  {voi_meta_path.name}")
    print(f"  {mask_meta_path.name}")
    if failed:
        print(f"  {failed_path.name}  ← check these")
    print("\nDone.")


if __name__ == "__main__":
    main()