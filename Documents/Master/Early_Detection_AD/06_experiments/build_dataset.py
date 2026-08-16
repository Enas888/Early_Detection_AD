# 06_experiments/build_dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
import yaml


GLCM_N_FEATURES = 6

def load_config():
    with open("config/data_config.yaml") as f:
        return yaml.safe_load(f)

def load_features_for_split(feat_dir, scale_name, split_subjects,
                             classes, meta):
    """
    Load all .npy feature files for subjects in a given split.
    Returns X (n_samples, 6), y (n_samples,), subject_ids
    """
    X, y, ids = [], [], []
    label_map = {cls: i for i, cls in enumerate(classes)}

    for _, row in meta.iterrows():
        if row['subject_id'] not in split_subjects:
            continue
        fname    = f"{row['subject_id']}_{row['image_id']}.npy"
        fpath    = feat_dir / scale_name / row['class'] / fname
        if not fpath.exists():
            continue
        feat = np.load(str(fpath))

        X.append(feat)
        y.append(label_map[row['class']])
        ids.append(row['subject_id'])

    return np.array(X), np.array(y), ids


def build_all_scales(feat_dir, scales, split_subjects,
                     classes, meta):
    """
    Returns dict: {scale_name: (X, y, ids)}
    All scales share the same y and ids (same subjects, same order).
    """
    data = {}
    for scale_name in scales:
        X, y, ids = load_features_for_split(
            feat_dir, scale_name, split_subjects, classes, meta)
        data[scale_name] = (X, y, ids)
    return data


def get_feature_fused(data):
    """
    Concatenate all scale feature vectors → 18-dim.
    WHY: 6 GLCM features × 3 spatial scales = 18 features.
    """
    arrays = [data[s][0] for s in data]
    X_fused = np.concatenate(arrays, axis=1)  # (n_samples, 18)
    y = list(data.values())[0][1]
    ids = list(data.values())[0][2]
    return X_fused, y, ids