#!/usr/bin/env python3
"""
run_glcm_classical.py

============================================================
GLCM CLASSICAL ML EXPERIMENT RUNNER
============================================================

CURRENT PURPOSE
---------------
Run classical Random Forest experiments using 2D GLCM
features.

Experiments currently supported:

    1. Single-scale:
        narrow
        original
        wide

    2. Feature-level fusion:
        narrow + original + wide

    3. Decision-level fusion:
        RF per scale + probability averaging


EVALUATION
----------
Current stage:

    TRAIN
      ↓
    5-fold Stratified CV
      ↓
    CV metrics
      ↓
    CV confusion matrix

The TEST set is intentionally NOT used yet.

After all experiments have been completed and the best
configuration selected, a separate final-evaluation stage
can retrain on TRAIN + VALIDATION and evaluate TEST once.


IMPORTANT
---------
The experiment configuration is controlled through:

    config/experiment_config.yaml

The classifier implementation is controlled through:

    04_classifiers/classical/classifier_factory.py
"""


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

# Add classifier directory
CLASSIFIER_DIR = PROJECT_ROOT / "04_classifiers" / "classical"
sys.path.insert(0, str(CLASSIFIER_DIR))

# Add fusion directory
FUSION_DIR = PROJECT_ROOT / "05_fusion"
sys.path.insert(0, str(FUSION_DIR))

# Add experiments directory
EXPERIMENTS_DIR = PROJECT_ROOT / "06_experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from classifier_factory import create_classifier
from feature_fusion import feature_fusion
from decision_fusion import decision_fusion_predict

from experiment_utils import (
    get_task_classes,
    filter_task,
    calculate_metrics,
    save_cv_results,
    save_confusion_matrix,
    save_experiment_summary,
)
from build_dataset import build_all_scales
# ============================================================
# CONFIGURATION
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "experiment_config.yaml"
)

DATA_CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "data_config.yaml"
)


# ============================================================
# LOAD YAML
# ============================================================

def load_yaml(
    path: Path
):
    """
    Load a YAML configuration file.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Configuration file not found:\n{path}"
        )

    with open(
        path,
        "r"
    ) as f:

        return yaml.safe_load(f)


# ============================================================
# TASK DATA
# ============================================================

def prepare_task_data(
    X,
    y,
    task_name,
):
    """
    Select samples belonging to the requested task.
    """

    classes = get_task_classes(
        task_name
    )

    X_task, y_task = filter_task(
        X,
        y,
        task_name,
    )

    return (
        X_task,
        y_task,
        classes,
    )


# ============================================================
# SINGLE-SCALE CV
# ============================================================

def run_single_scale_cv(
    X,
    y,
    task_name,
    classifier_name,
    classifier_config,
    cv_folds,
    random_state,
):
    """
    Run 5-fold Stratified CV for one GLCM scale.
    """

    classes = get_task_classes(
        task_name
    )

    X_task, y_task, _ = prepare_task_data(
        X,
        y,
        task_name,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics = []

    all_true = []
    all_pred = []

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    for fold_idx, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(X_task, y_task),
        start=1,
    ):

        print(
            f"      Fold {fold_idx}/{cv_folds}"
        )

        X_train = X_task[
            train_idx
        ]

        X_val = X_task[
            val_idx
        ]

        y_train = y_task[
            train_idx
        ]

        y_val = y_task[
            val_idx
        ]

        # ----------------------------------------------------
        # Fresh classifier for every fold.
        # ----------------------------------------------------

        model = create_classifier(
            classifier_name,
            classifier_config,
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_val
        )

        y_prob = model.predict_proba(
            X_val
        )

        metrics = calculate_metrics(
            y_val,
            y_pred,
            y_prob,
            task_name,
        )

        fold_metrics.append(
            metrics
        )

        all_true.extend(
            y_val
        )

        all_pred.extend(
            y_pred
        )

    return (
        fold_metrics,
        np.asarray(all_true),
        np.asarray(all_pred),
    )


# ============================================================
# FEATURE-LEVEL FUSION CV
# ============================================================

def run_feature_fusion_cv(
    scale_features,
    y,
    task_name,
    classifier_name,
    classifier_config,
    cv_folds,
    random_state,
):
    """
    Perform 5-fold CV using feature-level fusion.

    narrow + original + wide
            ↓
          18-D
            ↓
           RF
    """

    classes = get_task_classes(
        task_name
    )

    # --------------------------------------------------------
    # Fuse BEFORE CV.
    #
    # This is safe because concatenation itself does not learn
    # anything from the data.
    # --------------------------------------------------------

    X_fused = feature_fusion(
        scale_features["narrow"],
        scale_features["original"],
        scale_features["wide"],
    )

    X_task, y_task, _ = prepare_task_data(
        X_fused,
        y,
        task_name,
    )

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics = []

    all_true = []
    all_pred = []

    for fold_idx, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(X_task, y_task),
        start=1,
    ):

        print(
            f"      Fold {fold_idx}/{cv_folds}"
        )

        model = create_classifier(
            classifier_name,
            classifier_config,
        )

        model.fit(
            X_task[train_idx],
            y_task[train_idx],
        )

        y_pred = model.predict(
            X_task[val_idx]
        )

        y_prob = model.predict_proba(
            X_task[val_idx]
        )

        metrics = calculate_metrics(
            y_task[val_idx],
            y_pred,
            y_prob,
            task_name,
        )

        fold_metrics.append(
            metrics
        )

        all_true.extend(
            y_task[val_idx]
        )

        all_pred.extend(
            y_pred
        )

    return (
        fold_metrics,
        np.asarray(all_true),
        np.asarray(all_pred),
    )


# ============================================================
# DECISION-LEVEL FUSION CV
# ============================================================

def run_decision_fusion_cv(
    scale_features,
    y,
    task_name,
    classifier_name,
    classifier_config,
    cv_folds,
    random_state,
):
    """
    Perform 5-fold CV using decision-level fusion.

    For each fold:

        narrow   → RF
        original → RF
        wide     → RF

                    ↓

        average predicted probabilities

                    ↓

               final prediction
    """

    classes = get_task_classes(
        task_name
    )

    # --------------------------------------------------------
    # Filter task consistently across all scales.
    # --------------------------------------------------------

    filtered_scales = {}

    for scale, X in scale_features.items():

        X_task, y_task, _ = prepare_task_data(
            X,
            y,
            task_name,
        )

        filtered_scales[scale] = X_task

    y_task = y_task

    scales = [
        "narrow",
        "original",
        "wide",
    ]

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics = []

    all_true = []
    all_pred = []

    for fold_idx, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(
            filtered_scales["narrow"],
            y_task,
        ),
        start=1,
    ):

        print(
            f"      Fold {fold_idx}/{cv_folds}"
        )

        models = {}

        # ----------------------------------------------------
        # Train one model per scale.
        # ----------------------------------------------------

        for scale in scales:

            model = create_classifier(
                classifier_name,
                classifier_config,
            )

            model.fit(
                filtered_scales[scale][
                    train_idx
                ],
                y_task[
                    train_idx
                ],
            )

            models[scale] = model

        # ----------------------------------------------------
        # Generate predictions using all scales.
        # ----------------------------------------------------

        validation_features = {

            scale:
                filtered_scales[scale][
                    val_idx
                ]

            for scale in scales
        }

        y_pred, y_prob = (
            decision_fusion_predict(
                models,
                validation_features,
                scales,
                classes,
            )
        )

        metrics = calculate_metrics(
            y_task[val_idx],
            y_pred,
            y_prob,
            task_name,
        )

        fold_metrics.append(
            metrics
        )

        all_true.extend(
            y_task[val_idx]
        )

        all_pred.extend(
            y_pred
        )

    return (
        fold_metrics,
        np.asarray(all_true),
        np.asarray(all_pred),
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    experiment,
    datasets,
    experiment_cfg,
):
    """
    Run one configured experiment.

    Parameters
    ----------
    experiment:
        One experiment entry from YAML.

    datasets:
        Dictionary containing GLCM features.

    experiment_cfg:
        Complete experiment configuration.
    """

    experiment_id = experiment[
        "id"
    ]

    fusion_type = experiment[
        "fusion_type"
    ]

    classifier_name = experiment[
        "classifier"
    ]

    print("\n" + "=" * 75)

    print(
        f"EXPERIMENT: {experiment_id}"
    )

    print(
        f"Fusion    : {fusion_type}"
    )

    print(
        f"Classifier: {classifier_name}"
    )

    print("=" * 75)

    evaluation_cfg = (
        experiment_cfg[
            "evaluation"
        ]
    )

    cv_folds = evaluation_cfg.get(
        "cv_folds",
        5,
    )

    random_state = evaluation_cfg.get(
        "random_state",
        42,
    )

    classifier_cfg = (
        experiment_cfg[
            "classifiers"
        ][
            classifier_name
        ]
    )

    tasks = (
        experiment_cfg[
            "tasks"
        ]["binary"]
        +
        experiment_cfg[
            "tasks"
        ]["multiclass"]
    )

    # --------------------------------------------------------
    # Result directory
    # --------------------------------------------------------

    results_root = (
        PROJECT_ROOT
        / "08_results"
        / experiment_id
    )

    # --------------------------------------------------------
    # Run each task.
    # --------------------------------------------------------
    task_results = []

    for task_name in tasks:

        print(
            f"\n  Task: {task_name}"
        )

        task_dir = (
            results_root
            / task_name
        )

        task_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Save experiment metadata.
        # ----------------------------------------------------

        save_experiment_summary(
            task_dir,
            experiment_id,
            task_name,
            classifier_name,
            fusion_type,
        )

        # ----------------------------------------------------
        # Select appropriate experiment.
        # ----------------------------------------------------

        # Extract y from any scale (same for all)
        y = datasets["narrow"][1]
        


        if fusion_type in ["narrow", "original", "wide"]:
            X, y, _ = datasets[fusion_type]       # unpack tuple
            fold_metrics, y_true, y_pred = run_single_scale_cv(
                X, y, task_name, classifier_name,
                classifier_cfg, cv_folds, random_state,
            )

        elif fusion_type == "feature":
            scale_features = {
                "narrow":   datasets["narrow"][0],
                "original": datasets["original"][0],
                "wide":     datasets["wide"][0],
            }
            fold_metrics, y_true, y_pred = run_feature_fusion_cv(
                scale_features, y, task_name, classifier_name,
                classifier_cfg, cv_folds, random_state,
            )

        elif fusion_type == "decision":
            scale_features = {
                "narrow":   datasets["narrow"][0],
                "original": datasets["original"][0],
                "wide":     datasets["wide"][0],
            }
            fold_metrics, y_true, y_pred = run_decision_fusion_cv(
                scale_features, y, task_name, classifier_name,
                classifier_cfg, cv_folds, random_state,
            )

        else:

            raise ValueError(
                f"Unknown fusion type: "
                f"{fusion_type}"
            )

        # ----------------------------------------------------
        # Save CV metrics.
        # ----------------------------------------------------

        save_cv_results(
            fold_metrics,
            task_dir,
        )

        # ----------------------------------------------------
        # Save aggregated CV confusion matrix.
        #
        # This contains predictions from the held-out fold
        # for every training sample.
        # ----------------------------------------------------

        class_names = get_task_classes(
            task_name
        )

        save_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            task_dir / "cv_confusion_matrix.png",
            (
                f"{experiment_id} - "
                f"{task_name} - "
                f"5-Fold CV"
            ),
        )

        print(
            f"      Results saved to:"
            f"\n      {task_dir}"
        )


   
        cv_df = save_cv_results(fold_metrics, task_dir)
        mean_row = cv_df[cv_df['fold'] == 'mean'].iloc[0]

        task_results.append({
            'experiment_id': experiment_id,
            'task'         : task_name,
            'classifier'   : classifier_name,
            'fusion'       : fusion_type,
            'accuracy'     : float(mean_row['accuracy']),
            'balanced_acc' : float(mean_row['balanced_accuracy']),
            'f1'           : float(mean_row['f1']),
            'roc_auc'      : float(mean_row['roc_auc']),
        })

        print(f"      Results saved to:\n      {task_dir}")

    return task_results
        


# ============================================================
# LOAD GLCM DATA
# ============================================================

def load_glcm_datasets():

    print("\nLoading GLCM datasets...")

    # --------------------------------------------------------
    # Load data configuration
    # --------------------------------------------------------

    with open(DATA_CONFIG_PATH, "r") as f:
        data_cfg = yaml.safe_load(f)

    csv_dir = Path(
        data_cfg["paths"]["csv_dir"]
    )

    processed_dir = Path(
        data_cfg["paths"]["processed_dir"]
    )

    classes = data_cfg["classes"]["target"]

    scales = list(
        data_cfg["voi"]["scales"].keys()
    )

    # --------------------------------------------------------
    # GLCM feature directory
    #
    # IMPORTANT: according to your current structure:
    #
    # 01_data/processed/features_affine/glcm/
    # --------------------------------------------------------

    feat_dir = (
        PROJECT_ROOT
        / "01_data"
        / "processed"
        / "features_affine"
        / "glcm"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata_path = (
        csv_dir
        / data_cfg["paths"]["metadata_csv"]
    )

    meta = pd.read_csv(
        metadata_path
    )

    # Make sure subject IDs have the same type
    meta["subject_id"] = (
        meta["subject_id"].astype(str)
    )

    # --------------------------------------------------------
    # Split files
    # --------------------------------------------------------

    splits_dir = Path(
        data_cfg["paths"]["splits_dir"]
    )

    train_csv = splits_dir / "train_subjects.csv"
    val_csv   = splits_dir / "val_subjects.csv"
    test_csv  = splits_dir / "test_subjects.csv"

    train_split = pd.read_csv(train_csv)
    val_split   = pd.read_csv(val_csv)
    test_split  = pd.read_csv(test_csv)

    train_subjects = set(
        train_split["subject_id"].astype(str)
    )

    val_subjects = set(
        val_split["subject_id"].astype(str)
    )

    test_subjects = set(
        test_split["subject_id"].astype(str)
    )

    # --------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------

    # Combine train + val subjects for CV
    trainval_subjects = train_subjects | val_subjects  # set union

    trainval_data = build_all_scales(
        feat_dir=feat_dir,
        scales=scales,
        split_subjects=trainval_subjects,
        classes=classes,
        meta=meta
    )

    test_data = build_all_scales(
        feat_dir=feat_dir,
        scales=scales,
        split_subjects=test_subjects,
        classes=classes,
        meta=meta
    )

    return {
        "trainval": trainval_data,   # use this for CV
        "test"    : test_data        # never touch until final eval
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print("=" * 75)
    print("GLCM CLASSICAL EXPERIMENT PIPELINE")
    print("=" * 75)

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Load experiment configuration.
    # --------------------------------------------------------

    experiment_cfg = load_yaml(
        CONFIG_PATH
    )

    # --------------------------------------------------------
    # Load GLCM datasets.
    # --------------------------------------------------------

    print(
        "\nLoading GLCM datasets..."
    )

    datasets = load_glcm_datasets()


    print("\nDataset structure:")
    print(datasets.keys())

    for split in ["trainval", "test"]:
        print(f"\n{split}:")
        for scale in datasets[split]:
            X, y, ids = datasets[split][scale]
            print(f"  {scale:10s} X={X.shape}, y={y.shape}, subjects={len(set(ids))}")

    # NEW — matches the nested train/val/test structure
    print("\nDataset loaded successfully.")

    # Pass only train data to experiments
    # val and test are reserved for final evaluation
    train_datasets = datasets['trainval']

    # print(
    # f"  Labels   : "
    # f"{datasets['y'].shape}"
    # )

    # --------------------------------------------------------
    # Run only configured classical GLCM experiments.
    # --------------------------------------------------------

    experiments = (
        experiment_cfg[
            "experiments"
        ]
    )

    # In main() — replace the experiment loop
    all_results = []

    for experiment in experiments:
        if experiment.get("pipeline") != "classical":
            continue
        if experiment.get("feature_type") != "glcm":
            continue

        task_results = run_experiment(
            experiment, train_datasets, experiment_cfg)
        all_results.extend(task_results)

    # ── Final summary table ───────────────────────────────────────
    print("\n" + "=" * 75)
    print("FINAL CV RESULTS SUMMARY — ALL EXPERIMENTS")
    print("=" * 75)

    summary_df = pd.DataFrame(all_results)

    # Print to terminal
    print(summary_df[[
        'experiment_id', 'task',
        'accuracy', 'balanced_acc', 'f1', 'roc_auc'
    ]].to_string(index=False))

    # Save to CSV
    summary_path = PROJECT_ROOT / "08_results" / "all_experiments_summary_svm.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved → {summary_path}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()