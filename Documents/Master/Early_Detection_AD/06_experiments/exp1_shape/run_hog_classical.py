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

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

# ============================================================
# PROJECT IMPORTS
# ============================================================

from classifier_factory import create_classifier

from feature_fusion import feature_fusion

from decision_fusion import (
    decision_fusion_predict,
)

from experiment_utils import (
    get_task_classes,
    filter_task,
    calculate_metrics,
    save_cv_results,
    save_confusion_matrix,
    save_experiment_summary,
)


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
        / "04_results"
        / experiment_id
    )

    # --------------------------------------------------------
    # Run each task.
    # --------------------------------------------------------

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

        if fusion_type in [
            "narrow",
            "original",
            "wide",
        ]:

            X = datasets[
                fusion_type
            ]

            (
                fold_metrics,
                y_true,
                y_pred,
            ) = run_single_scale_cv(
                X,
                datasets["y"],
                task_name,
                classifier_name,
                classifier_cfg,
                cv_folds,
                random_state,
            )

        elif fusion_type == "feature":

            scale_features = {

                "narrow":
                    datasets["narrow"],

                "original":
                    datasets["original"],

                "wide":
                    datasets["wide"],
            }

            (
                fold_metrics,
                y_true,
                y_pred,
            ) = run_feature_fusion_cv(
                scale_features,
                datasets["y"],
                task_name,
                classifier_name,
                classifier_cfg,
                cv_folds,
                random_state,
            )

        elif fusion_type == "decision":

            scale_features = {

                "narrow":
                    datasets["narrow"],

                "original":
                    datasets["original"],

                "wide":
                    datasets["wide"],
            }

            (
                fold_metrics,
                y_true,
                y_pred,
            ) = run_decision_fusion_cv(
                scale_features,
                datasets["y"],
                task_name,
                classifier_name,
                classifier_cfg,
                cv_folds,
                random_state,
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


# ============================================================
# LOAD GLCM DATA
# ============================================================

def load_glcm_datasets():
    """
    Load the GLCM features for all three scales.

    IMPORTANT:
        This function assumes build_dataset.py provides
        build_all_scales().

    Adapt only this import/call if the exact function
    signature in your existing build_dataset.py differs.
    """

    from build_dataset import (
        build_all_scales
    )

    data_cfg = load_yaml(
        DATA_CONFIG_PATH
    )

    # --------------------------------------------------------
    # build_all_scales() should return the three aligned
    # feature matrices and labels.
    #
    # Expected conceptual output:
    #
    # {
    #     "narrow": X_narrow,
    #     "original": X_original,
    #     "wide": X_wide,
    #     "y": y
    # }
    #
    # If your current build_dataset.py returns a different
    # structure, only this adapter needs to be changed.
    # --------------------------------------------------------

    datasets = build_all_scales(
        data_cfg
    )

    required = [
        "narrow",
        "original",
        "wide",
        "y",
    ]

    for key in required:

        if key not in datasets:

            raise KeyError(
                f"build_all_scales() did not return "
                f"'{key}'"
            )

    return datasets


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

    print(
        "\nDataset loaded."
    )

    print(
        f"  Narrow   : "
        f"{datasets['narrow'].shape}"
    )

    print(
        f"  Original : "
        f"{datasets['original'].shape}"
    )

    print(
        f"  Wide     : "
        f"{datasets['wide'].shape}"
    )

    print(
        f"  Labels   : "
        f"{datasets['y'].shape}"
    )

    # --------------------------------------------------------
    # Run only configured classical GLCM experiments.
    # --------------------------------------------------------

    experiments = (
        experiment_cfg[
            "experiments"
        ]
    )

    for experiment in experiments:

        # ----------------------------------------------------
        # Ignore non-classical experiments.
        # ----------------------------------------------------

        if experiment.get(
            "pipeline"
        ) != "classical":

            continue

        # ----------------------------------------------------
        # Ignore non-GLCM experiments.
        # ----------------------------------------------------

        if experiment.get(
            "feature_type"
        ) != "glcm":

            continue

        run_experiment(
            experiment,
            datasets,
            experiment_cfg,
        )

    print("\n" + "=" * 75)
    print("ALL CONFIGURED GLCM EXPERIMENTS COMPLETE")
    print("=" * 75)

    print(
        "\nTest set was NOT used."
    )

    print(
        "Results saved under:"
    )

    print(
        f"  {PROJECT_ROOT / '04_results'}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()