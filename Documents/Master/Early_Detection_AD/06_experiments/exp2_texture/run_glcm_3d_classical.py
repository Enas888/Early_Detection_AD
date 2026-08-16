# 06_experiments/run_glcm_3d_classical.py

import copy
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from build_dataset import (
    load_config,
    build_all_scales,
    get_feature_fused,
)


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

# Primary metric for model selection.
# Because class imbalance is suspected, use balanced accuracy.
SELECTION_METRIC = "balanced_accuracy"


cfg = load_config()

CSV_DIR = Path(cfg["paths"]["csv_dir"])

FEAT_DIR = (
    Path(cfg["paths"]["processed_dir"]).parent
    / "features"
    / "glcm_3d"
)

RES_DIR = Path(cfg["paths"]["results_dir"])

SCALES = cfg["voi"]["scales"]

CLASSES = cfg["classes"]["target"]


# ============================================================
# LOAD METADATA
# ============================================================

meta = pd.read_csv(
    CSV_DIR / cfg["paths"]["metadata_csv"]
)

meta["subject_id"] = (
    meta["subject_id"]
    .astype(str)
)


# ============================================================
# SUBJECT SPLITS
# ============================================================

SPLITS_DIR = Path(cfg["paths"]["splits_dir"])
train_subj = set(pd.read_csv(SPLITS_DIR / "train_subjects.csv")["subject_id"].astype(str))
val_subj   = set(pd.read_csv(SPLITS_DIR / "val_subjects.csv")["subject_id"].astype(str))
test_subj  = set(pd.read_csv(SPLITS_DIR / "test_subjects.csv")["subject_id"].astype(str))


# ============================================================
# CLASSIFICATION TASKS
# ============================================================

TASKS = {
    "cn_vs_ad": ["CN", "AD"],
    "cn_vs_emci": ["CN", "EMCI"],
    "cn_vs_lmci": ["CN", "LMCI"],
    "four_class": ["CN", "EMCI", "LMCI", "AD"],
}


# ============================================================
# CLASSIFIERS
# ============================================================

def make_classifiers():

    classifiers = {

        # ----------------------------------------------------
        # 1. Balanced RBF SVM
        # ----------------------------------------------------

        "svm_rbf_balanced": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),


        # ----------------------------------------------------
        # 2. Balanced Linear SVM
        # ----------------------------------------------------

        "svm_linear_balanced": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                SVC(
                    kernel="linear",
                    C=1.0,
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),


        # ----------------------------------------------------
        # 3. Balanced Logistic Regression
        # ----------------------------------------------------

        "logistic_balanced": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]),


        # ----------------------------------------------------
        # 4. Balanced Random Forest
        # ----------------------------------------------------

        "rf_balanced": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


        # ----------------------------------------------------
        # 5. Balanced Extra Trees
        # ----------------------------------------------------

        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    return classifiers


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    meta,
    subjects,
    split_name
):

    split_meta = meta[
        meta["subject_id"].isin(subjects)
    ]

    print(f"\n{split_name}:")

    counts = (
        split_meta["class"]
        .value_counts()
        .sort_index()
    )

    print(counts.to_string())


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true,
    y_pred,
    y_prob,
    classes
):

    if len(classes) == 2:
        average = "binary"
    else:
        average = "macro"

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                average=average,
                zero_division=0
            ),
    }


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        if len(classes) == 2:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob[:, 1]
            )

        else:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro"
            )

    except Exception:

        metrics["roc_auc"] = np.nan


    return metrics


# ============================================================
# FEATURE-LEVEL FUSION
# ============================================================

def prepare_feature_fusion(
    data
):

    X, y, subjects = get_feature_fused(
        data
    )

    return X, y, subjects


# ============================================================
# DECISION-LEVEL FUSION
# ============================================================

def train_decision_models(
    train_data,
    clf_template,
    train_indices
):

    models = []

    for scale_name in SCALES:

        X = train_data[
            scale_name
        ][0]

        y = train_data[
            scale_name
        ][1]

        X_fold = X[
            train_indices
        ]

        y_fold = y[
            train_indices
        ]

        clf = copy.deepcopy(
            clf_template
        )

        clf.fit(
            X_fold,
            y_fold
        )

        models.append(
            clf
        )

    return models


def decision_fusion_predict(
    models,
    data_by_scale,
    indices
):

    probabilities = []

    for model, scale_name in zip(
        models,
        SCALES
    ):

        X = data_by_scale[
            scale_name
        ][0]

        X = X[
            indices
        ]

        prob = model.predict_proba(
            X
        )

        probabilities.append(
            prob
        )


    # --------------------------------------------------------
    # Average probabilities across scales
    # --------------------------------------------------------

    avg_prob = np.mean(
        probabilities,
        axis=0
    )


    y_pred = np.argmax(
        avg_prob,
        axis=1
    )


    return y_pred, avg_prob


# ============================================================
# CROSS-VALIDATION: FEATURE FUSION
# ============================================================

def cross_validate_feature_fusion(
    data,
    clf_template,
    classes
):

    X, y, subjects = (
        prepare_feature_fusion(data)
    )

    y = np.asarray(y)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    fold_results = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(X, y),
        start=1
    ):

        clf = copy.deepcopy(
            clf_template
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        clf.fit(
            X[train_idx],
            y[train_idx]
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        y_pred = clf.predict(
            X[val_idx]
        )

        y_prob = clf.predict_proba(
            X[val_idx]
        )


        metrics = compute_metrics(
            y[val_idx],
            y_pred,
            y_prob,
            classes
        )

        metrics["fold"] = fold

        fold_results.append(
            metrics
        )


    fold_df = pd.DataFrame(
        fold_results
    )


    mean_metrics = {
        "cv_accuracy":
            fold_df["accuracy"].mean(),

        "cv_balanced_accuracy":
            fold_df["balanced_accuracy"].mean(),

        "cv_f1":
            fold_df["f1"].mean(),

        "cv_roc_auc":
            fold_df["roc_auc"].mean(),
    }


    return mean_metrics, fold_df


# ============================================================
# CROSS-VALIDATION: DECISION FUSION
# ============================================================

def cross_validate_decision_fusion(
    data,
    clf_template,
    classes
):

    # --------------------------------------------------------
    # Use labels from first scale
    # --------------------------------------------------------

    first_scale = list(
        SCALES.keys()
    )[0]

    y = np.asarray(
        data[first_scale][1]
    )


    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


    fold_results = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        skf.split(
            np.zeros(len(y)),
            y
        ),
        start=1
    ):

        models = []


        # ----------------------------------------------------
        # Train one model per scale
        # ----------------------------------------------------

        for scale_name in SCALES:

            X = data[
                scale_name
            ][0]

            y_scale = np.asarray(
                data[
                    scale_name
                ][1]
            )


            # Safety check
            if not np.array_equal(
                y_scale,
                y
            ):
                raise ValueError(
                    "Labels differ between scales."
                )


            clf = copy.deepcopy(
                clf_template
            )


            clf.fit(
                X[train_idx],
                y_scale[train_idx]
            )


            models.append(
                clf
            )


        # ----------------------------------------------------
        # Predict validation fold
        # ----------------------------------------------------

        probabilities = []


        for model, scale_name in zip(
            models,
            SCALES
        ):

            X = data[
                scale_name
            ][0]

            prob = model.predict_proba(
                X[val_idx]
            )

            probabilities.append(
                prob
            )


        avg_prob = np.mean(
            probabilities,
            axis=0
        )


        y_pred = np.argmax(
            avg_prob,
            axis=1
        )


        metrics = compute_metrics(
            y[val_idx],
            y_pred,
            avg_prob,
            classes
        )

        metrics["fold"] = fold

        fold_results.append(
            metrics
        )


    fold_df = pd.DataFrame(
        fold_results
    )


    mean_metrics = {
        "cv_accuracy":
            fold_df["accuracy"].mean(),

        "cv_balanced_accuracy":
            fold_df["balanced_accuracy"].mean(),

        "cv_f1":
            fold_df["f1"].mean(),

        "cv_roc_auc":
            fold_df["roc_auc"].mean(),
    }


    return mean_metrics, fold_df


# ============================================================
# FINAL FEATURE-LEVEL MODEL
# ============================================================

def final_feature_model(
    train_data,
    val_data,
    test_data,
    clf_template,
    classes
):

    X_train, y_train, _ = (
        get_feature_fused(
            train_data
        )
    )

    X_val, y_val, _ = (
        get_feature_fused(
            val_data
        )
    )

    X_test, y_test, _ = (
        get_feature_fused(
            test_data
        )
    )


    # --------------------------------------------------------
    # Combine TRAIN + VAL
    # --------------------------------------------------------

    X_train_final = np.concatenate(
        [
            X_train,
            X_val
        ],
        axis=0
    )

    y_train_final = np.concatenate(
        [
            y_train,
            y_val
        ],
        axis=0
    )


    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    clf = copy.deepcopy(
        clf_template
    )

    clf.fit(
        X_train_final,
        y_train_final
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    y_pred = clf.predict(
        X_test
    )

    y_prob = clf.predict_proba(
        X_test
    )


    metrics = compute_metrics(
        y_test,
        y_pred,
        y_prob,
        classes
    )


    return (
        metrics,
        y_test,
        y_pred,
        y_prob
    )


# ============================================================
# FINAL DECISION-LEVEL MODEL
# ============================================================

def final_decision_model(
    train_data,
    val_data,
    test_data,
    clf_template,
    classes
):

    models = []


    # --------------------------------------------------------
    # Train one final model per scale
    # using TRAIN + VAL
    # --------------------------------------------------------

    for scale_name in SCALES:

        X_train = train_data[
            scale_name
        ][0]

        y_train = train_data[
            scale_name
        ][1]

        X_val = val_data[
            scale_name
        ][0]

        y_val = val_data[
            scale_name
        ][1]


        X_train_final = np.concatenate(
            [
                X_train,
                X_val
            ],
            axis=0
        )

        y_train_final = np.concatenate(
            [
                y_train,
                y_val
            ],
            axis=0
        )


        clf = copy.deepcopy(
            clf_template
        )


        clf.fit(
            X_train_final,
            y_train_final
        )


        models.append(
            clf
        )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    probabilities = []


    for model, scale_name in zip(
        models,
        SCALES
    ):

        X_test = test_data[
            scale_name
        ][0]

        prob = model.predict_proba(
            X_test
        )

        probabilities.append(
            prob
        )


    avg_prob = np.mean(
        probabilities,
        axis=0
    )


    y_pred = np.argmax(
        avg_prob,
        axis=1
    )


    y_test = test_data[
        list(SCALES.keys())[0]
    ][1]


    metrics = compute_metrics(
        y_test,
        y_pred,
        avg_prob,
        classes
    )


    return (
        metrics,
        y_test,
        y_pred,
        avg_prob
    )


# ============================================================
# MAIN
# ============================================================

cv_results = []

final_results = []

confusion_matrices = {}


for task_name, task_classes in TASKS.items():

    print("\n")
    print("=" * 80)
    print(
        f"TASK: {task_name}"
    )
    print(
        f"CLASSES: {task_classes}"
    )
    print("=" * 80)


    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print_class_distribution(
        meta,
        train_subj,
        "TRAIN"
    )

    print_class_distribution(
        meta,
        val_subj,
        "VALIDATION"
    )

    print_class_distribution(
        meta,
        test_subj,
        "TEST"
    )


    # ========================================================
    # TASK METADATA
    # ========================================================

    meta_task = meta[
        meta["class"].isin(
            task_classes
        )
    ].copy()


    # ========================================================
    # LOAD TRAIN / VAL / TEST
    # ========================================================

    print("\nLoading TRAIN features...")

    train_data = build_all_scales(
        FEAT_DIR,
        SCALES,
        train_subj,
        task_classes,
        meta_task
    )


    print("Loading VALIDATION features...")

    val_data = build_all_scales(
        FEAT_DIR,
        SCALES,
        val_subj,
        task_classes,
        meta_task
    )


    print("Loading TEST features...")

    test_data = build_all_scales(
        FEAT_DIR,
        SCALES,
        test_subj,
        task_classes,
        meta_task
    )


    classifiers = make_classifiers()


    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    for clf_name, clf_template in classifiers.items():

        # ====================================================
        # FEATURE FUSION CV
        # ====================================================

        print("\n" + "-" * 70)

        print(
            f"CV: feature | {clf_name}"
        )


        cv_metrics, fold_df = (
            cross_validate_feature_fusion(
                train_data,
                clf_template,
                task_classes
            )
        )


        exp_id = (
            f"glcm_3d_feature_"
            f"{clf_name}_"
            f"{task_name}"
        )


        cv_result = {

            "exp_id":
                exp_id,

            "task":
                task_name,

            "classes":
                str(task_classes),

            "fusion":
                "feature",

            "classifier":
                clf_name,

            **cv_metrics
        }


        cv_results.append(
            cv_result
        )


        print(
            f"CV Accuracy = "
            f"{cv_metrics['cv_accuracy']:.4f}"
        )

        print(
            f"CV Balanced Accuracy = "
            f"{cv_metrics['cv_balanced_accuracy']:.4f}"
        )

        print(
            f"CV F1 = "
            f"{cv_metrics['cv_f1']:.4f}"
        )

        print(
            f"CV AUC = "
            f"{cv_metrics['cv_roc_auc']:.4f}"
        )


        # ====================================================
        # DECISION FUSION CV
        # ====================================================

        print("\n" + "-" * 70)

        print(
            f"CV: decision | {clf_name}"
        )


        cv_metrics, fold_df = (
            cross_validate_decision_fusion(
                train_data,
                clf_template,
                task_classes
            )
        )


        exp_id = (
            f"glcm_3d_decision_"
            f"{clf_name}_"
            f"{task_name}"
        )


        cv_result = {

            "exp_id":
                exp_id,

            "task":
                task_name,

            "classes":
                str(task_classes),

            "fusion":
                "decision",

            "classifier":
                clf_name,

            **cv_metrics
        }


        cv_results.append(
            cv_result
        )


        print(
            f"CV Accuracy = "
            f"{cv_metrics['cv_accuracy']:.4f}"
        )

        print(
            f"CV Balanced Accuracy = "
            f"{cv_metrics['cv_balanced_accuracy']:.4f}"
        )

        print(
            f"CV F1 = "
            f"{cv_metrics['cv_f1']:.4f}"
        )

        print(
            f"CV AUC = "
            f"{cv_metrics['cv_roc_auc']:.4f}"
        )


    # ========================================================
    # FIND BEST MODEL FOR THIS TASK
    # ========================================================

    task_cv = pd.DataFrame(
        cv_results
    )

    task_cv = task_cv[
        task_cv["task"] == task_name
    ]


    # --------------------------------------------------------
    # IMPORTANT:
    # select using balanced accuracy
    # --------------------------------------------------------

    best_row = task_cv.loc[
        task_cv[
            "cv_balanced_accuracy"
        ].idxmax()
    ]


    best_exp_id = best_row[
        "exp_id"
    ]

    best_classifier_name = best_row[
        "classifier"
    ]

    best_fusion = best_row[
        "fusion"
    ]


    print("\n")
    print("=" * 80)

    print(
        f"BEST CV MODEL: {best_exp_id}"
    )

    print(
        f"CV balanced accuracy: "
        f"{best_row['cv_balanced_accuracy']:.4f}"
    )

    print("=" * 80)


    # ========================================================
    # FINAL MODEL
    # ========================================================

    best_classifier = classifiers[
        best_classifier_name
    ]


    print(
        "\nRetraining best model on "
        "TRAIN + VALIDATION..."
    )


    if best_fusion == "feature":

        (
            metrics,
            y_test,
            y_pred,
            y_prob
        ) = final_feature_model(

            train_data,
            val_data,
            test_data,

            best_classifier,

            task_classes
        )

    else:

        (
            metrics,
            y_test,
            y_pred,
            y_prob
        ) = final_decision_model(

            train_data,
            val_data,
            test_data,

            best_classifier,

            task_classes
        )


    # ========================================================
    # FINAL CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print("\nFINAL TEST RESULTS")

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.4f}"
    )


    print("\nConfusion Matrix:")

    print(cm)


    # ========================================================
    # SAVE FINAL RESULT
    # ========================================================

    final_result = {

        "exp_id":
            best_exp_id,

        "task":
            task_name,

        "classes":
            str(task_classes),

        "fusion":
            best_fusion,

        "classifier":
            best_classifier_name,

        "selection_metric":
            SELECTION_METRIC,

        "cv_balanced_accuracy":
            best_row[
                "cv_balanced_accuracy"
            ],

        "cv_accuracy":
            best_row[
                "cv_accuracy"
            ],

        "cv_f1":
            best_row[
                "cv_f1"
            ],

        "cv_roc_auc":
            best_row[
                "cv_roc_auc"
            ],

        "test_accuracy":
            metrics[
                "accuracy"
            ],

        "test_balanced_accuracy":
            metrics[
                "balanced_accuracy"
            ],

        "test_f1":
            metrics[
                "f1"
            ],

        "test_roc_auc":
            metrics[
                "roc_auc"
            ],

        "n_test":
            len(y_test)
    }


    final_results.append(
        final_result
    )


    confusion_matrices[
        task_name
    ] = cm


# ============================================================
# SAVE CV RESULTS
# ============================================================

RES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLE_DIR = (
    RES_DIR /
    "tables"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


cv_df = pd.DataFrame(
    cv_results
)


cv_file = (
    TABLE_DIR /
    "glcm_3d_classical_cv_results.csv"
)


cv_df.to_csv(
    cv_file,
    index=False
)


# ============================================================
# SAVE FINAL TEST RESULTS
# ============================================================

final_df = pd.DataFrame(
    final_results
)


final_file = (
    TABLE_DIR /
    "glcm_3d_classical_final_test_results.csv"
)


final_df.to_csv(
    final_file,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

for task_name, cm in confusion_matrices.items():

    cm_file = (
        TABLE_DIR /
        f"glcm_3d_confusion_matrix_{task_name}.csv"
    )


    np.savetxt(
        cm_file,
        cm,
        delimiter=",",
        fmt="%d"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 80)

print("CROSS-VALIDATION RESULTS")

print("=" * 80)


print(
    cv_df[
        [
            "exp_id",
            "cv_accuracy",
            "cv_balanced_accuracy",
            "cv_f1",
            "cv_roc_auc"
        ]
    ].to_string(
        index=False
    )
)


print("\n")
print("=" * 80)

print("FINAL TEST RESULTS")

print("=" * 80)


print(
    final_df[
        [
            "exp_id",
            "cv_balanced_accuracy",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1",
            "test_roc_auc"
        ]
    ].to_string(
        index=False
    )
)


print("\nSaved:")
print(cv_file)
print(final_file)