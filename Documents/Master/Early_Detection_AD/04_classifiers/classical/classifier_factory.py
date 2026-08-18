#!/usr/bin/env python3
"""
classifier_factory.py

Central factory for classical ML classifiers.

WHY:
    The experiment runner should not contain classifier-specific
    logic such as:

        if classifier == "rf":
            ...
        elif classifier == "svm":
            ...

    Instead, classifiers are selected through experiment_config.yaml
    and instantiated here.

CURRENTLY SUPPORTED:
    - Random Forest
    - SVM
    - Logistic Regression

FUTURE:
    - Extra Trees
    - XGBoost
    - other classical classifiers
"""

from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
# ============================================================
# RANDOM FOREST
# ============================================================

def make_rf(params: Dict[str, Any]):
    """
    Create a Random Forest classifier.

    Parameters come from experiment_config.yaml.
    """

    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 42),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        max_features=params.get("max_features", "sqrt"),
        n_jobs=params.get("n_jobs", -1),
    )


# ============================================================
# SVM
# ============================================================

def make_svm(params: Dict[str, Any]):
    """
    Create an SVM classifier.

    Included for future experiments.
    """

    return SVC(
        class_weight=params.get("class_weight", "balanced"),
        probability=params.get("probability", True),
        random_state=params.get("random_state", 42),
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "rbf"),
    )


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def make_logistic_regression(params: Dict[str, Any]):
    """
    Create Logistic Regression.

    Included for future experiments.
    """

    return LogisticRegression(
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 42),
        max_iter=params.get("max_iter", 1000),
        C=params.get("C", 1.0),
    )

# ============================================================
# MLP
# ============================================================

def make_mlp(params: Dict[str, Any]):
    """
    Create an MLP classifier.

    Designed for low-dimensional handcrafted features such as GLCM.

    Parameters come from experiment_config.yaml.
    """

    return MLPClassifier(
        hidden_layer_sizes=tuple(
            params.get("hidden_layer_sizes", [32, 16])
        ),
        activation=params.get("activation", "relu"),
        solver=params.get("solver", "adam"),

        alpha=params.get("alpha", 0.0001),

        learning_rate_init=params.get(
            "learning_rate_init",
            0.001
        ),

        max_iter=params.get(
            "max_iter",
            300
        ),

        batch_size=params.get(
            "batch_size",
            "auto"
        ),

        early_stopping=params.get(
            "early_stopping",
            True
        ),

        validation_fraction=params.get(
            "validation_fraction",
            0.15
        ),

        n_iter_no_change=params.get(
            "n_iter_no_change",
            20
        ),

        tol=params.get(
            "tol",
            1e-4
        ),

        random_state=params.get(
            "random_state",
            42
        ),
    )

# ============================================================
# FACTORY REGISTRY
# ============================================================

CLASSIFIER_FACTORIES = {

    "rf": make_rf,

    "svm": make_svm,

    "logistic_regression": make_logistic_regression,

    "mlp": make_mlp,
}
# ============================================================
# PUBLIC FACTORY FUNCTION
# ============================================================

def create_classifier(
    classifier_name: str,
    classifier_config: Dict[str, Any],
):
    """
    Create a fresh classifier instance.

    Parameters
    ----------
    classifier_name:
        Name used in experiment_config.yaml.

    classifier_config:
        Dictionary containing the classifier parameters.

    Returns
    -------
    sklearn classifier
    """

    classifier_name = classifier_name.lower()

    if classifier_name not in CLASSIFIER_FACTORIES:
        available = ", ".join(CLASSIFIER_FACTORIES.keys())

        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Available classifiers: {available}"
        )

    factory = CLASSIFIER_FACTORIES[classifier_name]

    return factory(classifier_config)