from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from .config import K
from .neighbors import StableEuclideanKNN


CLASSIFIER_CONFIGS = {
    "knn": {
        "normalize_l2": True,
        "params": {
            "weights": "uniform",
            "metric": "euclidean",
            "p": 2,
        },
    },
    "svm": {
        "normalize_l2": True,
        "params": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "class_weight": None,
        },
    },
    "logreg": {
        "normalize_l2": True,
        "params": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": None,
        },
    },
    "rf": {
        "normalize_l2": False,
        "params": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": None,
            "n_jobs": 1,
        },
    },
}


def available_classifiers() -> tuple[str, ...]:
    return tuple(CLASSIFIER_CONFIGS)


def normalize_classifier_name(classifier: str) -> str:
    classifier = str(classifier).strip().lower()
    aliases = {
        "random_forest": "rf",
        "randomforest": "rf",
        "logistic_regression": "logreg",
        "logistic": "logreg",
    }
    classifier = aliases.get(classifier, classifier)
    if classifier not in CLASSIFIER_CONFIGS:
        valid = ", ".join(available_classifiers())
        raise ValueError(f"Unknown classifier: {classifier!r}. Use one of: {valid}.")
    return classifier


def classifier_config(classifier: str) -> dict:
    classifier = normalize_classifier_name(classifier)
    config = CLASSIFIER_CONFIGS[classifier]
    result = {
        "classifier": classifier,
        "normalize_l2": bool(config["normalize_l2"]),
        "params": dict(config["params"]),
    }
    if classifier == "knn":
        result["euclidean_evaluation"] = {
            "version": "direct_cdist_v1",
            "distance": "scipy.cdist.euclidean.float64",
            "distance_ties": "training_order_exact",
            "vote_ties": "class_order",
        }
    return result


def classifier_config_json(classifier: str) -> str:
    return json.dumps(classifier_config(classifier), sort_keys=True)


def _split_xy(df: pd.DataFrame, label_col: str):
    if df.empty:
        raise ValueError("Input dataframe is empty.")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    feature_cols = [col for col in df.columns if col != label_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()
    return X, y


def _build_classifier(
    classifier: str,
    *,
    k_eff: int,
    random_state: int | None,
    classifier_params: dict | None = None,
):
    classifier = normalize_classifier_name(classifier)
    params = dict(CLASSIFIER_CONFIGS[classifier]["params"])
    if classifier_params:
        params.update(classifier_params)

    if classifier == "knn":
        return StableEuclideanKNN(n_neighbors=k_eff, **params)
    if classifier == "svm":
        return SVC(**params)
    if classifier == "logreg":
        return LogisticRegression(**params)
    if classifier == "rf":
        return RandomForestClassifier(random_state=random_state, **params)

    raise AssertionError(f"Unhandled classifier: {classifier}")


def _classification_metrics(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    classifier: str,
    k: int,
    random_state: int | None,
    classifier_params: dict | None = None,
):
    if len(X_train) < 1:
        raise ValueError("Classifier needs at least one training sample.")

    classifier = normalize_classifier_name(classifier)
    config = CLASSIFIER_CONFIGS[classifier]
    k_eff = min(int(k), len(X_train))

    if bool(config["normalize_l2"]):
        normalizer = Normalizer(norm="l2")
        X_train = normalizer.fit_transform(X_train)
        X_test = normalizer.transform(X_test)

    model = _build_classifier(
        classifier,
        k_eff=k_eff,
        random_state=random_state,
        classifier_params=classifier_params,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", np.nan))
    return acc, balanced_acc, macro_f1, report


def evaluate_real_vs_augmented(
    df_train: pd.DataFrame,
    df_synth: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    classifier: str = "knn",
    k=K,
    label_col="labels",
    random_state: int | None = None,
    classifier_params: dict | None = None,
    knn_weights: str | None = None,
    knn_metric: str | None = None,
    knn_p: int | None = None,
) -> dict:
    classifier = normalize_classifier_name(classifier)
    resolved_params = dict(classifier_params or {})
    if classifier == "knn":
        if knn_weights is not None:
            resolved_params["weights"] = knn_weights
        if knn_metric is not None:
            resolved_params["metric"] = knn_metric
        if knn_p is not None:
            resolved_params["p"] = int(knn_p)

    X_real, y_real = _split_xy(df_train, label_col)
    X_test, y_test = _split_xy(df_test, label_col)

    if df_synth.empty:
        X_aug = X_real
        y_aug = y_real
    else:
        X_synth, y_synth = _split_xy(df_synth, label_col)
        X_aug = np.vstack([X_real, X_synth])
        y_aug = np.concatenate([y_real, y_synth])

    acc_real, balanced_acc_real, macro_f1_real, report_real = _classification_metrics(
        X_real,
        y_real,
        X_test,
        y_test,
        classifier=classifier,
        k=k,
        random_state=random_state,
        classifier_params=resolved_params,
    )
    acc_aug, balanced_acc_aug, macro_f1_aug, report_aug = _classification_metrics(
        X_aug,
        y_aug,
        X_test,
        y_test,
        classifier=classifier,
        k=k,
        random_state=random_state,
        classifier_params=resolved_params,
    )

    return {
        "acc_real": acc_real,
        "acc_aug": acc_aug,
        "delta_acc": float(acc_aug - acc_real),
        "balanced_acc_real": balanced_acc_real,
        "balanced_acc_aug": balanced_acc_aug,
        "delta_balanced_acc": float(balanced_acc_aug - balanced_acc_real),
        "macro_f1_real": macro_f1_real,
        "macro_f1_aug": macro_f1_aug,
        "delta_macro_f1": float(macro_f1_aug - macro_f1_real),
        "report_real": report_real,
        "report_aug": report_aug,
    }
