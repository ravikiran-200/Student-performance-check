from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


GRADE_COLUMNS = {"g1", "g2", "g3", "grade", "final_grade", "finalgrade", "overall_average"}
PROTECTED_COLUMNS = {"sex", "gender", "address", "school", "age", "age_band"}
IDENTIFIER_COLUMNS = {"id", "student_id", "studentid", "name", "student_name"}
MODEL_NAMES = ("Logistic Regression", "Decision Tree", "Random Forest")


@dataclass
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    source_rows: pd.DataFrame
    audit_columns: list[str]
    target_label: str
    positive_rate: float
    task_kind: str
    dropped_rows: int


def _binary_outcome(
    series: pd.Series, risk_cutoff: float, treat_as_grade: bool = False
) -> tuple[pd.Series, str, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_values = numeric.dropna()
    if treat_as_grade or numeric_values.nunique() > 2:
        mask = numeric.notna()
        labels = (numeric[mask] < risk_cutoff).astype(int)
        return labels, f"At risk: grade < {risk_cutoff:g}", "grade"

    values = series.dropna()
    normalized = values.astype(str).str.strip().str.lower()
    positive_terms = {"1", "yes", "true", "at_risk", "at risk", "risk", "poor", "fail", "failed"}
    if normalized.isin(positive_terms).any():
        positive_values = set(values[normalized.isin(positive_terms)].unique())
        labels = series.dropna().isin(positive_values).astype(int)
        return labels, "At-risk class", "binary"
    unique = list(pd.unique(values))
    if len(unique) != 2:
        raise ValueError("Choose a numeric grade column or a target with exactly two classes.")
    unique_sorted = sorted(unique, key=lambda value: str(value))
    labels = (series.dropna() == unique_sorted[-1]).astype(int)
    return labels, f"Binary target (positive: {unique_sorted[-1]})", "binary"


def prepare_data(frame: pd.DataFrame, target_column: str, risk_cutoff: float = 10) -> PreparedData:
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is not in the data.")
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    if "age" in data.columns and "age_band" not in data.columns:
        age = pd.to_numeric(data["age"], errors="coerce")
        data["age_band"] = pd.cut(age, bins=[0, 17, 100], labels=["15–17", "18+"])

    outcome, target_label, task_kind = _binary_outcome(
        data[target_column],
        risk_cutoff,
        treat_as_grade=(
            target_column.lower() in GRADE_COLUMNS
            or target_column.lower().endswith("_marks")
        ),
    )
    valid_target = data[target_column].notna()
    clean = data.loc[valid_target].reset_index(drop=True)
    outcome = outcome.reset_index(drop=True).astype(int)
    if outcome.nunique() != 2:
        raise ValueError("Both at-risk and not-at-risk examples are needed. Adjust the grade cutoff or data.")

    columns_to_remove = {target_column}
    columns_to_remove.update(column for column in clean.columns if column.lower() in GRADE_COLUMNS)
    columns_to_remove.update(
        column for column in clean.columns if column.lower().endswith("_marks")
    )
    columns_to_remove.update(column for column in clean.columns if column.lower() in PROTECTED_COLUMNS)
    columns_to_remove.update(column for column in clean.columns if column.lower() in IDENTIFIER_COLUMNS)
    audit_columns = [
        column
        for column in ("gender", "sex", "address", "school", "age_band")
        if column in clean.columns and clean[column].dropna().nunique() > 1
    ]
    features = clean.drop(columns=list(columns_to_remove), errors="ignore")
    features = features.dropna(axis=1, how="all")
    features = features.loc[:, features.nunique(dropna=False) > 1]
    if features.shape[1] == 0:
        raise ValueError("No usable predictors remain after removing outcome, protected, and identifier columns.")
    if len(features) < 40:
        raise ValueError("At least 40 labeled examples are needed for a train/test comparison.")
    if outcome.value_counts().min() < 2:
        raise ValueError("Each class needs at least two examples for a stratified split.")
    return PreparedData(
        features=features.reset_index(drop=True),
        target=outcome,
        source_rows=clean,
        audit_columns=audit_columns,
        target_label=target_label,
        positive_rate=float(outcome.mean()),
        task_kind=task_kind,
        dropped_rows=int(frame[target_column].isna().sum()),
    )


def _make_pipeline(features: pd.DataFrame, model: Any) -> Pipeline:
    numeric_columns = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocessor = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _fairness_summary(y_true: pd.Series, predictions: np.ndarray, groups: pd.Series) -> tuple[dict, pd.DataFrame]:
    aligned_groups = (
        pd.Series(groups).reset_index(drop=True).fillna("Unknown").astype(str).rename("Group")
    )
    metric_frame = MetricFrame(
        metrics={"Selection rate": selection_rate, "True positive rate": true_positive_rate},
        y_true=np.asarray(y_true),
        y_pred=np.asarray(predictions),
        sensitive_features=aligned_groups,
    )
    by_group = metric_frame.by_group.reset_index()
    by_group = by_group.rename(columns={by_group.columns[0]: "Group"})
    gap = metric_frame.difference(method="between_groups")
    summary = {
        "demographic_parity_difference": float(gap["Selection rate"]),
        "equal_opportunity_difference": float(gap["True positive rate"]),
    }
    group_counts = aligned_groups.value_counts().rename_axis("Group").reset_index(name="Test examples")
    by_group = by_group.merge(group_counts, on="Group", how="left")
    at_risk_counts = (
        pd.DataFrame({"Group": aligned_groups, "At risk": np.asarray(y_true)})
        .groupby("Group", as_index=False)["At risk"]
        .sum()
        .rename(columns={"At risk": "At-risk examples"})
    )
    by_group = by_group.merge(at_risk_counts, on="Group", how="left")
    return summary, by_group


def _metric_summary(y_true: pd.Series, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC AUC": auc,
        "Balanced accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def run_benchmark(
    data: PreparedData,
    audit_column: str,
    threshold: float = 0.5,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    if audit_column not in data.source_rows.columns:
        raise ValueError("Select a protected-group column that is present in the dataset.")
    row_indices = np.arange(len(data.target))
    train_idx, test_idx = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=seed,
        stratify=data.target,
    )
    x_train, x_test = data.features.iloc[train_idx], data.features.iloc[test_idx]
    y_train, y_test = data.target.iloc[train_idx], data.target.iloc[test_idx]
    sensitive_train = data.source_rows.iloc[train_idx][audit_column].fillna("Unknown").astype(str)
    sensitive_test = data.source_rows.iloc[test_idx][audit_column].fillna("Unknown").astype(str)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=seed
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=220,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    runs: dict[str, dict[str, Any]] = {}
    for name, classifier in classifiers.items():
        model = _make_pipeline(x_train, classifier)
        fit_started = perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = perf_counter() - fit_started
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        inference_started = perf_counter()
        for _ in range(5):
            model.predict_proba(x_test)
        latency_ms = (perf_counter() - inference_started) * 1000 / (5 * max(1, len(x_test)))
        fairness, groups = _fairness_summary(y_test, predictions, sensitive_test)
        runs[name] = {
            "model": model,
            "metrics": _metric_summary(y_test, predictions, probabilities),
            "fairness": fairness,
            "group_metrics": groups,
            "predictions": predictions,
            "probabilities": probabilities,
            "fit_seconds": fit_seconds,
            "latency_ms": latency_ms,
        }

    fair_result = None
    selected_run = runs["Random Forest"]
    if sensitive_train.nunique() > 1 and y_train.nunique() == 2:
        try:
            optimizer = ThresholdOptimizer(
                estimator=selected_run["model"],
                constraints="true_positive_rate_parity",
                objective="balanced_accuracy_score",
                predict_method="predict_proba",
                prefit=True,
            )
            optimizer.fit(x_train, y_train, sensitive_features=sensitive_train)
            fair_predictions = optimizer.predict(x_test, sensitive_features=sensitive_test, random_state=seed)
            fairness, groups = _fairness_summary(y_test, fair_predictions, sensitive_test)
            fair_result = {
                "metrics": _metric_summary(y_test, np.asarray(fair_predictions), selected_run["probabilities"]),
                "fairness": fairness,
                "group_metrics": groups,
                "predictions": np.asarray(fair_predictions),
                "label": "Equal opportunity thresholding",
            }
        except Exception as error:
            fair_result = {"error": str(error)}

    return {
        "runs": runs,
        "fair_result": fair_result,
        "selected_model": "Random Forest",
        "selected_run": selected_run,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "test_indices": test_idx,
        "sensitive_test": sensitive_test,
        "audit_column": audit_column,
        "threshold": threshold,
    }


def local_shap_values(model: Pipeline, training_features: pd.DataFrame, student: pd.DataFrame) -> pd.DataFrame:
    """Return signed SHAP attributions for the positive (at-risk) outcome."""
    import shap

    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    transformed_background = np.asarray(preprocessor.transform(training_features.iloc[: min(80, len(training_features))]))
    transformed_student = np.asarray(preprocessor.transform(student))
    if isinstance(estimator, (DecisionTreeClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(
            estimator,
            data=transformed_background,
            feature_perturbation="interventional",
        )
        explanation = explainer(transformed_student, check_additivity=False)
    else:
        explainer = shap.LinearExplainer(estimator, transformed_background)
        explanation = explainer(transformed_student)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]
    names = preprocessor.get_feature_names_out()
    result = pd.DataFrame({"Feature": names, "SHAP contribution": values})
    return result.reindex(result["SHAP contribution"].abs().sort_values(ascending=False).index)
