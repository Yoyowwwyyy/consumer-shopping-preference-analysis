from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.visualization import plot_ml_feature_importance


# * Constants

TARGET_COL = "shopping_preference"

DEMOGRAPHIC_COLS = [
    "age",
    "monthly_income",
    "income_group",
    "gender",
    "city_tier",
]

DIGITAL_COLS = [
    "daily_internet_hours",
    "smartphone_usage_years",
    "social_media_hours",
    "online_payment_trust_score",
    "tech_savvy_score",
]

PSYCHOLOGICAL_COLS = [
    "discount_sensitivity",
    "delivery_fee_sensitivity",
    "free_return_importance",
    "product_availability_online",
    "impulse_buying_score",
    "need_touch_feel_score",
    "brand_loyalty_score",
    "environmental_awareness",
    "time_pressure_level",
]


@dataclass
class MLRunResult:
    model_name: str
    feature_set_name: str
    used_features: List[str]
    train_size: int
    test_size: int
    accuracy: float
    class_labels: List[str]
    confusion_matrix: pd.DataFrame
    classification_report: pd.DataFrame
    feature_importance: pd.DataFrame


# Return stable feature groups aligned with the main analysis module
def get_ml_feature_groups() -> Dict[str, List[str]]:
    return {
        "demographic": DEMOGRAPHIC_COLS.copy(),
        "digital": DIGITAL_COLS.copy(),
        "psychological": PSYCHOLOGICAL_COLS.copy(),
    }


# Define staged feature sets for lightweight validation
def get_feature_set_configs() -> Dict[str, List[str]]:
    return {
        "demographic_only": DEMOGRAPHIC_COLS.copy(),
        "demo_digital": DEMOGRAPHIC_COLS + DIGITAL_COLS,
        "full_core": DEMOGRAPHIC_COLS + DIGITAL_COLS + PSYCHOLOGICAL_COLS,
    }


def _validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for ML module: {missing_cols}"
        )


def _split_feature_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


# Prepare X and y using already preprocessed clean_df
def prepare_ml_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:

    required_cols = feature_cols + [target_col]
    _validate_required_columns(df, required_cols)

    work_df = df[required_cols].copy()
    work_df = work_df.dropna(subset=[target_col])

    X = work_df[feature_cols].copy()
    y = work_df[target_col].copy()

    return X, y


# Build a column transformer
def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:

    numeric_cols, categorical_cols = _split_feature_types(X, list(X.columns))

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


# Train a multinomial logistic regression model with a preprocessing pipeline
def train_logistic_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict[str, Any]:

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


# Get final feature names after preprocessing
def _get_transformed_feature_names(
    pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]

    feature_names: List[str] = []
    feature_names.extend(numeric_cols)

    if categorical_cols:
        cat_pipeline = preprocessor.named_transformers_["cat"]
        encoder: OneHotEncoder = cat_pipeline.named_steps["onehot"]
        encoded_feature_names = list(
            encoder.get_feature_names_out(categorical_cols)
        )
        feature_names.extend(encoded_feature_names)

    return feature_names


# Compute interpretable feature importance for multinomial logistic regression
def get_feature_importance(
    pipeline: Pipeline,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:

    model: LogisticRegression = pipeline.named_steps["model"]
    feature_names = _get_transformed_feature_names(
        pipeline=pipeline,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    coef_matrix = model.coef_  # shape: [n_classes, n_features]
    importance = np.mean(np.abs(coef_matrix), axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)
    importance_df = importance_df[["rank", "feature", "importance"]].reset_index(drop=True)

    return importance_df


# Return lightweight evaluation results
def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:

    y_pred = pipeline.predict(X_test)
    labels = sorted(pd.Series(y_test).astype(str).unique().tolist())

    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T

    return {
        "accuracy": acc,
        "class_labels": labels,
        "confusion_matrix": cm_df,
        "classification_report": report_df,
    }


# Run one interpretable logistic-regression experiment
def run_single_ml_experiment(
    df: pd.DataFrame,
    feature_set_name: str,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> MLRunResult:

    X, y = prepare_ml_data(df=df, feature_cols=feature_cols, target_col=target_col)

    train_result = train_logistic_model(
        X=X,
        y=y,
        random_state=random_state,
    )

    pipeline = train_result["pipeline"]

    eval_result = evaluate_model(
        pipeline=pipeline,
        X_test=train_result["X_test"],
        y_test=train_result["y_test"],
    )

    importance_df = get_feature_importance(
        pipeline=pipeline,
        numeric_cols=train_result["numeric_cols"],
        categorical_cols=train_result["categorical_cols"],
    )

    return MLRunResult(
        model_name="Multinomial Logistic Regression",
        feature_set_name=feature_set_name,
        used_features=feature_cols,
        train_size=len(train_result["X_train"]),
        test_size=len(train_result["X_test"]),
        accuracy=float(eval_result["accuracy"]),
        class_labels=eval_result["class_labels"],
        confusion_matrix=eval_result["confusion_matrix"],
        classification_report=eval_result["classification_report"],
        feature_importance=importance_df,
    )


# Run staged experiments for the three planned feature sets
def compare_feature_sets(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> Dict[str, MLRunResult]:

    configs = get_feature_set_configs()
    results: Dict[str, MLRunResult] = {}

    for feature_set_name, feature_cols in configs.items():
        results[feature_set_name] = run_single_ml_experiment(
            df=df,
            feature_set_name=feature_set_name,
            feature_cols=feature_cols,
            target_col=target_col,
            random_state=random_state,
        )

    return results


# Create a compact summary table for notebook/report use
def build_ml_summary_table(results: Dict[str, MLRunResult]) -> pd.DataFrame:

    rows = []

    for feature_set_name, result in results.items():
        top_feature = None
        if not result.feature_importance.empty:
            top_feature = result.feature_importance.iloc[0]["feature"]

        rows.append(
            {
                "feature_set": feature_set_name,
                "model": result.model_name,
                "train_size": result.train_size,
                "test_size": result.test_size,
                "accuracy": round(result.accuracy, 4),
                "top_feature": top_feature,
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


# Return top-N important features from a single experiment
def extract_top_features(
    result: MLRunResult,
    top_n: int = 10,
) -> pd.DataFrame:

    return result.feature_importance.head(top_n).copy()


# Main entry point for the ML module
def run_ml_analysis(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> Dict[str, Any]:

    results = compare_feature_sets(
        df=df,
        target_col=target_col,
        random_state=random_state,
    )

    summary_table = build_ml_summary_table(results)

    return {
        "target_col": target_col,
        "feature_groups": get_ml_feature_groups(),
        "feature_set_configs": get_feature_set_configs(),
        "results": results,
        "summary_table": summary_table,
    }


# Return the best-performing experiment result based on accuracy.
def get_ml_best_result(ml_output: Dict[str, Any]) -> MLRunResult:
    results: Dict[str, MLRunResult] = ml_output["results"]

    best_key = max(results, key=lambda k: results[k].accuracy)
    return results[best_key]


# Return a specific experiment result by feature set name.
def get_ml_result_by_name(
    ml_output: Dict[str, Any],
    feature_set_name: str = "full_core",
) -> MLRunResult:

    results: Dict[str, MLRunResult] = ml_output["results"]

    if feature_set_name not in results:
        raise ValueError(
            f"Unknown feature_set_name: {feature_set_name}. "
            f"Available: {list(results.keys())}"
        )

    return results[feature_set_name]


# Return a clean top-N feature importance table for downstream visualization/reporting
def build_ml_feature_importance_table(
    ml_output: Dict[str, Any],
    feature_set_name: str = "full_core",
    top_n: int = 10,
) -> pd.DataFrame:

    result = get_ml_result_by_name(
        ml_output=ml_output,
        feature_set_name=feature_set_name,
    )

    top_df = result.feature_importance.head(top_n).copy()

    top_df = top_df.reset_index(drop=True)
    top_df["importance"] = top_df["importance"].round(6)

    return top_df


# Map a feature name to one of the project-level feature groups:
#   demographic / digital / psychological / derived_categorical / other
def classify_feature_group(feature_name: str) -> str:
    demographic_prefixes = [
        "age",
        "monthly_income",
        "income_group",
        "gender",
        "city_tier",
    ]

    digital_prefixes = [
        "daily_internet_hours",
        "smartphone_usage_years",
        "social_media_hours",
        "online_payment_trust_score",
        "tech_savvy_score",
    ]

    psychological_prefixes = [
        "discount_sensitivity",
        "delivery_fee_sensitivity",
        "free_return_importance",
        "product_availability_online",
        "impulse_buying_score",
        "need_touch_feel_score",
        "brand_loyalty_score",
        "environmental_awareness",
        "time_pressure_level",
    ]

    for prefix in demographic_prefixes:
        if feature_name.startswith(prefix):
            return "demographic"

    for prefix in digital_prefixes:
        if feature_name.startswith(prefix):
            return "digital"

    for prefix in psychological_prefixes:
        if feature_name.startswith(prefix):
            return "psychological"

    return "other"


# Return top-N important features with their broader feature group labels
def build_ml_feature_importance_with_groups(
    ml_output: Dict[str, Any],
    feature_set_name: str = "full_core",
    top_n: int = 10,
) -> pd.DataFrame:

    top_df = build_ml_feature_importance_table(
        ml_output=ml_output,
        feature_set_name=feature_set_name,
        top_n=top_n,
    )

    top_df["feature_group"] = top_df["feature"].apply(classify_feature_group)

    return top_df


# Aggregate total importance by feature group for quick interpretation
def build_ml_group_importance_summary(
    ml_output: Dict[str, Any],
    feature_set_name: str = "full_core",
) -> pd.DataFrame:

    result = get_ml_result_by_name(
        ml_output=ml_output,
        feature_set_name=feature_set_name,
    )

    importance_df = result.feature_importance.copy()
    importance_df["feature_group"] = importance_df["feature"].apply(classify_feature_group)

    group_summary = (
        importance_df.groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    group_summary["importance"] = group_summary["importance"].round(6)
    return group_summary


# Build a compact, notebook/report-friendly summary of the ML module
def build_ml_insight_summary(
    ml_output: Dict[str, Any],
    top_n: int = 10,
) -> Dict[str, Any]:

    summary_table = ml_output["summary_table"].copy()

    best_result = get_ml_best_result(ml_output)
    full_core_result = get_ml_result_by_name(ml_output, feature_set_name="full_core")

    top_features_df = build_ml_feature_importance_with_groups(
        ml_output=ml_output,
        feature_set_name="full_core",
        top_n=top_n,
    )

    group_summary_df = build_ml_group_importance_summary(
        ml_output=ml_output,
        feature_set_name="full_core",
    )

    return {
        "summary_table": summary_table,
        "best_feature_set": best_result.feature_set_name,
        "best_accuracy": round(best_result.accuracy, 4),
        "full_core_accuracy": round(full_core_result.accuracy, 4),
        "top_feature": top_features_df.iloc[0]["feature"] if not top_features_df.empty else None,
        "top_features": top_features_df,
        "group_importance_summary": group_summary_df,
    }


# Convenience wrapper for plotting ML top feature importance
def plot_ml_top_features(
    ml_output: Dict[str, Any],
    feature_set_name: str = "full_core",
    top_n: int = 10,
    save_fig: bool = False,
    filename: str = "ml_feature_importance_top10.png",
):

    top_features_df = build_ml_feature_importance_with_groups(
        ml_output=ml_output,
        feature_set_name=feature_set_name,
        top_n=top_n,
    )

    fig, ax = plot_ml_feature_importance(
        importance_df=top_features_df,
        top_n=top_n,
        save_fig=save_fig,
        filename=filename,
    )

    return fig, ax