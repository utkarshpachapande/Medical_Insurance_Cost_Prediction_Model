from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ARTIFACT_PATH, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET
from .data import load_data


@dataclass
class ModelBundle:
    best_model: object
    models: dict[str, object]
    metrics: pd.DataFrame
    residuals: np.ndarray
    feature_names: list[str]
    dataset_size: int


def _preprocessor(scale=False) -> ColumnTransformer:
    numeric_steps = []
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _models() -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline([
            ("preprocessor", _preprocessor(scale=True)),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", _preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=350,
                max_depth=12,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", _preprocessor()),
            ("model", GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.035,
                max_depth=3,
                min_samples_leaf=4,
                loss="huber",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Extra Trees": Pipeline([
            ("preprocessor", _preprocessor()),
            ("model", ExtraTreesRegressor(
                n_estimators=350,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def build_bundle(df: pd.DataFrame | None = None) -> ModelBundle:
    df = load_data() if df is None else df.copy()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records = []
    fitted = {}
    oof_by_model = {}

    for name, pipeline in _models().items():
        oof = cross_val_predict(pipeline, X_train, y_train, cv=folds, n_jobs=None)
        pipeline.fit(X_train, y_train)
        pred_test = pipeline.predict(X_test)
        oof_by_model[name] = oof
        fitted[name] = pipeline
        records.append({
            "Model": name,
            "CV MAE ($)": mean_absolute_error(y_train, oof),
            "CV RMSE ($)": mean_squared_error(y_train, oof) ** 0.5,
            "CV R²": r2_score(y_train, oof),
            "Holdout MAE ($)": mean_absolute_error(y_test, pred_test),
            "Holdout RMSE ($)": mean_squared_error(y_test, pred_test) ** 0.5,
            "Holdout R²": r2_score(y_test, pred_test),
        })

    metrics = pd.DataFrame(records).sort_values("CV RMSE ($)").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]

    # Refit the selected model on all available training data for production use.
    best_model = fitted[best_name]
    best_model.fit(X_train, y_train)
    residuals = y_train.to_numpy() - oof_by_model[best_name]

    # Lock the final model after evaluation on the held-out test set.
    final_model = _models()[best_name]
    final_model.fit(X, y)

    # Recover final feature names from the fitted encoder.
    preprocessor = final_model.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())

    return ModelBundle(
        best_model=final_model,
        models=fitted | {best_name: final_model},
        metrics=metrics,
        residuals=residuals,
        feature_names=feature_names,
        dataset_size=len(df),
    )


def save_bundle(bundle: ModelBundle, path: str | Path = ARTIFACT_PATH) -> None:
    # Fitted GradientBoostingRegressor objects retain a NumPy RandomState
    # used only during training. Removing it before serialization makes the
    # production artifact much more portable across NumPy patch versions.
    for model in bundle.models.values():
        estimator = getattr(model, "named_steps", {}).get("model")
        if estimator is not None and hasattr(estimator, "_rng"):
            estimator._rng = None

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: str | Path = ARTIFACT_PATH) -> ModelBundle:
    return joblib.load(path)


def train_and_save() -> ModelBundle:
    bundle = build_bundle()
    save_bundle(bundle)
    return bundle
