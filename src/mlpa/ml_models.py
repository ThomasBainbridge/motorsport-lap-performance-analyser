from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STYLE_FEATURES = [
    "cmp_entry_speed_kph",
    "cmp_min_speed_kph",
    "cmp_exit_speed_kph",
    "cmp_brake_fraction",
    "cmp_mean_throttle_pct",
    "cmp_segment_length_m",
    "cmp_apex_to_exit_gain_kph",
]

REGRESSION_FEATURES = [
    "entry_speed_delta_kph",
    "min_speed_delta_kph",
    "exit_speed_delta_kph",
    "mean_speed_delta_kph",
    "brake_start_delta_m",
    "brake_end_delta_m",
    "apex_delta_m",
    "throttle_pickup_delta_m",
    "mean_throttle_delta_pct",
    "full_throttle_fraction_delta",
    "brake_fraction_delta",
    "entry_to_apex_drop_delta_kph",
    "apex_to_exit_gain_delta_kph",
]


@dataclass
class RegressionResult:
    model: Any | None
    metrics: dict[str, float | str]
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame



def _build_regression_candidates(random_state: int) -> dict[str, Any]:
    return {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }



def _make_regression_pipeline(estimator: Any, feature_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_cols,
            )
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])



def _cluster_archetype_name(center_row: pd.Series) -> str:
    min_speed = float(center_row.get("cmp_min_speed_kph", np.nan))
    exit_speed = float(center_row.get("cmp_exit_speed_kph", np.nan))
    brake_fraction = float(center_row.get("cmp_brake_fraction", np.nan))

    if pd.notna(min_speed) and min_speed < 130:
        return "Low-speed traction"
    if pd.notna(brake_fraction) and brake_fraction > 0.35:
        return "Heavy-braking"
    if pd.notna(exit_speed) and exit_speed > 210:
        return "High-speed flow"
    return "Balanced mid-speed"



def run_style_clustering(
    training_segment_features: pd.DataFrame,
    *,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = training_segment_features.copy()
    required = [c for c in STYLE_FEATURES if c in df.columns]
    empty_centers = pd.DataFrame(columns=["Cluster", "Archetype", "Feature", "CenterValue"])
    empty_profiles = pd.DataFrame(columns=["Cluster", "Archetype"] + required)

    if len(df) < max(n_clusters, 4) or len(required) < 4:
        df["StyleCluster"] = -1
        df["Archetype"] = "Unassigned"
        df["PC1"] = np.nan
        df["PC2"] = np.nan
        return df, empty_centers, empty_profiles

    X = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < max(n_clusters, 4):
        df["StyleCluster"] = -1
        df["Archetype"] = "Unassigned"
        df["PC1"] = np.nan
        df["PC2"] = np.nan
        return df, empty_centers, empty_profiles

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=30)
    labels = kmeans.fit_predict(X_scaled)
    df.loc[X.index, "StyleCluster"] = labels
    df["StyleCluster"] = df["StyleCluster"].fillna(-1).astype(int)

    pca = PCA(n_components=2, random_state=random_state)
    pcs = pca.fit_transform(X_scaled)
    df.loc[X.index, "PC1"] = pcs[:, 0]
    df.loc[X.index, "PC2"] = pcs[:, 1]

    centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    center_frame = pd.DataFrame(centers_unscaled, columns=required)
    archetype_names = {cluster_idx: _cluster_archetype_name(center_frame.iloc[cluster_idx]) for cluster_idx in range(len(center_frame))}
    df["Archetype"] = df["StyleCluster"].map(archetype_names).fillna("Unassigned")

    center_rows = []
    for cluster_idx, center in enumerate(centers_unscaled):
        for feature_name, center_value in zip(required, center):
            center_rows.append(
                {
                    "Cluster": cluster_idx,
                    "Archetype": archetype_names[cluster_idx],
                    "Feature": feature_name,
                    "CenterValue": float(center_value),
                }
            )
    centers_df = pd.DataFrame(center_rows)

    profile_df = (
        df[df["StyleCluster"] >= 0]
        .groupby(["StyleCluster", "Archetype"], as_index=False)[required]
        .mean()
        .rename(columns={"StyleCluster": "Cluster"})
        .sort_values("Cluster")
        .reset_index(drop=True)
    )
    return df, centers_df, profile_df



def train_time_loss_regressor(
    training_segment_features: pd.DataFrame,
    *,
    random_state: int = 42,
    cv_folds: int = 5,
    test_size: float = 0.30,
    candidate_names: list[str] | None = None,
) -> RegressionResult:
    df = training_segment_features.copy()
    feature_cols = [c for c in REGRESSION_FEATURES if c in df.columns]
    target_col = "time_loss_s"

    if len(df) < 12 or len(feature_cols) < 6:
        return RegressionResult(
            model=None,
            metrics={
                "n_rows": float(len(df)),
                "selected_model": "not_available",
                "cv_r2_mean": np.nan,
                "cv_r2_std": np.nan,
                "cv_mae_mean": np.nan,
                "cv_mae_std": np.nan,
                "test_r2": np.nan,
                "test_mae_s": np.nan,
            },
            feature_importance=pd.DataFrame(columns=["Feature", "Importance"]),
            predictions=pd.DataFrame(columns=["actual_time_loss_s", "predicted_time_loss_s", "subset"]),
        )

    usable = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan)
    usable = usable.dropna(subset=[target_col]).reset_index(drop=True)
    if len(usable) < 12:
        return RegressionResult(
            model=None,
            metrics={
                "n_rows": float(len(usable)),
                "selected_model": "not_available",
                "cv_r2_mean": np.nan,
                "cv_r2_std": np.nan,
                "cv_mae_mean": np.nan,
                "cv_mae_std": np.nan,
                "test_r2": np.nan,
                "test_mae_s": np.nan,
            },
            feature_importance=pd.DataFrame(columns=["Feature", "Importance"]),
            predictions=pd.DataFrame(columns=["actual_time_loss_s", "predicted_time_loss_s", "subset"]),
        )

    X = usable[feature_cols]
    y = usable[target_col]
    candidates = _build_regression_candidates(random_state=random_state)
    requested = candidate_names or ["ridge", "random_forest", "gradient_boosting"]
    requested = [name for name in requested if name in candidates]
    if not requested:
        requested = ["ridge", "random_forest", "gradient_boosting"]

    cv = KFold(n_splits=min(cv_folds, len(usable)), shuffle=True, random_state=random_state)
    model_scores: list[dict[str, float | str]] = []
    best_name = None
    best_score = np.inf
    best_pipeline = None

    for model_name in requested:
        pipeline = _make_regression_pipeline(candidates[model_name], feature_cols)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
            n_jobs=None,
        )
        mae_mean = float(-scores["test_mae"].mean())
        mae_std = float(scores["test_mae"].std())
        r2_mean = float(scores["test_r2"].mean())
        r2_std = float(scores["test_r2"].std())
        model_scores.append(
            {
                "model": model_name,
                "cv_r2_mean": r2_mean,
                "cv_r2_std": r2_std,
                "cv_mae_mean": mae_mean,
                "cv_mae_std": mae_std,
            }
        )
        if mae_mean < best_score:
            best_score = mae_mean
            best_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("Regression model selection failed unexpectedly.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    final_model = clone(best_pipeline)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)

    test_r2 = float(r2_score(y_test, preds))
    test_mae = float(mean_absolute_error(y_test, preds))

    perm = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=random_state,
        scoring="neg_mean_absolute_error",
    )
    importance_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": perm.importances_mean})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    score_df = pd.DataFrame(model_scores)
    selected_row = score_df.loc[score_df["model"] == best_name].iloc[0]
    metrics = {
        "n_rows": float(len(usable)),
        "selected_model": best_name,
        "cv_r2_mean": float(selected_row["cv_r2_mean"]),
        "cv_r2_std": float(selected_row["cv_r2_std"]),
        "cv_mae_mean": float(selected_row["cv_mae_mean"]),
        "cv_mae_std": float(selected_row["cv_mae_std"]),
        "test_r2": test_r2,
        "test_mae_s": test_mae,
    }

    predictions = pd.DataFrame(
        {
            "actual_time_loss_s": y_test.to_numpy(dtype=float),
            "predicted_time_loss_s": preds.astype(float),
            "subset": "test",
        }
    )
    train_preds = final_model.predict(X_train)
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "actual_time_loss_s": y_train.to_numpy(dtype=float),
                    "predicted_time_loss_s": train_preds.astype(float),
                    "subset": "train",
                }
            ),
            predictions,
        ],
        ignore_index=True,
    )

    return RegressionResult(
        model=final_model,
        metrics=metrics,
        feature_importance=importance_df,
        predictions=predictions,
    )
