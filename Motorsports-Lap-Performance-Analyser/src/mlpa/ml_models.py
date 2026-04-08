from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


STYLE_FEATURES = [
    "cmp_entry_speed_kph",
    "cmp_min_speed_kph",
    "cmp_exit_speed_kph",
    "cmp_brake_fraction",
    "cmp_mean_throttle_pct",
    "cmp_segment_length_m",
]

REGRESSION_FEATURES = [
    "entry_speed_delta_kph",
    "min_speed_delta_kph",
    "exit_speed_delta_kph",
    "brake_start_delta_m",
    "apex_delta_m",
    "throttle_pickup_delta_m",
    "mean_throttle_delta_pct",
    "brake_fraction_delta",
]


@dataclass
class RegressionResult:
    model: Any | None
    metrics: dict[str, float]
    feature_importance: pd.DataFrame


def run_style_clustering(
    training_segment_features: pd.DataFrame,
    *,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = training_segment_features.copy()
    required = [c for c in STYLE_FEATURES if c in df.columns]
    if len(df) < max(n_clusters, 3) or len(required) < 3:
        df["StyleCluster"] = -1
        return df, pd.DataFrame(columns=["Cluster", "Feature", "CenterValue"])

    X = df[required].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < max(n_clusters, 3):
        df["StyleCluster"] = -1
        return df, pd.DataFrame(columns=["Cluster", "Feature", "CenterValue"])

    kmeans = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("cluster", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)),
        ]
    )
    labels = kmeans.fit_predict(X)
    df.loc[X.index, "StyleCluster"] = labels
    df["StyleCluster"] = df["StyleCluster"].fillna(-1).astype(int)

    scaler = kmeans.named_steps["scaler"]
    clusterer = kmeans.named_steps["cluster"]
    centers_unscaled = scaler.inverse_transform(clusterer.cluster_centers_)

    center_rows = []
    for cluster_idx, center in enumerate(centers_unscaled):
        for feature_name, center_value in zip(required, center):
            center_rows.append(
                {
                    "Cluster": cluster_idx,
                    "Feature": feature_name,
                    "CenterValue": float(center_value),
                }
            )
    centers_df = pd.DataFrame(center_rows)
    return df, centers_df


def train_time_loss_regressor(
    training_segment_features: pd.DataFrame,
    *,
    random_state: int = 42,
) -> RegressionResult:
    df = training_segment_features.copy()
    feature_cols = [c for c in REGRESSION_FEATURES if c in df.columns]
    target_col = "time_loss_s"

    if len(df) < 10 or len(feature_cols) < 4:
        return RegressionResult(
            model=None,
            metrics={"n_rows": float(len(df)), "r2": np.nan, "mae_s": np.nan},
            feature_importance=pd.DataFrame(columns=["Feature", "Importance"]),
        )

    usable = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(usable) < 10:
        return RegressionResult(
            model=None,
            metrics={"n_rows": float(len(usable)), "r2": np.nan, "mae_s": np.nan},
            feature_importance=pd.DataFrame(columns=["Feature", "Importance"]),
        )

    X = usable[feature_cols]
    y = usable[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_cols)],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
            )),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rf = model.named_steps["rf"]
    importance_df = (
        pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    metrics = {
        "n_rows": float(len(usable)),
        "r2": float(r2_score(y_test, preds)),
        "mae_s": float(mean_absolute_error(y_test, preds)),
    }
    return RegressionResult(model=model, metrics=metrics, feature_importance=importance_df)
