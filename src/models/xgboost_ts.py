"""
XGBoost time-series model with lag features.

Transforms the forecasting problem into supervised tabular learning by creating
lag features, rolling statistics, and trend features. Unlike ARIMA/Prophet,
this model is trained on ALL series pooled together — it can learn cross-series
patterns, making it a fairer comparison to the TFT.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from ..data.preprocessor import TRAIN_END, TEST_END

# Number of lag years to include as features
LAG_YEARS = [1, 2, 3, 5, 10]
# Rolling window sizes for mean/std features
ROLLING_WINDOWS = [3, 5, 10]


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and trend features to the panel dataset.

    Must be called on data sorted by (group_id, year).
    """
    df = df.copy()

    for lag in LAG_YEARS:
        df[f"lag_{lag}"] = df.groupby("group_id")["value"].shift(lag)

    for w in ROLLING_WINDOWS:
        roll = df.groupby("group_id")["value"].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"rolling_mean_{w}"] = roll

        roll_std = df.groupby("group_id")["value"].transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=2).std()
        )
        df[f"rolling_std_{w}"] = roll_std

    # Year-over-year change
    df["yoy_change"] = df.groupby("group_id")["value"].pct_change()

    # Trend: years since start
    df["trend"] = df["year"] - df["year"].min()

    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns for XGBoost."""
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    rolling_cols = [c for c in df.columns if c.startswith("rolling_")]
    covariate_cols = [
        "gdp_per_capita_ppp", "health_expenditure_pc", "physicians_per_1000",
        "urbanisation_pct", "life_expectancy", "population",
    ]
    covariate_cols = [c for c in covariate_cols if c in df.columns]
    encoded_cols = [c for c in df.columns if c.endswith("_encoded")]

    return lag_cols + rolling_cols + ["yoy_change", "trend"] + covariate_cols + encoded_cols


def train_xgboost(
    df: pd.DataFrame,
    train_end: int = TRAIN_END,
) -> dict:
    """Train a single pooled XGBoost model on all series.

    Parameters
    ----------
    df : modelling dataset with lag features already added
    train_end : last year of training data

    Returns
    -------
    dict with keys: model, feature_cols, label_encoders, feature_importances
    """
    df = df.copy()

    # Encode categoricals
    encoders = {}
    for col in ["location", "cause"]:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            encoders[col] = le

    feature_cols = _get_feature_cols(df)

    train = df[df["year"] <= train_end].dropna(subset=feature_cols + ["value"])

    X_train = train[feature_cols]
    y_train = train["value"]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    importances = dict(zip(feature_cols, model.feature_importances_))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "label_encoders": encoders,
        "feature_importances": importances,
    }


def predict_xgboost_recursive(
    df: pd.DataFrame,
    trained: dict,
    train_end: int = TRAIN_END,
    forecast_through: int = TEST_END,
) -> pd.DataFrame:
    """Generate recursive multi-step forecasts with the trained XGBoost.

    For each step beyond train_end, uses previous predictions as lag features
    (recursive forecasting strategy).
    """
    model = trained["model"]
    feature_cols = trained["feature_cols"]
    encoders = trained["label_encoders"]

    df = df.copy()
    for col, le in encoders.items():
        if f"{col}_encoded" not in df.columns:
            df[f"{col}_encoded"] = le.transform(df[col])

    results = []
    forecast_years = range(train_end + 1, forecast_through + 1)

    for gid, group in df.groupby("group_id"):
        group = group.sort_values("year").copy()

        # Start with actual historical values
        history = group[group["year"] <= train_end]["value"].tolist()
        all_values = list(history)

        for yr in forecast_years:
            row = group[group["year"] == yr]
            if len(row) == 0:
                continue

            row = row.iloc[0:1].copy()

            # Update lag features from history + predictions
            for lag in LAG_YEARS:
                idx = len(all_values) - lag
                row[f"lag_{lag}"] = all_values[idx] if idx >= 0 else np.nan

            # Update rolling features
            for w in ROLLING_WINDOWS:
                window_vals = all_values[-w:]
                row[f"rolling_mean_{w}"] = np.mean(window_vals)
                row[f"rolling_std_{w}"] = np.std(window_vals) if len(window_vals) > 1 else 0

            # YoY change
            if len(all_values) >= 2:
                row["yoy_change"] = (all_values[-1] - all_values[-2]) / all_values[-2] if all_values[-2] != 0 else 0
            else:
                row["yoy_change"] = 0

            X = row[feature_cols]
            if X.isna().any(axis=1).iloc[0]:
                X = X.fillna(0)

            pred = model.predict(X)[0]
            all_values.append(pred)

            actual_row = group[group["year"] == yr]
            actual = actual_row["value"].values[0] if len(actual_row) > 0 else np.nan

            results.append({
                "group_id": gid,
                "year": yr,
                "actual": actual,
                "predicted": pred,
                "model": "XGBoost",
            })

    return pd.DataFrame(results)


def run_xgboost_baseline(
    df: pd.DataFrame,
    train_end: int = TRAIN_END,
    forecast_through: int = TEST_END,
) -> pd.DataFrame:
    """End-to-end: add features, train, predict.

    Parameters
    ----------
    df : modelling dataset from preprocessor
    train_end : last year of training data
    forecast_through : last year to forecast

    Returns
    -------
    DataFrame with columns: group_id, year, actual, predicted, model
    """
    df_feat = create_lag_features(df)
    trained = train_xgboost(df_feat, train_end=train_end)
    predictions = predict_xgboost_recursive(df_feat, trained, train_end, forecast_through)

    print(f"  XGBoost top features: {sorted(trained['feature_importances'].items(), key=lambda x: -x[1])[:5]}")

    return predictions
