"""
Merge GBD outcome data with covariates and prepare datasets for modelling.

Produces a single panel DataFrame suitable for:
- pytorch-forecasting TimeSeriesDataSet (TFT)
- statsmodels (ARIMA/ETS baselines)
- Prophet and XGBoost wrappers
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Temporal split boundaries
TRAIN_END = 2018
VAL_END = 2021
TEST_END = 2023

# Forecast horizon for TFT
MAX_PREDICTION_LENGTH = 17  # 2024–2040
MAX_ENCODER_LENGTH = 20     # 20-year lookback


def build_modelling_dataset(
    gbd: pd.DataFrame,
    covariates: pd.DataFrame,
    risk_factors: pd.DataFrame | None = None,
    age_group: str = "all_ages",
    sex: str = "both",
) -> pd.DataFrame:
    """Merge GBD outcomes with covariates into a single panel.

    Parameters
    ----------
    gbd : output of gbd_loader.load_gbd()
    covariates : output of covariate_loader.download_all_covariates()
    risk_factors : optional, output of gbd_loader.load_gbd_risk_factors()
    age_group : age group to filter to (default "all_ages")
    sex : sex to filter to (default "both")

    Returns
    -------
    Panel DataFrame with columns:
        time_idx, group_id, year, location, cause, value,
        + covariate columns
        + split (train/val/test)
    """
    # Filter GBD to requested demographic slice
    df = gbd.copy()
    df = df[df["age_group"] == age_group]
    df = df[df["sex"] == sex]

    # Keep only DALY rate as primary outcome
    if "measure" in df.columns:
        daly_mask = df["measure"].str.contains("DALY", case=False, na=False)
        if daly_mask.any():
            df = df[daly_mask]

    # Create group identifier: each unique location × cause is one time series
    df["group_id"] = df["location"] + "_" + df["cause"]

    # Merge covariates on (year, location)
    df = df.merge(covariates, on=["year", "location"], how="left")

    # Pivot risk factors wide and merge if available
    if risk_factors is not None and not risk_factors.empty:
        rf = risk_factors[risk_factors["sex"] == sex].copy()
        rf_wide = rf.pivot_table(
            index=["year", "location"],
            columns="risk_factor",
            values="value",
        ).reset_index()
        df = df.merge(rf_wide, on=["year", "location"], how="left")

    # Create integer time index (required by pytorch-forecasting)
    min_year = df["year"].min()
    df["time_idx"] = df["year"] - min_year

    # Assign temporal split
    df["split"] = "train"
    df.loc[df["year"] > TRAIN_END, "split"] = "val"
    df.loc[df["year"] > VAL_END, "split"] = "test"

    # Sort for time-series consistency
    df = df.sort_values(["group_id", "year"]).reset_index(drop=True)

    return df


def get_covariate_columns(df: pd.DataFrame) -> dict:
    """Identify covariate columns and classify them for TFT.

    Returns dict with keys:
        - time_varying_known: covariates known into the future (population, time features)
        - time_varying_unknown: covariates not known in advance (GDP, health expenditure, etc.)
        - static: group-level features (location, cause)
    """
    core_cols = {
        "time_idx", "group_id", "year", "location", "cause", "sex",
        "age_group", "value", "lower", "upper", "measure", "split",
    }

    covariate_cols = [c for c in df.columns if c not in core_cols]

    # Population and time features are "known" — we have UN projections for the future.
    # Everything else is "unknown" — we observe it historically but don't know future values.
    known = [c for c in covariate_cols if c in ("population", "time_idx")]
    unknown = [c for c in covariate_cols if c not in known]

    return {
        "time_varying_known": known + ["time_idx"],
        "time_varying_unknown": ["value"] + unknown,
        "static_categoricals": ["group_id", "location", "cause"],
    }


def scale_covariates(
    df: pd.DataFrame,
    covariate_cols: list[str],
    fit_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, StandardScaler]]:
    """Standardise covariate columns. Fits on training data only.

    Parameters
    ----------
    df : panel DataFrame
    covariate_cols : columns to scale
    fit_mask : boolean mask for rows to fit on (default: split == "train")

    Returns
    -------
    (scaled_df, scalers_dict)
    """
    if fit_mask is None:
        fit_mask = df["split"] == "train"

    df = df.copy()
    scalers = {}

    for col in covariate_cols:
        if col not in df.columns:
            continue
        scaler = StandardScaler()
        scaler.fit(df.loc[fit_mask, [col]].dropna())
        df[col] = scaler.transform(df[[col]])
        scalers[col] = scaler

    return df, scalers


def interpolate_missing(df: pd.DataFrame, group_col: str = "group_id") -> pd.DataFrame:
    """Forward-fill then back-fill missing values within each group.

    World Bank data has occasional gaps; linear interpolation within each
    country-cause series handles these cleanly.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    df = df.copy()
    df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    return df
