"""
Facebook Prophet baseline with optional covariates.

Prophet adds automatic trend changepoint detection and can incorporate
external regressors. It fits one model per series (no cross-learning).
"""

import warnings

import numpy as np
import pandas as pd

from ..data.preprocessor import TRAIN_END, TEST_END

# Covariate columns that Prophet can use as additional regressors
REGRESSOR_COLS = [
    "gdp_per_capita_ppp",
    "health_expenditure_pc",
    "physicians_per_1000",
    "urbanisation_pct",
    "life_expectancy",
    "population",
]


def fit_prophet_single(
    series_df: pd.DataFrame,
    train_end: int = TRAIN_END,
    forecast_through: int = TEST_END,
    use_regressors: bool = True,
) -> dict:
    """Fit Prophet on a single series.

    Parameters
    ----------
    series_df : DataFrame for one group_id, must have year, value, and covariate columns
    train_end : last year of training data
    forecast_through : forecast through this year
    use_regressors : whether to add covariates as extra regressors

    Returns
    -------
    dict with keys: forecast_df, model
    """
    from prophet import Prophet

    series_df = series_df.sort_values("year").copy()

    # Prophet requires 'ds' and 'y' columns
    series_df["ds"] = pd.to_datetime(series_df["year"], format="%Y")
    series_df["y"] = series_df["value"]

    train = series_df[series_df["year"] <= train_end]
    future_years = list(range(train_end + 1, forecast_through + 1))

    # Identify available regressors
    available_regs = []
    if use_regressors:
        for col in REGRESSOR_COLS:
            if col in series_df.columns and series_df[col].notna().sum() > 10:
                available_regs.append(col)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        m = Prophet(
            yearly_seasonality=False,   # annual data, no sub-annual seasonality
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.1,
            seasonality_mode="additive",
        )

        for reg in available_regs:
            m.add_regressor(reg)

        train_prophet = train[["ds", "y"] + available_regs].copy()
        m.fit(train_prophet)

    # Build future dataframe
    # For regressors in the future period, use last known value (naive projection)
    future_df = pd.DataFrame({"ds": pd.to_datetime(future_years, format="%Y")})
    for reg in available_regs:
        future_rows = series_df[series_df["year"].isin(future_years)]
        if len(future_rows) > 0:
            future_df[reg] = future_rows[reg].values[:len(future_years)]
        else:
            future_df[reg] = train[reg].iloc[-1]

    forecast = m.predict(future_df)

    return {
        "forecast_df": forecast,
        "model": m,
        "regressors_used": available_regs,
    }


def run_prophet_baseline(
    df: pd.DataFrame,
    train_end: int = TRAIN_END,
    forecast_through: int = TEST_END,
    use_regressors: bool = True,
) -> pd.DataFrame:
    """Run Prophet on every series in the panel dataset.

    Parameters
    ----------
    df : modelling dataset from preprocessor
    train_end : last year of training data
    forecast_through : last year to forecast
    use_regressors : whether to use covariates as extra regressors

    Returns
    -------
    DataFrame with columns: group_id, year, actual, predicted, model
    """
    results = []

    for gid, group in df.groupby("group_id"):
        group = group.sort_values("year")
        test = group[group["year"] > train_end]

        try:
            fit = fit_prophet_single(
                group, train_end=train_end, forecast_through=forecast_through,
                use_regressors=use_regressors,
            )
        except Exception as e:
            print(f"  Prophet failed for {gid}: {e}")
            continue

        forecast = fit["forecast_df"]
        forecast["forecast_year"] = forecast["ds"].dt.year

        for _, row in forecast.iterrows():
            yr = int(row["forecast_year"])
            pred = row["yhat"]
            actual_row = test[test["year"] == yr]
            actual = actual_row["value"].values[0] if len(actual_row) > 0 else np.nan
            results.append({
                "group_id": gid,
                "year": yr,
                "actual": actual,
                "predicted": pred,
                "predicted_lower": row["yhat_lower"],
                "predicted_upper": row["yhat_upper"],
                "model": "Prophet",
            })

    return pd.DataFrame(results)
