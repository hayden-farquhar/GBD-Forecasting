"""
Auto-ARIMA baseline: fits one ARIMA model per time series.

Uses statsmodels' auto-determination of (p,d,q) order via AIC minimisation.
This is the simplest baseline — purely univariate, no covariates, no cross-learning.
"""

import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from ..data.preprocessor import TRAIN_END, VAL_END, TEST_END


def _select_order(series: np.ndarray, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> tuple:
    """Select best (p,d,q) order by AIC with ADF-based differencing."""
    # Determine d via ADF test
    d = 0
    temp = series.copy()
    for _ in range(max_d):
        try:
            pval = adfuller(temp, maxlag=5, autolag="AIC")[1]
        except Exception:
            break
        if pval < 0.05:
            break
        temp = np.diff(temp)
        d += 1

    best_aic = np.inf
    best_order = (1, d, 0)

    for p, q in product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(series, order=(p, d, q), enforce_stationarity=False,
                                enforce_invertibility=False)
                result = model.fit(disp=False, maxiter=100)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q)
        except Exception:
            continue

    return best_order


def fit_arima_single(
    series: np.ndarray,
    order: tuple | None = None,
    forecast_steps: int = 5,
) -> dict:
    """Fit ARIMA on a single series and produce forecasts.

    Parameters
    ----------
    series : 1D array of observed values (training period)
    order : (p,d,q) tuple; if None, auto-selects via AIC
    forecast_steps : number of steps ahead to forecast

    Returns
    -------
    dict with keys: order, fitted_values, forecast, aic
    """
    if order is None:
        order = _select_order(series)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(series, order=order, enforce_stationarity=False,
                        enforce_invertibility=False)
        result = model.fit(disp=False, maxiter=200)

    fitted = result.fittedvalues
    forecast = result.forecast(steps=forecast_steps)

    return {
        "order": order,
        "fitted_values": fitted,
        "forecast": forecast,
        "aic": result.aic,
        "model": result,
    }


def run_arima_baseline(
    df: pd.DataFrame,
    train_end: int = TRAIN_END,
    forecast_through: int = TEST_END,
) -> pd.DataFrame:
    """Run auto-ARIMA on every series in the panel dataset.

    Parameters
    ----------
    df : modelling dataset from preprocessor (must have group_id, year, value)
    train_end : last year of training data
    forecast_through : last year to forecast to

    Returns
    -------
    DataFrame with columns: group_id, year, actual, predicted, model
    """
    results = []

    for gid, group in df.groupby("group_id"):
        group = group.sort_values("year")
        train = group[group["year"] <= train_end]
        test = group[group["year"] > train_end]

        if len(train) < 10:
            continue

        series = train["value"].values
        n_forecast = forecast_through - train_end

        try:
            fit = fit_arima_single(series, forecast_steps=n_forecast)
        except Exception as e:
            print(f"  ARIMA failed for {gid}: {e}")
            continue

        forecast_years = list(range(train_end + 1, forecast_through + 1))
        forecast_values = np.asarray(fit["forecast"])[:len(forecast_years)]

        for yr, pred in zip(forecast_years, forecast_values):
            actual_row = test[test["year"] == yr]
            actual = actual_row["value"].values[0] if len(actual_row) > 0 else np.nan
            results.append({
                "group_id": gid,
                "year": yr,
                "actual": actual,
                "predicted": pred,
                "model": "ARIMA",
                "order": str(fit["order"]),
            })

    return pd.DataFrame(results)
