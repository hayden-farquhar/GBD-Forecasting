"""
Model comparison and evaluation framework.

Computes standard forecasting metrics across all models and produces
comparison tables and plots for the manuscript.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute forecasting metrics per model and group.

    Parameters
    ----------
    results : DataFrame with columns group_id, year, actual, predicted, model

    Returns
    -------
    DataFrame with columns: model, group_id, mae, rmse, mape, n
    """
    rows = []
    for (model, gid), grp in results.groupby(["model", "group_id"]):
        valid = grp.dropna(subset=["actual", "predicted"])
        if len(valid) == 0:
            continue

        actual = valid["actual"].values
        predicted = valid["predicted"].values
        errors = actual - predicted

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        # MAPE: avoid division by zero
        nonzero = actual != 0
        if nonzero.sum() > 0:
            mape = np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100
        else:
            mape = np.nan

        rows.append({
            "model": model,
            "group_id": gid,
            "location": gid.split("_")[0],
            "cause": "_".join(gid.split("_")[1:]),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "n": len(valid),
        })

    return pd.DataFrame(rows)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model — the main results table.

    Returns
    -------
    DataFrame with one row per model: mean MAE, RMSE, MAPE across all series.
    """
    summary = metrics.groupby("model").agg(
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        mean_rmse=("rmse", "mean"),
        median_rmse=("rmse", "median"),
        mean_mape=("mape", "mean"),
        median_mape=("mape", "median"),
        n_series=("group_id", "nunique"),
    ).round(2)

    return summary.sort_values("mean_mae")


def metrics_by_cause(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model × cause — for disease-level comparison."""
    return metrics.groupby(["model", "cause"]).agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mape=("mape", "mean"),
    ).round(2).reset_index()


def metrics_by_country(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by model × country — for country-level comparison."""
    return metrics.groupby(["model", "location"]).agg(
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mape=("mape", "mean"),
    ).round(2).reset_index()


def plot_model_comparison_bar(
    metrics: pd.DataFrame,
    metric: str = "mae",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart: mean metric by model, overall."""
    summary = metrics.groupby("model")[metric].mean().sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    summary.plot(kind="barh", ax=ax, color=sns.color_palette("muted", len(summary)))
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison: Mean {metric.upper()} Across All Series")
    ax.axvline(x=summary.min(), color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_by_cause(
    metrics: pd.DataFrame,
    metric: str = "mae",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: metric by cause, grouped by model."""
    by_cause = metrics_by_cause(metrics)
    pivot = by_cause.pivot(index="cause", columns="model", values=f"mean_{metric}")

    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(f"Mean {metric.upper()}")
    ax.set_title(f"Model Comparison by Disease: Mean {metric.upper()}")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_forecast_vs_actual(
    results: pd.DataFrame,
    df_full: pd.DataFrame,
    group_ids: list[str] | None = None,
    n_cols: int = 3,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot forecast vs actual for selected series, with historical context.

    Parameters
    ----------
    results : predictions DataFrame (all models stacked)
    df_full : full modelling dataset (for historical values)
    group_ids : specific series to plot (default: 6 representative ones)
    """
    if group_ids is None:
        group_ids = ["AUS_ihd", "AUS_depression", "USA_dm_type2",
                     "GBR_dementia", "CAN_cancer_all", "NZL_self_harm"]
        group_ids = [g for g in group_ids if g in results["group_id"].unique()]

    n = len(group_ids)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes] if n == 1 else list(axes)
    else:
        axes = axes.flat

    models = sorted(results["model"].unique())
    colors = dict(zip(models, sns.color_palette("muted", len(models))))

    for ax, gid in zip(axes, group_ids):
        # Historical
        hist = df_full[df_full["group_id"] == gid].sort_values("year")
        ax.plot(hist["year"], hist["value"], "k-", linewidth=1.5, label="Historical")

        # Forecasts by model
        for model_name in models:
            pred = results[(results["group_id"] == gid) & (results["model"] == model_name)]
            pred = pred.sort_values("year")
            ax.plot(pred["year"], pred["predicted"], "--", color=colors[model_name],
                    linewidth=1.5, label=model_name)

        ax.axvline(x=hist[hist["split"] == "train"]["year"].max(), color="gray",
                   linestyle=":", alpha=0.7)
        ax.set_title(gid, fontsize=11)
        ax.set_xlabel("Year")
        ax.set_ylabel("DALYs/100k")
        ax.legend(fontsize=7)

    # Hide unused axes
    for ax in list(axes)[n:]:
        ax.set_visible(False)

    plt.suptitle("Forecast vs Actual: Selected Series", fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
