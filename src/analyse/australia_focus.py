"""
Australia-focused analysis: projected disease burden, divergence from peers,
and disease category trends for the manuscript.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def compute_percent_change(forecasts: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """Compute projected % change from 2023 baseline for each series.

    Returns DataFrame with columns: group_id, location, cause, value_2023,
        value_2030, value_2040, pct_change_2030, pct_change_2040
    """
    rows = []
    for gid, grp in forecasts.groupby("group_id"):
        loc = gid.split("_")[0]
        cause = "_".join(gid.split("_")[1:])

        hist = historical[(historical["group_id"] == gid) & (historical["year"] == 2023)]
        if len(hist) == 0:
            continue
        val_2023 = hist["value"].values[0]

        val_2030 = grp[grp["year"] == 2030]["predicted"].values
        val_2040 = grp[grp["year"] == 2040]["predicted"].values
        val_2030 = val_2030[0] if len(val_2030) > 0 else np.nan
        val_2040 = val_2040[0] if len(val_2040) > 0 else np.nan

        rows.append({
            "group_id": gid,
            "location": loc,
            "cause": cause,
            "value_2023": val_2023,
            "value_2030": val_2030,
            "value_2040": val_2040,
            "pct_change_2030": (val_2030 - val_2023) / val_2023 * 100 if val_2023 > 0 else np.nan,
            "pct_change_2040": (val_2040 - val_2023) / val_2023 * 100 if val_2023 > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def australia_forecast_table(
    forecasts: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:
    """Summary table: Australia's projected burden for each cause at 2030 and 2040."""
    aus_fc = forecasts[forecasts["group_id"].str.startswith("AUS_")]
    changes = compute_percent_change(aus_fc, historical)
    changes = changes.sort_values("pct_change_2040", ascending=False)
    return changes


def plot_australia_projections(
    forecasts: pd.DataFrame,
    historical: pd.DataFrame,
    causes: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Multi-panel: historical + projected DALY trends for Australia, by cause."""
    if causes is None:
        causes = ["ihd", "depression", "cancer_all", "dm_type2", "dementia", "copd"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, cause in zip(axes.flat, causes):
        gid = f"AUS_{cause}"
        hist = historical[historical["group_id"] == gid].sort_values("year")
        fc = forecasts[forecasts["group_id"] == gid].sort_values("year")

        ax.plot(hist["year"], hist["value"], "k-", lw=2, label="Historical")
        ax.plot(fc["year"], fc["predicted"], "r--", lw=2, label="Forecast")

        if "q0.10" in fc.columns:
            ax.fill_between(fc["year"], fc["q0.10"], fc["q0.90"],
                           alpha=0.15, color="red", label="90% PI")

        ax.axvline(x=2023, color="gray", ls=":", alpha=0.7)
        ax.set_title(cause.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("DALYs/100k")
        ax.legend(fontsize=8)

    plt.suptitle("Australia: Projected DALY Burden 2024-2040", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_divergence(
    forecasts: pd.DataFrame,
    historical: pd.DataFrame,
    cause: str = "depression",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one cause across all 5 countries: historical + forecast trajectories."""
    countries = ["AUS", "NZL", "CAN", "GBR", "USA"]
    colors = {"AUS": "#C44E52", "NZL": "#55A868", "CAN": "#4C72B0",
              "GBR": "#DD8452", "USA": "#8172B3"}

    fig, ax = plt.subplots(figsize=(12, 6))
    for loc in countries:
        gid = f"{loc}_{cause}"
        hist = historical[historical["group_id"] == gid].sort_values("year")
        fc = forecasts[forecasts["group_id"] == gid].sort_values("year")

        ax.plot(hist["year"], hist["value"], "-", color=colors[loc], lw=2, label=f"{loc} historical")
        if len(fc) > 0:
            ax.plot(fc["year"], fc["predicted"], "--", color=colors[loc], lw=2, alpha=0.7)
            if "q0.10" in fc.columns:
                ax.fill_between(fc["year"], fc["q0.10"], fc["q0.90"],
                               alpha=0.1, color=colors[loc])

    ax.axvline(x=2023, color="gray", ls=":", alpha=0.7, label="Forecast start")
    ax.set_title(f"{cause.replace('_', ' ').title()}: Cross-Country Trajectories 1990-2040", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("DALYs per 100,000")
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compute_divergence_index(
    forecasts: pd.DataFrame,
    historical: pd.DataFrame,
    reference_country: str = "AUS",
) -> pd.DataFrame:
    """Compute how much Australia's projected trajectory diverges from peers.

    For each cause, computes the difference between Australia's 2040 % change
    and the mean % change of the other 4 countries. Positive = Australia is
    getting worse faster (or improving slower) than peers.
    """
    changes = compute_percent_change(forecasts, historical)
    aus = changes[changes["location"] == reference_country].copy()
    others = changes[changes["location"] != reference_country].copy()

    peer_mean = others.groupby("cause")["pct_change_2040"].mean().reset_index()
    peer_mean = peer_mean.rename(columns={"pct_change_2040": "peer_mean_pct_change_2040"})

    aus = aus.merge(peer_mean, on="cause", how="left")
    aus["divergence_2040"] = aus["pct_change_2040"] - aus["peer_mean_pct_change_2040"]
    aus = aus.sort_values("divergence_2040", ascending=False)

    return aus[["cause", "value_2023", "pct_change_2040", "peer_mean_pct_change_2040", "divergence_2040"]]
