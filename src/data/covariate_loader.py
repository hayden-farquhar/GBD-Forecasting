"""
Load covariate data from World Bank API and Our World in Data GitHub CSVs.

World Bank indicators are pulled via the wbgapi library.
OWID data is fetched directly from their GitHub-hosted CSV catalogue.
"""

from pathlib import Path

import pandas as pd

# ISO-3 codes for our 5 study countries
COUNTRIES = ["AUS", "NZL", "CAN", "GBR", "USA"]

# World Bank indicator codes and short names
WB_INDICATORS = {
    "NY.GDP.PCAP.PP.KD": "gdp_per_capita_ppp",      # GDP per capita, PPP (constant 2021 intl $)
    "SH.XPD.CHEX.PC.CD": "health_expenditure_pc",    # Current health expenditure per capita (USD)
    "SH.MED.PHYS.ZS": "physicians_per_1000",         # Physicians per 1,000 people
    "SP.URB.TOTL.IN.ZS": "urbanisation_pct",         # Urban population (% of total)
    "SP.DYN.LE00.IN": "life_expectancy",             # Life expectancy at birth
    "SP.POP.TOTL": "population",                     # Total population
}

# OWID GitHub CSV URLs for supplementary covariates
OWID_DATASETS = {
    "sdi": "https://raw.githubusercontent.com/owid/etl/master/etl/steps/data/garden/ihme_gbd/2024-05-20/gbd_cause/gbd_cause.csv",
}

YEAR_RANGE = range(1990, 2024)


def load_world_bank_covariates(
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Pull World Bank indicators for study countries, 1990-2023.

    Parameters
    ----------
    cache_dir : if provided, saves/loads a CSV cache to avoid repeated API calls

    Returns
    -------
    DataFrame with columns: year, location, <indicator_short_name>, ...
    Wide format — one row per country-year, one column per indicator.
    """
    cache_path = Path(cache_dir) / "world_bank_covariates.csv" if cache_dir else None

    if cache_path and cache_path.exists():
        return pd.read_csv(cache_path)

    try:
        import wbgapi as wb
    except ImportError:
        raise ImportError("Install wbgapi: pip install wbgapi")

    frames = []
    for indicator_code, short_name in WB_INDICATORS.items():
        try:
            raw = wb.data.DataFrame(
                indicator_code,
                economy=COUNTRIES,
                time=range(1990, 2024),
                labels=False,
            )
        except Exception as e:
            print(f"Warning: could not fetch {indicator_code} ({short_name}): {e}")
            continue

        # wbgapi returns countries as rows, years as columns (e.g. "YR1990")
        raw = raw.reset_index()
        raw = raw.rename(columns={"economy": "location"})

        # Melt year columns to long format
        year_cols = [c for c in raw.columns if c.startswith("YR")]
        melted = raw.melt(
            id_vars=["location"],
            value_vars=year_cols,
            var_name="year_raw",
            value_name=short_name,
        )
        melted["year"] = melted["year_raw"].str.replace("YR", "").astype(int)
        melted = melted.drop(columns=["year_raw"])
        melted = melted[melted["year"].isin(YEAR_RANGE)]
        frames.append(melted)

    if not frames:
        raise RuntimeError("No World Bank data fetched — check network connection")

    # Merge all indicators on (location, year)
    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["location", "year"], how="outer")

    result = result.sort_values(["location", "year"]).reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        print(f"Cached World Bank data to {cache_path}")

    return result


def load_owid_covariates() -> pd.DataFrame:
    """Load supplementary covariates from Our World in Data.

    Currently a placeholder — OWID's GBD-related datasets change structure
    frequently. This returns an empty DataFrame if the expected files are
    not available, and can be extended as needed.

    Returns
    -------
    DataFrame with columns: year, location, <covariate_name>, ...
    """
    # OWID publishes a comprehensive SDI dataset, but the exact URL/schema
    # shifts with their ETL pipeline. For now we rely on World Bank indicators
    # and GBD risk factors as covariates, and can add OWID sources later.
    return pd.DataFrame(columns=["year", "location"])


def download_all_covariates(
    cache_dir: str | Path = "data/covariates",
) -> pd.DataFrame:
    """Convenience function: download and merge all covariates.

    Parameters
    ----------
    cache_dir : directory for caching downloaded data

    Returns
    -------
    Merged DataFrame of all covariates, keyed on (year, location).
    """
    cache_dir = Path(cache_dir)

    wb = load_world_bank_covariates(cache_dir=cache_dir)
    owid = load_owid_covariates()

    if owid.empty:
        return wb

    return wb.merge(owid, on=["year", "location"], how="left")
