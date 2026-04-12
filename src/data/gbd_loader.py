"""
Load and standardise GBD 2023 CSV exports from IHME.

Expected input: CSV files downloaded from vizhub.healthdata.org/gbd-results
with columns including measure_name, location_name, sex_name, age_name,
cause_name, year, val, upper, lower.
"""

from pathlib import Path

import pandas as pd

# Mapping from GBD cause names to short identifiers used throughout the project.
# Keys are substrings matched case-insensitively against cause_name.
CAUSE_MAP = {
    "ischemic heart disease": "ihd",
    "stroke": "stroke",
    "hypertensive heart disease": "hypertensive_hd",
    "chronic obstructive pulmonary": "copd",
    "asthma": "asthma",
    "diabetes mellitus type 1": "dm_type1",
    "diabetes mellitus type 2": "dm_type2",
    "depressive disorders": "depression",
    "anxiety disorders": "anxiety",
    "self-harm": "self_harm",
    "chronic kidney disease": "ckd",
    "alzheimer": "dementia",
    "neoplasms": "cancer_all",
    "tracheal, bronchus, and lung cancer": "cancer_lung",
    "breast cancer": "cancer_breast",
    "colon and rectum cancer": "cancer_colorectal",
    "prostate cancer": "cancer_prostate",
    "liver cancer": "cancer_liver",
}

LOCATION_MAP = {
    "Australia": "AUS",
    "New Zealand": "NZL",
    "Canada": "CAN",
    "United Kingdom": "GBR",
    "United States of America": "USA",
}

AGE_MAP = {
    "All ages": "all_ages",
    "Under 5": "0_4",
    "5-14 years": "5_14",
    "0-14 years": "0_14",
    "15-49 years": "15_49",
    "50-69 years": "50_69",
    "70+ years": "70_plus",
    "80+ years": "80_plus",
}


def _match_cause(cause_name: str) -> str | None:
    """Match a GBD cause_name string to a short identifier."""
    lower = cause_name.lower()
    for pattern, short in CAUSE_MAP.items():
        if pattern in lower:
            return short
    return None


def load_gbd(
    filepath: str | Path,
    measure: str | None = None,
) -> pd.DataFrame:
    """Load a GBD CSV export and return a clean, long-format DataFrame.

    Parameters
    ----------
    filepath : path to the GBD CSV file
    measure : optional filter, e.g. "DALYs (Disability-Adjusted Life Years)"

    Returns
    -------
    DataFrame with columns:
        year, location, sex, age_group, cause, measure, value, lower, upper
    """
    df = pd.read_csv(filepath)

    # Standardise column names — IHME sometimes varies casing
    df.columns = df.columns.str.strip().str.lower()

    # Filter to measure if specified
    if measure is not None:
        df = df[df["measure_name"].str.contains(measure, case=False, na=False)]

    # Map location
    df = df[df["location_name"].isin(LOCATION_MAP)]
    df["location"] = df["location_name"].map(LOCATION_MAP)

    # Map cause
    df["cause"] = df["cause_name"].apply(_match_cause)
    df = df.dropna(subset=["cause"])

    # Map age group
    df["age_group"] = df["age_name"].map(AGE_MAP)
    df = df.dropna(subset=["age_group"])

    # Standardise sex
    df["sex"] = df["sex_name"].str.lower()

    # Extract core columns
    result = df[["year", "location", "sex", "age_group", "cause", "measure_name"]].copy()
    result = result.rename(columns={"measure_name": "measure"})
    result["value"] = pd.to_numeric(df["val"], errors="coerce")
    result["lower"] = pd.to_numeric(df["lower"], errors="coerce")
    result["upper"] = pd.to_numeric(df["upper"], errors="coerce")
    result["year"] = result["year"].astype(int)

    result = result.dropna(subset=["value"]).reset_index(drop=True)
    return result


def load_gbd_risk_factors(filepath: str | Path) -> pd.DataFrame:
    """Load GBD risk factor summary exposure values.

    Returns DataFrame with columns:
        year, location, sex, risk_factor, value, lower, upper
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    df = df[df["location_name"].isin(LOCATION_MAP)]
    df["location"] = df["location_name"].map(LOCATION_MAP)
    df["sex"] = df["sex_name"].str.lower()

    risk_map = {
        "smoking": "smoking_sev",
        "high body-mass index": "bmi_sev",
        "high fasting plasma glucose": "glucose_sev",
    }

    df["risk_factor"] = df["rei_name"].str.lower().map(
        {k.lower(): v for k, v in risk_map.items()}
    )
    df = df.dropna(subset=["risk_factor"])

    result = pd.DataFrame({
        "year": df["year"].astype(int),
        "location": df["location"],
        "sex": df["sex"],
        "risk_factor": df["risk_factor"],
        "value": pd.to_numeric(df["val"], errors="coerce"),
        "lower": pd.to_numeric(df["lower"], errors="coerce"),
        "upper": pd.to_numeric(df["upper"], errors="coerce"),
    })

    return result.dropna(subset=["value"]).reset_index(drop=True)
