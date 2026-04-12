"""
End-to-end pipeline: load data → preprocess → train → forecast → analyse.

Run phases individually or use run_all() for the full pipeline.
"""

from pathlib import Path

from .data import load_gbd, load_world_bank_covariates, build_modelling_dataset
from .data.gbd_loader import load_gbd_risk_factors
from .data.covariate_loader import download_all_covariates
from .data.preprocessor import interpolate_missing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_and_prepare(
    gbd_path: str | Path = DATA_DIR / "gbd" / "gbd_dalys_rate.csv",
    risk_path: str | Path | None = DATA_DIR / "gbd" / "gbd_risk_factors.csv",
    covariate_cache: str | Path = DATA_DIR / "covariates",
    age_group: str = "all_ages",
    sex: str = "both",
):
    """Phase 1: load all data sources and merge into modelling dataset."""
    print("Loading GBD data...")
    gbd = load_gbd(gbd_path)

    print("Loading covariates...")
    covariates = download_all_covariates(cache_dir=covariate_cache)

    risk_factors = None
    if risk_path and Path(risk_path).exists():
        print("Loading risk factors...")
        risk_factors = load_gbd_risk_factors(risk_path)

    print("Building modelling dataset...")
    df = build_modelling_dataset(
        gbd=gbd,
        covariates=covariates,
        risk_factors=risk_factors,
        age_group=age_group,
        sex=sex,
    )
    df = interpolate_missing(df)

    n_series = df["group_id"].nunique()
    n_years = df["year"].nunique()
    print(f"Ready: {n_series} time series × {n_years} years = {len(df)} rows")

    return df
