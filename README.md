# Foundation models, deep learning, and statistical methods for disease burden forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19528301.svg)](https://doi.org/10.5281/zenodo.19528301)

Code and data for the paper:

> Farquhar H. Foundation models, deep learning, and statistical methods for disease burden forecasting: a 10-model comparison using GBD 2023 data. 2026.

## Overview

This repository contains the complete analysis pipeline for comparing nine forecasting models across four paradigms (statistical, machine learning, deep learning, foundation model) for Global Burden of Disease (GBD) DALY rate forecasting across five high-income countries (Australia, Canada, New Zealand, United Kingdom, United States).

### Key findings

- **PatchTST** achieved the best individual-model accuracy (MAE 36.70 DALYs/100,000)
- **Chronos** (zero-shot, no training) outperformed ARIMA, XGBoost, Prophet, TFT, N-BEATS, and N-HiTS
- A **5-model ensemble** (ETS + Prophet + XGBoost + TFT + PatchTST) achieved MAE 33.01
- Chronos provided the best-calibrated prediction intervals (62.9% coverage for nominal 80%)
- 502 ensemble combinations were systematically tested

## Repository structure

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── src/
│   ├── data/
│   │   ├── gbd_loader.py          # Parse GBD CSV exports
│   │   ├── covariate_loader.py    # World Bank API data retrieval
│   │   └── preprocessor.py        # Merge, split, normalise
│   ├── models/
│   │   ├── arima_baseline.py      # Auto-ARIMA per series
│   │   ├── prophet_model.py       # Prophet with covariates
│   │   ├── xgboost_ts.py          # XGBoost with lag features
│   │   └── tft.py                 # Temporal Fusion Transformer
│   └── analyse/
│       ├── comparison.py          # Model comparison metrics and plots
│       └── australia_focus.py     # Australia projections and divergence
├── data/
│   └── modelling_dataset.csv      # Processed analysis-ready dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── outputs/
│   ├── figures/                   # All manuscript figures
│   └── tables/                    # All supplementary tables (CSV)
└── manuscript/
    ├── manuscript.md
    └── supplementary_materials.md
```

## Data sources

- **GBD 2023 Results:** https://vizhub.healthdata.org/gbd-results/ (requires free registration)
- **World Bank Open Data:** https://data.worldbank.org/ (accessed via `wbgapi` Python package)

Raw GBD data is not included due to IHME terms of use. The processed modelling dataset (`data/modelling_dataset.csv`) contains DALY rates merged with covariates.

## Requirements

```
Python >= 3.12
torch >= 2.2
pytorch-forecasting >= 1.7
neuralforecast >= 3.0
chronos-forecasting >= 2.2
statsmodels >= 0.14
xgboost >= 2.0
prophet >= 1.3
pandas >= 3.0
numpy < 2
scikit-learn >= 1.8
scipy >= 1.12
matplotlib >= 3.10
seaborn >= 0.13
wbgapi >= 1.0
```

## Installation

```bash
git clone https://github.com/hayden-farquhar/GBD-Forecasting.git
cd gbd-forecasting
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproducing the analysis

### 1. Data preparation

```bash
# Download GBD data manually from vizhub.healthdata.org/gbd-results/
# Place CSV files in data/gbd/

# Pull World Bank covariates (automatic via API)
python -c "from src.data.covariate_loader import download_all_covariates; download_all_covariates()"
```

### 2. Model training and evaluation

```python
from src.pipeline import load_and_prepare
from src.models.arima_baseline import run_arima_baseline

# Load data
df = load_and_prepare()

# Run ARIMA baseline
arima_results = run_arima_baseline(df)

# See notebooks/ for full pipeline including all 9 models
```

### 3. Chronos zero-shot evaluation

```python
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", device_map="cpu")
context = torch.tensor(series_values, dtype=torch.float32)
forecast = pipeline.predict(context, prediction_length=5, num_samples=100)
```

## Models compared

| Model | Type | Training | Covariates |
|-------|------|----------|-----------|
| ARIMA | Statistical | Per-series | No |
| ETS | Statistical | Per-series | No |
| Prophet | Statistical | Per-series | Yes |
| XGBoost | ML | Pooled | Yes |
| TFT | Deep learning | Pooled | Yes |
| N-BEATS | Deep learning | Pooled | No |
| N-HiTS | Deep learning | Pooled | No |
| PatchTST | Deep learning | Pooled | No |
| Chronos | Foundation model | Zero-shot | No |

## Citation

If you use this code or data, please cite:

```bibtex
@article{farquhar2026gbd,
  title={Foundation models, deep learning, and statistical methods for disease burden forecasting: a 10-model comparison using GBD 2023 data},
  author={Farquhar, Hayden},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

Hayden Farquhar — hayden.farquhar@icloud.com
