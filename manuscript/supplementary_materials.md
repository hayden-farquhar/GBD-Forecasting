# Supplementary Materials

## Foundation models, deep learning, and statistical methods for disease burden forecasting: a 10-model comparison using GBD 2023 data

Hayden Farquhar

---

## Table of Contents

- Supplementary Table S1. Full forecast matrix: DALY rates at milestone years (ARIMA)
- Supplementary Table S2. Expanding-window cross-validation results
- Supplementary Table S3. Diebold-Mariano pairwise test results
- Supplementary Table S4. Bootstrap 95% confidence intervals for MAE differences
- Supplementary Table S5. Bias analysis by model
- Supplementary Table S6. Augmented Dickey-Fuller stationarity tests
- Supplementary Table S7. Ljung-Box residual autocorrelation tests
- Supplementary Table S8. Residual normality tests
- Supplementary Table S9. Granger causality tests for covariates
- Supplementary Table S10. Leave-one-country-out validation (XGBoost)
- Supplementary Table S11. COVID-19 disruption quantification
- Supplementary Table S12. Model agreement at 2040 forecast horizon
- Supplementary Table S13. Sex-stratified projections for Australia
- Supplementary Table S14. Age-stratified projections for Australia
- Supplementary Table S15. Deaths (mortality rate) projections for Australia
- Supplementary Table S16. Sensitivity analysis: performance by training cutoff
- Supplementary Table S17. Multiple testing correction (Benjamini-Hochberg)
- Supplementary Table S18. Prediction interval width at 2040 (Prophet)
- Supplementary Table S19. Continuous Ranked Probability Score (CRPS)
- Supplementary Table S20. TRIPOD+AI reporting checklist
- Supplementary Methods. TFT architecture and training details
- Supplementary Figure S1. ACF/PACF diagnostic plots for representative series
- Supplementary Figure S2. Covariate correlation heatmap
- Supplementary Figure S3. Model agreement heatmap at 2040
- Supplementary Figure S4. COVID-19 disruption heatmap
- Supplementary Figure S5. Prediction interval width heatmap at 2040
- Supplementary Figure S6. Volatility vs forecast accuracy
- Supplementary Figure S7. Sensitivity analysis: MAE by training cutoff
- Supplementary Figure S8. Age-stratified projection heatmap (Australia)
- Supplementary Figure S9. Sex-stratified projection chart (Australia)
- Supplementary Figure S10. Cross-country divergence plots (depression, DM type 2, dementia)
- Supplementary Figure S11. All-model forecast comparison (Australia, 6 causes)
- Supplementary Figure S12. TFT temporal attention weights

---

## Supplementary Table S1. Full forecast matrix: projected DALY rates per 100,000 at milestone years (ARIMA, retrained on 1990-2023)

| Location | Cause | 2023 (observed) | 2025 | % change | 2030 | % change | 2035 | % change | 2040 | % change |
|----------|-------|-----------------|------|----------|------|----------|------|----------|------|----------|
| AUS | Anxiety disorders | 1314.1 | 1476.0 | +12.3 | 1910.2 | +45.4 | 2344.4 | +78.4 | 2778.7 | +111.5 |
| AUS | Asthma | 519.3 | 535.2 | +3.1 | 541.2 | +4.2 | 541.3 | +4.2 | 541.3 | +4.2 |
| AUS | Cancer (all) | 4345.3 | 4292.4 | -1.2 | 4330.5 | -0.3 | 4333.2 | -0.3 | 4333.4 | -0.3 |
| AUS | Breast cancer | 341.0 | 323.2 | -5.2 | 295.1 | -13.4 | 256.5 | -24.8 | 226.1 | -33.7 |
| AUS | Colorectal cancer | 478.4 | 466.2 | -2.5 | 442.8 | -7.4 | 422.7 | -11.6 | 404.3 | -15.5 |
| AUS | Liver cancer | 198.0 | 201.5 | +1.8 | 210.4 | +6.3 | 217.3 | +9.8 | 225.0 | +13.6 |
| AUS | Lung cancer | 733.1 | 730.2 | -0.4 | 717.0 | -2.2 | 717.1 | -2.2 | 716.7 | -2.2 |
| AUS | Prostate cancer | 311.1 | 312.5 | +0.4 | 314.7 | +1.1 | 316.0 | +1.6 | 316.6 | +1.8 |
| AUS | CKD | 379.3 | 383.0 | +1.0 | 387.8 | +2.2 | 387.8 | +2.2 | 387.8 | +2.2 |
| AUS | COPD | 732.0 | 729.7 | -0.3 | 728.7 | -0.5 | 728.7 | -0.5 | 728.7 | -0.5 |
| AUS | Dementia | 971.7 | 999.9 | +2.9 | 1064.7 | +9.6 | 1137.2 | +17.0 | 1215.1 | +25.0 |
| AUS | Depression | 751.6 | 752.9 | +0.2 | 755.7 | +0.5 | 755.7 | +0.5 | 755.7 | +0.5 |
| AUS | DM type 1 | 45.2 | 45.2 | -0.1 | 45.1 | -0.4 | 45.1 | -0.4 | 45.1 | -0.4 |
| AUS | DM type 2 | 612.5 | 610.7 | -0.3 | 602.6 | -1.6 | 602.6 | -1.6 | 602.6 | -1.6 |
| AUS | Hypertensive HD | 72.5 | 71.2 | -1.9 | 69.7 | -3.9 | 69.7 | -3.9 | 69.7 | -3.9 |
| AUS | IHD | 1375.6 | 1296.2 | -5.8 | 1113.4 | -19.1 | 948.7 | -31.0 | 798.8 | -41.9 |
| AUS | Self-harm | 544.6 | 527.5 | -3.1 | 511.1 | -6.2 | 511.1 | -6.2 | 511.1 | -6.2 |
| AUS | Stroke | 719.9 | 681.3 | -5.4 | 604.3 | -16.1 | 547.8 | -23.9 | 497.5 | -30.9 |

*Note: Table shows Australia only. Full data for all 5 countries (90 series) is available in the machine-readable file `supplementary_forecast_matrix.csv`. Caution: some individual ARIMA projections at the 2040 horizon are implausible for volatile series (see Supplementary Methods note on long-horizon ARIMA instability).*

---

## Supplementary Table S2. Expanding-window cross-validation (ARIMA, 3-year forecast horizon)

| Training cutoff | Mean MAE | Median MAE | SD of MAE | Series evaluated |
|-----------------|----------|------------|-----------|------------------|
| 2015 | 16.56 | 11.26 | 21.02 | 90 |
| 2016 | 16.36 | 10.64 | 16.76 | 90 |
| 2017 | 21.62 | 14.01 | 23.92 | 90 |
| 2018 | 30.60 | 14.31 | 39.40 | 90 |

*Note: Increasing error at the 2018 cutoff reflects inclusion of COVID-19-disrupted years (2019-2021) in the validation window.*

---

## Supplementary Table S3. Diebold-Mariano pairwise test results

| Model 1 | Model 2 | DM statistic | p-value | Significant (p<0.05) | Significant (BH-corrected) |
|---------|---------|-------------|---------|----------------------|---------------------------|
| ARIMA | Prophet | -2.503 | 0.012 | Yes | Yes |
| Prophet | TFT | 2.469 | 0.014 | Yes | Yes |
| ARIMA | XGBoost | -1.932 | 0.053 | No | No |
| XGBoost | TFT | 1.955 | 0.051 | No | No |
| XGBoost | Prophet | 0.434 | 0.665 | No | No |
| ARIMA | TFT | 0.065 | 0.948 | No | No |

*Note: Negative DM statistic indicates Model 1 is more accurate. Benjamini-Hochberg correction applied across all 11 hypothesis tests (6 DM + 5 bias tests).*

---

## Supplementary Table S4. Bootstrap 95% confidence intervals for pairwise MAE differences (10,000 resamples)

| Comparison | Mean MAE difference | 95% CI lower | 95% CI upper | Significant | Better model |
|------------|-------------------|-------------|-------------|-------------|--------------|
| Ensemble vs ARIMA | -5.07 | -7.66 | -2.50 | Yes | Ensemble |
| ARIMA vs TFT | -7.56 | -12.39 | -2.52 | Yes | ARIMA |
| ARIMA vs XGBoost | -10.54 | -19.14 | -3.73 | Yes | ARIMA |
| ARIMA vs Prophet | -17.84 | -25.10 | -11.16 | Yes | ARIMA |
| TFT vs Prophet | -10.28 | -18.44 | -2.85 | Yes | TFT |

*Note: Negative difference indicates the first-named model has lower MAE.*

---

## Supplementary Table S5. Bias analysis (one-sample t-test: H0: mean error = 0)

| Model | Mean bias (DALYs/100k) | Direction | t-statistic | p-value | Significant |
|-------|----------------------|-----------|-------------|---------|-------------|
| ARIMA | 20.95 | Under-predicts | 5.509 | <0.001 | Yes |
| XGBoost | 37.74 | Under-predicts | 7.039 | <0.001 | Yes |
| Prophet | 7.53 | Under-predicts | 1.427 | 0.154 | No |
| TFT | 14.12 | Under-predicts | 3.667 | <0.001 | Yes |
| Ensemble | 19.95 | Under-predicts | 5.777 | <0.001 | Yes |

*Note: Positive bias indicates actual values exceed predictions (under-prediction).*

---

## Supplementary Table S6. Augmented Dickey-Fuller stationarity tests

Of 90 series:
- **Stationary at level (p<0.05):** 8 (8.9%)
- **Stationary after first differencing:** 59 (65.6%)
- **Requiring second differencing or exhibiting structural breaks:** 23 (25.6%)

*Full per-series results available in `adf_stationarity_tests.csv`.*

---

## Supplementary Table S7. Ljung-Box residual autocorrelation tests (ARIMA models)

Of 90 fitted ARIMA models:
- **No significant residual autocorrelation (p>0.05):** 40 (44.4%)
- **Significant residual autocorrelation detected:** 50 (55.6%)

*Note: Tests conducted at up to 10 lags. The minimum p-value across lags was used to classify each series. Full results in `ljung_box_tests.csv`.*

---

## Supplementary Table S8. Residual normality tests

| Model | Shapiro-Wilk p | Jarque-Bera p | Normally distributed |
|-------|---------------|---------------|---------------------|
| ARIMA | <0.001 | <0.001 | No |
| XGBoost | <0.001 | <0.001 | No |
| Prophet | <0.001 | <0.001 | No |
| TFT | <0.001 | <0.001 | No |
| Ensemble | <0.001 | <0.001 | No |

*Note: Residuals were non-normal for all models, reflecting heavy-tailed error distributions driven by outlier series (particularly mental health conditions).*

---

## Supplementary Table S9. Granger causality tests

| Covariate | Series tested | Significant (p<0.05) | % significant | Median p-value |
|-----------|--------------|---------------------|---------------|----------------|
| Urbanisation rate | 90 | 53 | 58.9 | 0.033 |
| Population | 90 | 53 | 58.9 | 0.029 |
| Life expectancy | 90 | 44 | 48.9 | 0.055 |
| Health expenditure per capita | 90 | 37 | 41.1 | 0.091 |
| Physicians per 1,000 | 90 | 36 | 40.0 | 0.081 |
| GDP per capita (PPP) | 90 | 32 | 35.6 | 0.088 |

*Note: Bivariate Granger causality tests (F-test, maximum 3 lags) assessed whether each covariate had predictive value for DALY rates beyond the series' own lagged values.*

---

## Supplementary Table S10. Leave-one-country-out validation (XGBoost)

| Held-out country | Test observations | MAE | MAPE (%) |
|-----------------|-------------------|-----|----------|
| Australia | 90 | 39.48 | 2.65 |
| New Zealand | 90 | 25.85 | 3.50 |
| Canada | 90 | 17.11 | 1.75 |
| United Kingdom | 90 | 54.38 | 3.50 |
| United States | 90 | 37.46 | 2.39 |
| **Mean** | — | **34.86** | **2.76** |

*Note: Standard XGBoost (all countries in training) achieved MAE 53.35. The lower LOCO MAE (34.86) indicates that cross-country spatial prediction is easier than temporal forecasting; country-specific temporal dynamics, not cross-country patterns, are the primary source of forecast error.*

---

## Supplementary Table S11. COVID-19 disruption classification

Of 90 series:
- **Permanently shifted (>5% deviation from pre-2020 trend in 2023):** 45 (50.0%)
- **Recovered (<5% deviation):** 45 (50.0%)

**Five most positively disrupted series (increased burden):**

| Series | 2020 deviation (%) | 2023 deviation (%) | Status |
|--------|-------------------|-------------------|--------|
| GBR depression | +23.6 | +22.7 | Shifted |
| USA anxiety | +23.1 | +49.1 | Shifted |
| USA depression | +21.2 | +22.4 | Shifted |
| GBR anxiety | +20.0 | +37.8 | Shifted |
| CAN depression | +18.0 | +17.3 | Shifted |

**Five most negatively disrupted series (decreased burden):**

| Series | 2020 deviation (%) | 2023 deviation (%) | Status |
|--------|-------------------|-------------------|--------|
| GBR COPD | -7.1 | -3.7 | Recovered |
| AUS COPD | -7.0 | +1.8 | Recovered |
| CAN COPD | -6.7 | -2.0 | Recovered |
| NZL self-harm | -6.7 | -5.0 | Recovered |
| NZL stroke | -5.5 | +8.4 | Shifted |

*Full results in `covid_disruption.csv`.*

---

## Supplementary Table S12. Model agreement at 2040 forecast horizon

Inter-model coefficient of variation (CV) across ARIMA, XGBoost, and Prophet at 2040:

| Confidence level | CV range | Number of series | Percentage |
|-----------------|----------|-----------------|------------|
| High | <10% | 14 | 15.6% |
| Moderate | 10-25% | 32 | 35.6% |
| Low | 25-50% | 27 | 30.0% |
| Very low | >50% | 17 | 18.9% |

*Full per-series results in `model_agreement_2040.csv`.*

---

## Supplementary Table S13. Sex-stratified projections for Australia (ARIMA, 2023-2040)

| Cause | Male % change | Female % change | Sex gap (pp) |
|-------|--------------|----------------|-------------|
| CKD | +2.2 | -48.9 | 51.1 |
| Breast cancer | +2.3 | -33.5 | 35.8 |
| Dementia | +34.8 | +16.0 | 18.8 |
| Stroke | -15.5 | -33.7 | 18.2 |
| IHD | -39.4 | -54.2 | 14.9 |
| Self-harm | +1.7 | -5.3 | 7.0 |
| Hypertensive HD | -0.8 | -4.0 | 3.3 |
| Colorectal cancer | -13.2 | -14.6 | 1.4 |
| COPD | -0.9 | -1.7 | 0.8 |
| DM type 1 | +0.3 | -0.4 | 0.7 |
| Cancer (all) | +0.3 | +0.2 | 0.1 |
| Depression | -0.7 | +0.6 | -1.2 |
| DM type 2 | -5.1 | +1.1 | -6.2 |
| Asthma | -7.2 | +2.8 | -10.0 |
| Lung cancer | -30.9 | +2.6 | -33.5 |
| Liver cancer | -6.4 | +43.2 | -49.6 |
| Anxiety | +115.7 | +357.6 | -241.9 |

*Note: Sex gap = male % change minus female % change. Positive gap indicates males worsening relative to females. Prostate cancer excluded (sex-specific condition).*

---

## Supplementary Table S14. Age-stratified projections for Australia (ARIMA, 2023-2040, % change)

| Cause | 0-14 years | 15-49 years | 50-69 years | 70+ years |
|-------|-----------|------------|------------|----------|
| Anxiety | +3570.4 | +489.9 | +89.1 | +184.2 |
| Asthma | +75.9 | -6.1 | -13.8 | -27.7 |
| Cancer (all) | -0.7 | -56.0 | -43.3 | -3.5 |
| Breast cancer | — | -75.1 | -37.9 | -9.7 |
| CKD | +38.5 | +0.1 | -4.3 | +0.8 |
| Dementia | — | -45.1 | -4.6 | +3.9 |
| Depression | +1.0 | -0.1 | +0.2 | -0.5 |
| DM type 1 | +12.0 | -0.3 | +0.1 | -5.0 |
| DM type 2 | — | -8.1 | -2.5 | -2.3 |
| IHD | — | -23.4 | +34.5 | -59.1 |
| Self-harm | +2.4 | -7.4 | -2.3 | -0.1 |
| Stroke | -39.2 | -41.2 | -42.2 | -41.6 |

*Note: "—" indicates the cause is rare or absent in that age group (base rate near zero). Only causes with meaningful burden in the age group are shown. Full results in `age_stratified_changes.csv`.*

---

## Supplementary Table S15. Mortality rate projections for Australia (ARIMA, deaths per 100,000)

| Cause | Deaths/100k (2023) | Deaths/100k (2040) | Change (%) |
|-------|-------------------|-------------------|------------|
| Liver cancer | 9.2 | 13.4 | +45.3 |
| Dementia | 54.7 | 66.6 | +21.7 |
| Cancer (all) | 213.6 | 234.8 | +9.9 |
| Prostate cancer | 18.1 | 19.5 | +7.9 |
| Asthma | 1.7 | 1.8 | +4.3 |
| CKD | 20.6 | 21.4 | +3.7 |
| Lung cancer | 37.2 | 37.6 | +1.0 |
| Colorectal cancer | 24.2 | 24.4 | +0.6 |
| DM type 2 | 19.3 | 19.4 | +0.4 |
| Breast cancer | 14.0 | 14.0 | -0.0 |
| COPD | 35.6 | 34.9 | -1.9 |
| DM type 1 | 0.4 | 0.4 | -2.1 |
| Hypertensive HD | 4.8 | 4.6 | -2.8 |
| Self-harm | 12.3 | 11.9 | -3.4 |
| IHD | 83.5 | 58.7 | -29.7 |
| Stroke | 40.8 | 22.2 | -45.7 |

*Note: ARIMA fitted to GBD 2023 death rates (per 100,000). Anxiety and depression are excluded as they have negligible direct mortality in GBD. Directional consistency with DALY projections was observed for all causes.*

---

## Supplementary Table S16. Sensitivity analysis: mean MAE by training cutoff

| Training cutoff | Forecast horizon (years) | ARIMA MAE | XGBoost MAE |
|-----------------|-------------------------|-----------|-------------|
| 2016 | 7 | 37.87 | 48.22 |
| 2017 | 6 | 36.68 | 53.84 |
| 2018 | 5 | 42.83 | 53.35 |
| 2019 | 4 | 49.21 | 61.72 |
| 2020 | 3 | 169.85 | 62.90 |

*Note: ARIMA performance degraded sharply when trained through 2020, as COVID-19 disruptions in the training data distorted autoregressive trend estimates. XGBoost was more robust due to its non-parametric feature-based architecture.*

---

## Supplementary Table S17. Multiple testing correction (Benjamini-Hochberg FDR)

| Test | Raw p-value | BH threshold | Significant (raw) | Significant (BH-corrected) |
|------|------------|-------------|-------------------|---------------------------|
| Bias: ARIMA | <0.001 | 0.005 | Yes | Yes |
| Bias: XGBoost | <0.001 | 0.009 | Yes | Yes |
| Bias: Ensemble | <0.001 | 0.014 | Yes | Yes |
| Bias: TFT | <0.001 | 0.018 | Yes | Yes |
| DM: ARIMA vs Prophet | 0.012 | 0.023 | Yes | Yes |
| DM: Prophet vs TFT | 0.014 | 0.027 | Yes | Yes |
| DM: XGBoost vs TFT | 0.051 | 0.032 | No | No |
| DM: ARIMA vs XGBoost | 0.053 | 0.036 | No | No |
| Bias: Prophet | 0.154 | 0.041 | No | No |
| DM: XGBoost vs Prophet | 0.665 | 0.045 | No | No |
| DM: ARIMA vs TFT | 0.948 | 0.050 | No | No |

*Note: All 6 tests significant at the uncorrected level remained significant after BH-FDR correction at q<0.05.*

---

## Supplementary Table S18. Prediction interval width at 2040 (Prophet)

Summary statistics for Prophet 80% prediction interval width at the 2040 horizon:

| Statistic | PI width (DALYs/100k) | PI width (% of forecast) |
|-----------|----------------------|-------------------------|
| Mean | 271 | 55% |
| Median | 90 | 17% |
| Minimum | 1 | 2% |
| Maximum | 2,755 | 1,165% |

*Note: Extreme widths (>100% of forecast) were concentrated in GBR series with volatile post-COVID trajectories. Full per-series results in `pi_width_2040.csv`.*

---

## Supplementary Table S19. Continuous Ranked Probability Score (CRPS)

| Model | CRPS (DALYs/100k) |
|-------|-------------------|
| TFT (5-quantile) | 45.90 |
| Prophet (3-quantile approximation) | 51.73 |

*Note: Lower CRPS indicates better probabilistic forecast calibration. TFT outperformed Prophet despite both having poor interval coverage, indicating that TFT's quantile predictions were closer to the true distribution even though neither achieved nominal coverage rates.*

---

## Supplementary Figures

### Supplementary Figure S1. ACF and PACF diagnostic plots for six representative series

*File: `outputs/figures/acf_pacf_diagnostics.png`*

Autocorrelation function (ACF) and partial autocorrelation function (PACF) plots for six representative series spanning different disease categories and temporal dynamics. Dashed lines indicate 95% significance bounds. Strong positive autocorrelation at multiple lags (typical of trending series) is evident in most series, confirming the need for differencing. The PACF structure (significant at lag 1, decaying thereafter) supports ARIMA model specifications of order p=1-3.

### Supplementary Figure S2. Spearman correlation between covariates and DALY rates by cause

*File: `outputs/figures/covariate_correlation_heatmap.png`*

Correlation heatmap showing Spearman rank correlations between each World Bank covariate and DALY rates, computed separately for each of the 18 disease causes. Positive correlations (red) indicate covariates associated with higher burden; negative correlations (blue) indicate covariates associated with lower burden. Notable patterns include strong negative correlation between GDP/health expenditure and cardiovascular diseases (reflecting the epidemiological transition in wealthier economies) and positive correlation between these economic indicators and mental health conditions.

### Supplementary Figure S3. Model agreement heatmap at 2040

*File: `outputs/figures/model_agreement_heatmap.png`*

Coefficient of variation (CV%) across ARIMA, XGBoost, and Prophet forecast values at 2040, by cause and country. Green cells indicate high model agreement (<10% CV); red cells indicate high disagreement (>50% CV). GBR series show the highest disagreement, driven by volatile post-COVID trajectories that different model architectures extrapolate differently.

### Supplementary Figure S4. COVID-19 disruption heatmap

*File: `outputs/figures/covid_disruption_heatmap.png`*

Percentage deviation of observed 2020 DALY rates from the 2015-2019 linear trend extrapolation, by cause and country. Red cells indicate burden exceeded expectations (positive disruption); blue cells indicate burden fell below expectations (negative disruption). Depression and anxiety show consistent positive disruption across all countries. COPD shows consistent negative disruption, likely reflecting reduced respiratory infections during pandemic restrictions.

### Supplementary Figure S5. Prediction interval width at 2040

*File: `outputs/figures/pi_width_2040.png`*

Prophet 80% prediction interval width at 2040, expressed as a percentage of the point forecast, by cause and country. Values exceeding 100% indicate prediction intervals wider than the forecast itself, rendering them uninformative for planning purposes.

### Supplementary Figure S6. Historical volatility vs forecast accuracy

*File: `outputs/figures/volatility_vs_accuracy.png`*

Scatter plot of historical volatility (standard deviation of year-over-year percentage changes, 1990-2023) versus ARIMA forecast MAE on the test set (2019-2023), with points coloured by country. Spearman r=0.196 (p=0.064). The weak positive association suggests that historical volatility is a limited predictor of forecast accuracy; structural breaks (particularly COVID-19) contribute substantial error independently of historical variability.

### Supplementary Figure S7. Sensitivity analysis: MAE by training cutoff

*File: `outputs/figures/sensitivity_cutoff.png`*

Mean MAE for ARIMA and XGBoost models across five training cutoff years (2016-2020), all forecasting through 2023. ARIMA performance degrades sharply when trained through 2020 (MAE 169.85), while XGBoost remains relatively stable (MAE 62.90), demonstrating the vulnerability of autoregressive models to training-period disruptions.

### Supplementary Figure S8. Age-stratified projection heatmap (Australia)

*File: `outputs/figures/age_stratified_heatmap.png`*

Projected percentage change in DALY rates from 2023 to 2040 for Australia, by cause and age group (0-14, 15-49, 50-69, 70+). The extreme projected increase in anxiety for the 0-14 age group (+3,570%) is driven by a steep recent trend from a low base (3.6 DALYs/100,000 in 2023) and should be interpreted as indicative of trajectory direction rather than a precise point estimate.

### Supplementary Figure S9. Sex-stratified projections (Australia)

*File: `outputs/figures/sex_stratified_projections.png`*

Projected percentage change in DALY rates from 2023 to 2040 for Australia, by cause and sex. Blue bars represent males; red bars represent females. The largest sex disparities are observed for liver cancer (female +43.2%, male -6.4%), anxiety (female +357.6%, male +115.7%), and CKD (male +2.2%, female -48.9%).

### Supplementary Figure S10. Cross-country divergence plots

*Files: `outputs/figures/divergence_depression.png`, `divergence_dm_type2.png`, `divergence_dementia.png`*

Historical (1990-2023, solid lines) and projected (2024-2040, dashed lines) DALY rate trajectories for depression, diabetes mellitus type 2, and dementia across all five countries. These plots illustrate where country-specific trajectories diverge, highlighting conditions where Australia's projected path differs most from peer country means.

### Supplementary Figure S11. All-model forecasts for Australia (6 key causes)

*File: `outputs/figures/aus_all_models_forecast.png`*

Historical data (black solid line) and 2024-2040 forecasts from ARIMA (blue), XGBoost (green), and Prophet (orange) for six key causes in Australia. Model disagreement is most visible for depression (divergent forecast directions) and dementia (divergent growth rates), while IHD shows strong model consensus on continued decline.

### Supplementary Figure S12. TFT temporal attention weights

*File: `outputs/figures/tft_temporal_attention.png`*

Mean temporal attention weight across all prediction windows in the validation set. The TFT's self-attention mechanism assigns weights to each position in the encoder (historical input), indicating which past time points the model considers most informative for generating forecasts.

---

## Supplementary Table S20. TRIPOD+AI Reporting Checklist

| Item | Section | Page |
|------|---------|------|
| Title identifies as prediction model study | Title | 1 |
| Structured abstract | Abstract | 1 |
| Background and rationale | Introduction | 2 |
| Objectives | Introduction (final paragraph) | 2 |
| Source of data | Methods: Study design | 2 |
| Participants/setting | Methods: Study design (5 countries, 18 causes) | 2 |
| Outcome defined | Methods: Study design (DALY rates) | 2 |
| Predictors/covariates | Methods: Study design (6 World Bank covariates) | 2 |
| Sample size | Methods: Data preparation (90 series, 34 obs each) | 3 |
| Missing data handling | Methods: Study design (linear interpolation) | 3 |
| Model development | Methods: Forecasting models | 3 |
| Model performance measures | Methods: Statistical validation | 4 |
| Statistical methods | Methods: Statistical validation | 4 |
| Risk groups | Methods: Subgroup analyses (sex, age) | 4 |
| Model performance results | Results: Model comparison | 5 |
| Model updating | Methods: Production forecasts (retrained on full data) | 4 |
| AI: Model architecture | Methods: TFT paragraph | 3 |
| AI: Training procedure | Methods: TFT paragraph (loss, lr, early stopping) | 3 |
| AI: Hyperparameters | Methods: TFT paragraph (hidden size, heads, dropout) | 3 |
| AI: Computing environment | Methods: Software (Python 3.12, CPU, PyTorch 2.2) | 4 |
| AI: Code availability | Methods: Software (repository URL) | 4 |
| Limitations | Discussion: Limitations | 7 |
| Interpretation | Discussion | 6-7 |
| Implications | Discussion: final paragraphs | 7 |
| Reporting checklist | This table (Supplementary Table S20) | Supp |

---

## Supplementary Methods. TFT Architecture and Training Details

### Model architecture

The temporal fusion transformer comprised the following components:
- **Input embeddings:** Entity embeddings for static categorical variables (location: 5 categories; cause: 18 categories), each embedded to dimension 64.
- **Variable selection networks:** Separate networks for static, encoder (time-varying unknown), and decoder (time-varying known) inputs. Each used a GRN (gated residual network) with ELU activation to produce softmax-normalised importance weights.
- **LSTM encoder-decoder:** Single-layer LSTM with hidden size 64, processing the encoder sequence (up to 20 time steps) and decoder sequence (up to 5 time steps).
- **Multi-head attention:** 4 attention heads operating over the LSTM encoder outputs, with interpretable attention weights providing temporal importance scores.
- **Output layer:** Position-wise feed-forward network producing 5 quantile predictions (0.10, 0.25, 0.50, 0.75, 0.90) per forecast horizon step.

### Target normalisation

Each time series was independently normalised using a softplus group normaliser, which computes per-group (per series) location and scale parameters. This ensures the model sees comparable magnitudes across series with vastly different scales (e.g., IHD at ~1,376 vs DM type 1 at ~45 DALYs/100k).

### Training procedure

- **Loss function:** Quantile loss across 5 quantiles, summed over all forecast horizon steps
- **Optimiser:** Adam with initial learning rate 1e-3, reduced on plateau (patience 5 epochs, factor 0.1)
- **Gradient clipping:** Maximum gradient norm 0.5
- **Early stopping:** Patience 10 epochs monitoring validation loss
- **Batch size:** 64
- **Training device:** CPU (Intel, no GPU acceleration)
- **Training duration:** Approximately 35 epochs (early stopping triggered), ~10 minutes on CPU
- **Total parameters:** 306,255

### Convergence

Training converged after approximately 35 epochs with validation loss stabilising. No evidence of overfitting was observed (training and validation loss curves tracked closely after epoch 15).

### ARIMA order selection

The most common ARIMA orders selected across the 90 series were: (0,2,3) for 19 series, (1,1,3) for 17 series, (3,1,3) for 10 series, and (0,1,3) for 10 series. The predominance of d=1 and d=2 confirms the non-stationarity documented by ADF testing. The high q values (q=3 in most models) suggest significant moving-average structure in the differenced series.

### Note on long-horizon ARIMA instability

Some ARIMA models produced implausible long-horizon (2040) forecasts, including negative values (GBR asthma) and extreme exponential growth (CAN anxiety +7,544%, GBR depression +323%, GBR IHD +157%). These arise from ARIMA's tendency to extrapolate recent volatile trends (particularly post-COVID disruptions) indefinitely. For policy interpretation, the ensemble forecasts and the model agreement analysis (Supplementary Table S12) should be preferred over individual ARIMA projections for series with high model disagreement (CV >50%).
