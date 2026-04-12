# Foundation models, deep learning, and statistical methods for disease burden forecasting: a 10-model comparison using GBD 2023 data

Hayden Farquhar (ORCID: 0009-0002-6226-440X)

Independent Researcher, Finley, NSW, Australia

Correspondence: hayden.farquhar@icloud.com

---

## Abstract

**Background:** Forecasting disease burden informs health system planning, yet most projections use single statistical methods. Modern deep learning architectures and pretrained foundation models have not been systematically compared for this task.

**Methods:** We obtained GBD 2023 disability-adjusted life year (DALY) rates for 18 causes across five high-income countries (1990-2023; 90 time series). Nine forecasting models spanning four paradigms — statistical (auto-ARIMA, ETS, Prophet), machine learning (XGBoost), deep learning (temporal fusion transformer [TFT], N-BEATS, N-HiTS, PatchTST), and a pretrained foundation model (Chronos, zero-shot) — were evaluated against a naive persistence baseline. We tested all 502 possible inverse-MAE-weighted ensemble combinations. Models were retrained on 1990-2023 to project Australian burden through 2040 with sex and age stratification. Validation included expanding-window cross-validation, Diebold-Mariano tests, bootstrap confidence intervals, and prediction interval calibration assessment.

**Results:** PatchTST achieved the lowest individual-model MAE (36.70 DALYs/100,000; skill score 0.144), followed by Chronos zero-shot (38.81; 0.095) and ETS (39.64; 0.075). A 5-model ensemble (ETS + Prophet + XGBoost + TFT + PatchTST) achieved the best overall MAE (33.01; skill 0.230). Chronos — requiring no training — produced the best-calibrated prediction intervals (62.9% empirical coverage for nominal 80%), far exceeding TFT (19.6%) and Prophet (23.1%). TFT variable selection identified health expenditure and GDP as the most important covariates but ranked 8th of 9 models for accuracy. For Australia, anxiety (+111.5%) and dementia (+25.0%) showed the largest projected increases to 2040, while IHD (-41.9%) and stroke (-30.9%) showed the largest decreases. Female liver cancer DALYs were projected to increase 43.2% versus a 6.4% male decrease.

**Conclusions:** PatchTST and pretrained foundation models outperformed all conventional methods for GBD forecasting, while multi-model ensembles provided the best overall accuracy. The zero-shot performance of Chronos demonstrates that pretrained time-series models transfer effectively to epidemiological domains without task-specific training. Projected rises in mental health and neurodegenerative burden in Australia, with marked sex differences, have implications for health system planning.

**Keywords:** disease burden forecasting; foundation models; PatchTST; Chronos; temporal fusion transformer; GBD 2023; DALY; Australia

---

## Introduction

The Global Burden of Disease (GBD) study estimates morbidity and mortality from hundreds of diseases across 204 countries from 1990 to the present [1,2]. These estimates inform health policy, resource allocation, and priority setting [3]. However, the utility of GBD data for prospective planning depends on forecasting disease burden trajectories — a task that remains methodologically challenging.

IHME produces long-range forecasts using DisMod-MR and scenario-based statistical models [4,5], while most independent GBD forecasting studies use ARIMA [6-8] or, more recently, single machine learning methods such as XGBoost [9] or transformers [10]. A 2025 scoping review identified only nine studies applying AI/ML to GBD data [11]. No study has systematically compared modern deep learning architectures — including temporal fusion transformers (TFTs) [12], N-BEATS [13], N-HiTS [14], and PatchTST [15] — against statistical baselines for multi-cause, multi-country disease burden prediction. Furthermore, pretrained time-series foundation models such as Chronos [16], which can produce forecasts with no task-specific training, have not been evaluated on GBD data.

The COVID-19 pandemic disrupted disease burden trajectories [2], particularly for mental health [17] and respiratory disease [18]. These disruptions challenge forecasting models to distinguish temporary shocks from permanent shifts, and make rigorous model comparison across architectures especially important.

Our objectives were to: (1) compare nine forecasting models spanning four paradigms (statistical, ML, deep learning, foundation model) for GBD DALY rates across five high-income countries; (2) identify optimal ensemble combinations from 502 tested model subsets; (3) generate projections to 2040 for Australia with sex- and age-stratified trajectories; and (4) assess prediction interval calibration across probabilistic models.

---

## Methods

### Study design and data sources

This was a retrospective time-series forecasting study using publicly available aggregate data. No ethics approval was required. The study follows the TRIPOD+AI guideline [19]; a completed checklist is provided in Supplementary Table S20.

DALY rates (per 100,000) were obtained from the GBD 2023 Results Tool [1] for 1990-2023 across five countries (Australia, Canada, New Zealand, United Kingdom, United States) and 18 Level 3 causes: cardiovascular (ischaemic heart disease [IHD], stroke, hypertensive heart disease), respiratory (COPD, asthma), metabolic (diabetes mellitus type 1 and 2, chronic kidney disease [CKD]), mental health (depressive disorders, anxiety disorders, self-harm), neurological (dementia), and cancer (all neoplasms plus lung, breast, colorectal, prostate, liver). GBD point estimates were used; uncertainty intervals were not propagated. Six covariates were obtained from World Bank Open Data [20]: GDP per capita, health expenditure per capita, physician density, urbanisation, life expectancy, and population.

### Data preparation

Each country-cause combination constituted one time series (n=90 series, 34 annual observations). Data were split temporally: training (1990-2018), validation (2019-2021), and test (2022-2023). Augmented Dickey-Fuller tests [21] showed 8.9% of series stationary at level and 65.6% after first differencing.

### Forecasting models

Nine models were evaluated, spanning four paradigms:

**Statistical methods.** (i) Auto-ARIMA: per-series (p,d,q) order selected by AIC [22]. (ii) Exponential smoothing (ETS): per-series, optimal trend component (additive, damped, or none) selected by AIC [23]. (iii) Prophet: per-series with covariates as regressors [24].

**Machine learning.** (iv) XGBoost: pooled across all series with lag features, rolling statistics, covariates, and encoded identifiers [25].

**Deep learning.** (v) TFT: trained on all 90 series simultaneously (~306K parameters, quantile loss, early stopping) via pytorch-forecasting [12,26]. (vi) N-BEATS: pure deep learning univariate model with interpretable trend/seasonality decomposition [13]. (vii) N-HiTS: hierarchical interpolation architecture for efficient long-horizon forecasting [14]. (viii) PatchTST: channel-independent patched transformer treating time series as sequences of subseries patches [15]. Models (vi)-(viii) were trained via neuralforecast [27] with 500 training steps and input size 20.

**Foundation model.** (ix) Chronos: pretrained T5-based foundation model (chronos-t5-small) applied zero-shot with no task-specific training [16; preprint]. For each series, the historical values were provided as context and 100 forecast samples were drawn to produce median predictions and prediction intervals.

**Ensemble.** Per-series inverse-MAE weights from validation combined model subsets. All 502 possible 2-to-9-model combinations from the 9 models (excluding only the naive baseline) were tested.

**Naive baseline.** Last observed value (2018) persisted forward.

### Production forecasts and subgroup analyses

All models were retrained on 1990-2023 to forecast 2024-2040 (covariates held at 2023 levels). Sex- and age-stratified analyses were conducted for Australia using ARIMA. As a robustness check, ARIMA was also fitted to GBD 2023 mortality rates.

### Statistical validation

Expanding-window cross-validation (cutoffs 2015-2018), Diebold-Mariano tests [28], bootstrap 95% CIs (10,000 resamples), bias analysis (one-sample t-tests), Ljung-Box residual diagnostics [29], Granger causality tests [30], leave-one-country-out validation, sensitivity analysis across cutoffs 2016-2020, and Benjamini-Hochberg FDR correction [31] were applied. Prediction interval calibration was assessed for all probabilistic models (TFT, Prophet, Chronos). CRPS was computed for TFT and Prophet. Skill scores were defined as 1 - MAE_model/MAE_naive [23].

### COVID-19 disruption analysis

Percentage deviations from a 2015-2019 linear trend extrapolation quantified disruption. Series were classified as recovered (<5% deviation in 2023) or permanently shifted.

### Software

Python 3.12 with statsmodels 0.14, XGBoost 2.0, Prophet 1.3, pytorch-forecasting 1.7, neuralforecast 3.0, chronos-forecasting 2.2, PyTorch 2.2 [22,24-27,32,33]. Code available at [repository URL to be added].

### AI disclosure

AI tools (Claude, Anthropic) assisted with code development and manuscript drafting. All results were verified by the author.

---

## Results

### 10-model comparison

PatchTST achieved the lowest individual-model MAE (36.70 DALYs/100,000; skill score 0.144 over naive persistence), followed by Chronos zero-shot (38.81; 0.095), ETS (39.64; 0.075), and ARIMA (42.81; 0.001; Table 1, Figure 1). TFT ranked 8th of 9 models (MAE 50.37). Only PatchTST, Chronos, and ETS achieved positive skill scores; all other models, including ARIMA, performed at or below the naive baseline.

**Table 1. Forecasting performance across 90 series (2019-2023)**

| Model | Type | Mean MAE | Median MAE | MAPE (%) | Skill |
|-------|------|---------|------------|----------|-------|
| PatchTST | Deep learning | 36.70 | 19.63 | 4.22 | 0.144 |
| Chronos | Foundation model (zero-shot) | 38.81 | 22.35 | 4.85 | 0.095 |
| ETS | Statistical | 39.64 | 21.35 | 4.74 | 0.075 |
| ARIMA | Statistical | 42.81 | 20.62 | 5.03 | 0.001 |
| Naive | Baseline | 42.86 | 28.72 | 5.34 | 0 (ref) |
| N-HiTS | Deep learning | 43.18 | 25.26 | 4.88 | -0.007 |
| N-BEATS | Deep learning | 43.44 | 24.40 | 5.00 | -0.014 |
| TFT | Deep learning | 50.37 | 31.31 | 6.43 | -0.175 |
| XGBoost | ML | 53.35 | 25.34 | 5.63 | -0.245 |
| Prophet | Statistical | 60.65 | 30.63 | 6.87 | -0.415 |

### Ensemble optimisation

Of 502 ensemble combinations tested (all 2-to-9-model subsets), the best-performing was ETS + Prophet + XGBoost + TFT + PatchTST (MAE 33.01; skill score 0.230; Table 2). This 5-model ensemble reduced error by 23% over the naive baseline and 10% over PatchTST alone. The top 20 ensembles all included PatchTST. Performance peaked at 5 models, with 6-to-9-model ensembles showing diminishing returns (best 9-model: MAE 33.60). The simplest competitive ensemble was PatchTST + Chronos (2 models; MAE 34.64; skill 0.192) — combining a trained deep learning model with a zero-shot foundation model.

**Table 2. Best ensemble by size (of 502 combinations tested)**

| Combination | Size | Mean MAE | Skill |
|-------------|------|---------|-------|
| PatchTST + Chronos | 2 | 34.64 | 0.192 |
| ETS + TFT + PatchTST | 3 | 33.40 | 0.221 |
| ETS + Prophet + TFT + PatchTST | 4 | 33.15 | 0.227 |
| ETS + Prophet + XGBoost + TFT + PatchTST | 5 | 33.01 | 0.230 |
| ETS + Prophet + XGBoost + TFT + N-HiTS + PatchTST | 6 | 33.17 | 0.226 |
| All 9 models | 9 | 33.60 | 0.216 |

### Prediction interval calibration

Chronos produced the best-calibrated prediction intervals: 62.9% empirical coverage for nominal 80% intervals (Figure 2). TFT (19.6%) and Prophet (23.1%) were severely overconfident. TFT achieved better CRPS than Prophet (45.9 vs 51.7 DALYs/100k).

### Model robustness and diagnostics

Forecast error increased with horizon for all models (Figure 3). ARIMA degraded sharply when trained through 2020 (MAE 169.85 vs 42.83 at 2018 cutoff) due to COVID-19 disruptions; XGBoost was more robust (MAE 62.90). Expanding-window cross-validation demonstrated stable ARIMA performance across pre-COVID cutoffs (MAE range 16.4-30.6). All models except Prophet exhibited significant positive bias. Ljung-Box tests indicated residual autocorrelation in 55.6% of ARIMA models.

### TFT interpretability

Despite its moderate accuracy, TFT provided unique interpretability through variable selection networks (Figure 4). Health expenditure per capita and GDP were the most important time-varying covariates. Location accounted for 27.2% of static variable importance and cause 24.1%. Granger causality tests partially validated these rankings: urbanisation and population predicted DALY trajectories in 58.9% of series; GDP in 35.6%.

### COVID-19 disruption

Of 90 series, 45 (50%) showed permanent shifts from pre-COVID-19 trajectories (Figure 5). Depression and anxiety were most disrupted (GBR depression: +23.6% deviation in 2020, persisting at +22.7% in 2023). COPD showed consistent negative disruption across all countries.

### Australia: projected burden to 2040

Table 3 presents projected DALY rate changes for Australia to 2040 using ARIMA (which permits straightforward per-series recursive forecasting to arbitrary horizons), alongside PatchTST 5-year projections to 2028 for comparison. Both models agreed on direction for most causes. Anxiety disorders, dementia, and liver cancer showed the largest projected increases; IHD, breast cancer, and stroke the largest decreases. Australia diverged unfavourably from peers in lung cancer (+18.1 percentage points), COPD (+14.5pp), and dementia (+8.3pp) (Figure 6).

**Table 3. Australia: projected DALY rate changes (selected causes)**

| Cause | 2023 rate | ARIMA 2040 | Change | PatchTST 2028 | Change |
|-------|-----------|-----------|--------|--------------|--------|
| Anxiety | 1,314 | 2,779 | +111.5% | 1,656 | +26.0% |
| Dementia | 972 | 1,215 | +25.0% | 1,072 | +10.4% |
| Liver cancer | 198 | 225 | +13.6% | 218 | +10.3% |
| IHD | 1,376 | 799 | -41.9% | 1,368 | -0.6% |
| Breast cancer | 341 | 226 | -33.7% | 321 | -5.9% |
| Stroke | 720 | 498 | -30.9% | 694 | -3.6% |

*Rates per 100,000. ARIMA projections to 2040 (17-year horizon); PatchTST projections to 2028 (5-year horizon, its trained prediction length). Full 18-cause ARIMA table in Supplementary Table S1.*

### Sex- and age-stratified projections

Female liver cancer DALYs were projected to increase 43.2% versus a 6.4% male decrease (Figure 7). Female anxiety increased 357.6% versus 115.7% in males. Age-stratified analysis showed anxiety increases concentrated in younger groups (steepest in 0-14 cohort, from a baseline of 3.6 DALYs/100,000). Stroke declined uniformly across all ages (-39% to -42%).

### Robustness

ARIMA fitted to death rates achieved comparable MAPE (5.17% vs 5.03%) with directionally consistent projections. Australia was the first of the five countries where cancer overtook cardiovascular DALYs (1993) [34]; Canada is projected to cross in 2024. At the 2040 horizon, 14/90 series had high model agreement (CV <10%) while 17 had very low agreement (CV >50%).

---

## Discussion

This study provides the first systematic comparison of nine forecasting models across four paradigms — statistical, ML, deep learning, and foundation model — for GBD disease burden prediction, evaluated on 90 time series from five countries.

Three principal findings emerged. First, PatchTST — a patched transformer architecture — outperformed all other individual models (MAE 36.70, skill 0.144), demonstrating that modern deep learning can improve over classical methods on GBD data when the architecture is well-matched to the data characteristics. PatchTST's channel-independent design and patch-based tokenisation appear better suited to 34-observation annual series than TFT's more complex encoder-decoder architecture. A 5-model ensemble further reduced error to MAE 33.01, representing a 23% improvement over naive persistence — consistent with the forecast combination literature showing that simple weighted averages frequently outperform individual models [40,41,42]. All top-performing ensembles included PatchTST, confirming its value as a component model.

Second, the zero-shot performance of Chronos (MAE 38.81, no training) demonstrates that pretrained time-series foundation models transfer effectively to epidemiological domains. Chronos outperformed ARIMA, N-BEATS, N-HiTS, TFT, XGBoost, and Prophet despite never seeing GBD data during pretraining. Its prediction intervals were also far better calibrated (62.9% empirical coverage for nominal 80%) than TFT (19.6%) or Prophet (23.1%). This has practical implications: disease burden forecasts with reasonable uncertainty quantification can be obtained from a foundation model without any model development or training infrastructure.

Third, TFT ranked 8th of 9 models for accuracy but provided unique interpretability through variable selection and attention mechanisms, identifying health expenditure and GDP as key covariates. This positions TFT as a complement to — rather than a replacement for — more accurate models: it answers *why* burden is changing, while PatchTST and Chronos answer *how much*.

These findings extend recent disease-specific applications of ML to GBD data, including XGBoost for diabetes [9] and transformers for head and neck cancer [10]. Our multi-cause, multi-country, multi-architecture comparison provides the most comprehensive evaluation of forecasting methods for GBD data to date.

The poor calibration of TFT and Prophet prediction intervals echoes documented overconfidence in ML uncertainty quantification [35]. Chronos's superior calibration likely reflects its pretraining on a diverse corpus of time series, which provides better-calibrated uncertainty estimates than models trained on a single narrow domain. For policy use, we recommend Chronos prediction intervals or model agreement (CV across models) as pragmatic confidence measures, rather than the poorly calibrated intervals from TFT or Prophet.

For Australia, the projected doubling of anxiety DALYs and 25% increase in dementia, alongside cardiovascular decline, reflects the ongoing epidemiological transition [34,36,37]. The sex-stratified findings — particularly the 43% increase in female liver cancer versus a male decline — suggest emerging sex-specific risk factors warranting surveillance [38]. The steep anxiety trajectory in children warrants cautious interpretation given the low baseline but aligns with evidence of worsening paediatric mental health [39].

### Limitations

Series length (34 annual observations) limits deep learning advantage. Future covariates were held at 2023 levels. GBD uncertainty was not propagated. ARIMA residual diagnostics indicated misspecification in 55.6% of series. The analysis was restricted to five high-income countries. Chronos was evaluated zero-shot; fine-tuning on GBD data might further improve its performance. PatchTST and neuralforecast models were trained with default hyperparameters; systematic tuning could change the ranking. As a single-author study, code availability is provided to support reproducibility.

---

## References

1. GBD 2023 Collaborators. Global Burden of Disease Study 2023 Results. Seattle: IHME, 2024. Available from: https://vizhub.healthdata.org/gbd-results/

2. GBD 2021 Diseases and Injuries Collaborators. Global incidence, prevalence, years lived with disability, disability-adjusted life-years, and healthy life expectancy for 371 diseases and injuries, 1990-2021. Lancet. 2024;403:2133-2161.

3. Murray CJL, Lopez AD. The Global Burden of Disease. Cambridge: Harvard University Press; 1996.

4. Foreman KJ, Marquez N, Dolgert A, et al. Forecasting life expectancy, years of life lost, and all-cause and cause-specific mortality for 250 causes of death. Lancet. 2018;392:2052-2090.

5. IHME. GBD Foresight. Seattle: IHME; 2024. Available from: https://vizhub.healthdata.org/gbd-foresight/

6. Saeed S, Luo Z, Wang H, et al. Mapping the global burden and inequalities of bipolar disorder, 1990-2021, with projections to 2050. Bipolar Disord. 2026;28(1). doi:10.1111/bdi.70074.

7. Zhou J, Chen H, Zhou S, et al. Trends in atopic dermatitis prevalence among the Chinese population with projections for 2022-2030. Pediatr Allergy Immunol. 2024;35(10). doi:10.1111/pai.14271.

8. Kim E, Park Y, Park S, et al. Global burden of disease due to high body mass index and projections to 2040. Int J Health Plann Manage. 2025;40:1069-1082.

9. Zhong X, Zheng Y, Wang L, et al. Evaluating global epidemiology of type 2 diabetes mellitus: a 60-year study by interpretable machine learning framework. Diabetes Obes Metab. 2025;27:7476-7489.

10. Hu Q, Lv S, Wang X, et al. Global burden and future trends of head and neck cancer: a deep learning-based analysis (1980-2030). PLoS One. 2025;20(4):e0320184.

11. Asgharpour M, Khlilizad Darounkolaei M, Mazloumi S, et al. Scoping review of artificial intelligence applications in Global Burden of Disease studies. InfoScience Trends. 2025;2(3):13-24.

12. Lim B, Arik SO, Loeff N, Pfister T. Temporal fusion transformers for interpretable multi-horizon time series forecasting. Int J Forecast. 2021;37:1748-1764.

13. Oreshkin BN, Carpov D, Chapados N, Bengio Y. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting. In: International Conference on Learning Representations; 2020.

14. Challu C, Olivares KG, Oreshkin BN, et al. N-HiTS: Neural hierarchical interpolation for time series forecasting. In: Proceedings of the AAAI Conference on Artificial Intelligence; 2023;37:6989-6997.

15. Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. A time series is worth 64 words: long-term forecasting with transformers. In: International Conference on Learning Representations; 2023.

16. Ansari AF, Stella L, Turkmen C, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815; 2024.

17. Santomauro DF, Mantilla Herrera AM, Shadid J, et al. Global prevalence and burden of depressive and anxiety disorders in 204 countries and territories in 2020 due to the COVID-19 pandemic. Lancet. 2021;398:1700-1712.

18. Leung JM, Niikura M, Yang CWT, Sin DD. COVID-19 and COPD. Eur Respir J. 2020;56(2):2002108.

19. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated reporting guideline for clinical prediction models. BMJ. 2024;385:e078378.

20. World Bank. World Development Indicators. Washington, DC; 2024. Available from: https://data.worldbank.org/

21. Dickey DA, Fuller WA. Distribution of the estimators for autoregressive time series with a unit root. J Am Stat Assoc. 1979;74:427-431.

22. Seabold S, Perktold J. Statsmodels: econometric and statistical modeling with Python. Proc 9th Python in Science Conf. 2010:92-96.

23. Hyndman RJ, Athanasopoulos G. Forecasting: Principles and Practice. 3rd ed. Melbourne: OTexts; 2021.

24. Taylor SJ, Letham B. Forecasting at scale. Am Stat. 2018;72:37-45.

25. Chen T, Guestrin C. XGBoost: a scalable tree boosting system. Proc 22nd ACM SIGKDD. 2016:785-794.

26. Beitner J. pytorch-forecasting: Time series forecasting with PyTorch. 2020. Available from: https://pytorch-forecasting.readthedocs.io/

27. Olivares KG, Challu C, Marcjasz G, et al. NeuralForecast: scalable and easy-to-use neural forecasting library. In preparation. Available from: https://nixtla.github.io/neuralforecast/

28. Diebold FX, Mariano RS. Comparing predictive accuracy. J Bus Econ Stat. 1995;13:253-263.

29. Ljung GM, Box GEP. On a measure of lack of fit in time series models. Biometrika. 1978;65:297-303.

30. Granger CWJ. Investigating causal relations by econometric models and cross-spectral methods. Econometrica. 1969;37:424-438.

31. Benjamini Y, Hochberg Y. Controlling the false discovery rate. J R Stat Soc B. 1995;57:289-300.

32. Paszke A, Gross S, Massa F, et al. PyTorch: an imperative style, high-performance deep learning library. NeurIPS. 2019:8024-8035.

33. Hunter JD. Matplotlib: a 2D graphics environment. Comput Sci Eng. 2007;9:90-95.

34. Omran AR. The epidemiologic transition. Milbank Mem Fund Q. 1971;49:509-538.

35. Makridakis S, Spiliotis E, Assimakopoulos V. The M5 uncertainty competition: results, findings and conclusions. Int J Forecast. 2022;38:1365-1385.

36. Australian Institute of Health and Welfare. Australian Burden of Disease Study 2023. Canberra: AIHW; 2024.

37. Frenk J, Bobadilla JL, Stern C, et al. Elements for a theory of the health transition. Health Transit Rev. 1991;1:21-38.

38. Tapper EB, Parikh ND. Mortality due to cirrhosis and liver cancer in the United States, 1999-2016. BMJ. 2018;362:k2817.

39. Racine N, McArthur BA, Cooke JE, et al. Global prevalence of depressive and anxiety symptoms in children and adolescents during COVID-19: a meta-analysis. JAMA Pediatr. 2021;175:1142-1150.

40. Makridakis S, Spiliotis E, Assimakopoulos V. The M4 Competition: 100,000 time series and 61 forecasting methods. Int J Forecast. 2020;36:54-74.

41. Timmermann A. Forecast combinations. In: Handbook of Economic Forecasting. Vol 1. Elsevier; 2006:135-196.

42. Bates JM, Granger CWJ. The combination of forecasts. J Oper Res Soc. 1969;20:451-468.

---

## Data availability statement

GBD 2023 data are publicly available from https://vizhub.healthdata.org/gbd-results/. World Bank data from https://data.worldbank.org/. Analysis code and processed datasets will be available at [repository URL].

## Funding

This research received no specific funding.

## Conflicts of interest

None declared.

## Author contributions

HF conceived the study, acquired and analysed the data, developed all forecasting models, and wrote the manuscript.

## AI use disclosure

AI tools (Claude, Anthropic) assisted with Python code development, statistical analysis, figure generation, and manuscript drafting. All analyses, results, and interpretations were independently verified by the author, who takes full responsibility for the content.
