# Model comparison: primary energy consumption

## Shared chronological holdout

The models were trained and evaluated on the same features and rows from `ISDBcleaned_DB.xlsx`. Data covers 220 countries and 2000-2023; 5,205 rows have an observed target. Models were trained on 2000-2018 (4,115 rows) and tested on 2019-2023 (1,090 country-years, 218 countries per year). The one- and two-year lag features use only the corresponding country's earlier calendar years.

LassoCV and Random Forest were refitted on the chronological training period rather than scoring the existing random-split artifacts, which may have seen later years during training. LightGBM absolute predicts consumption directly. The app-style LightGBM delta model predicts the change from the previous year when both lags exist and falls back to the absolute model otherwise. The carry-forward baseline predicts the previous year's consumption.

| Model | MAE (TWh) ↓ | RMSE (TWh) ↓ | WAPE (%) ↓ | MASE ↓ | R² ↑ |
|---|---:|---:|---:|---:|---:|
| LassoCV | 85.296 | 150.613 | 16.315 | 6.359 | 0.988583 |
| Random Forest | 21.038 | 61.675 | 4.024 | 1.568 | 0.998086 |
| LightGBM (absolute target) | 23.055 | 68.784 | 4.410 | 1.719 | 0.997619 |
| LightGBM (delta + fallback; app path) | **18.348** | **52.581** | **3.509** | **1.368** | **0.998608** |
| Carry-forward baseline | 18.655 | 52.709 | 3.568 | 1.391 | 0.998602 |

On this global holdout, the app-style LightGBM is slightly better than the carry-forward baseline on each reported metric. The gains are small: MAE improves by 0.307 TWh (about 1.6%), and WAPE by 0.059 percentage points. R² is high for both because the dataset contains countries with very different consumption levels; it is not a percent-accuracy score.

The aggregate WAPE is heavily weighted toward large-consuming countries. For the 212 countries with complete data for all five test years, median country WAPE was 6.12% for LightGBM and 5.62% for carry-forward; LightGBM had lower country WAPE in 93 of 212 countries. So the app model's small global advantage does not mean it is better for a typical individual country.

## Tuning check

LightGBM alternatives were selected on rolling validation years 2016-2018, then evaluated on the same untouched 2019-2023 holdout. The selected L2-residual alternative narrowly improved MAE (18.618 TWh) and WAPE (3.561%) versus the baseline, but had worse RMSE (53.295 versus 52.709 TWh). It did not consistently beat the current app-style LightGBM, so its hyperparameters were not adopted.

## Metric definitions

- **MAE**: average absolute forecast error, in TWh.
- **RMSE**: square root of the average squared error, in TWh; larger misses receive more weight.
- **WAPE**: total absolute error divided by total absolute observed consumption, as a percentage. It is an aggregate measure and can be dominated by large-consuming countries.
- **MASE**: MAE divided by the training-period mean absolute one-year change. Below 1 is better than that training-scale naïve error; above 1 is worse. Here, all MASE values exceed 1, even though LightGBM modestly outperforms the carry-forward forecast on this particular holdout.
- **R²**: proportion of target variance explained relative to predicting the test-set mean. It is context, not a percent-accuracy score.

`MLcreation.py` reports MAE, RMSE, WAPE, MASE, and R² on the same latest-years split, and also reports carry-forward metrics for rows with previous-year values. It now trains and evaluates the delta path only when both lags are present, matching the API's model-selection behavior. The saved app artifact was regenerated with that alignment.

## Graphical comparisons

The following plots use the same chronological holdout and the same model predictions as the table above:

- [MAE comparison](model_mae.svg) — average absolute error in TWh.
- [RMSE comparison](model_rmse.svg) — emphasizes large errors.
- [WAPE comparison](model_wape.svg) — aggregate percentage error.
- [Relative MAE comparison](model_relative_to_baseline.svg) — improvement or deterioration relative to carry-forward.
- [Year-by-year MAE](model_mae_by_year.svg) — shows how performance changes across 2019–2023.
- [Actual versus predicted](actual_vs_predicted.svg) — visualizes calibration and spread across the target range.

PNG versions are also available for convenient insertion into a Word report:
[MAE](model_mae.png), [RMSE](model_rmse.png), [WAPE](model_wape.png),
[relative baseline](model_relative_to_baseline.png), [year-by-year MAE](model_mae_by_year.png),
and [actual versus predicted](actual_vs_predicted.png).

The plots can be regenerated with [generate_model_graphs.py](generate_model_graphs.py).
