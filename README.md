# Global Electricity Consumption Predictor

An interactive web app for exploring country-level energy data and forecasting primary energy consumption. The project combines a FastAPI backend, a LightGBM forecasting model, and a browser-based dashboard.

## Features

- Select a country and review its latest energy profile.
- Adjust renewable energy share, emissions, electricity access, GDP per capita, and other assumptions.
- Generate a consumption forecast in TWh.
- Upload CSV, Excel, or SQLite data to inspect and predict from a custom dataset.
- Compare additional regression models and training scripts in the project files.

## Requirements

- Python 3.10 or newer
- Internet access for the interactive country map and globe textures

Install the Python dependencies from the project root:

```bash
python3 -m pip install -r requirements.txt
```

## Run the web app

From the project root, start the API and static frontend together:

```bash
uvicorn api:app --reload
```

Open <http://127.0.0.1:8000>. Keep the terminal running while using the app; press **Ctrl+C** to stop it.

The API loads the saved model at `artifacts/energy_consumption_model.joblib` and training data at `ISDBcleaned_DB.xlsx`. Both are included in this repository.

## Retrain the LightGBM model

To regenerate the app's model artifact from the included workbook:

```bash
python3 MLcreation.py
```

The script writes the trained model to `artifacts/energy_consumption_model.joblib` by default. You can override the input workbook or output path with `--input` and `--model-output`.

The training run reports MAE, RMSE, WAPE, MASE, and R², including carry-forward baseline metrics where lagged observations are available. See [the shared walk-forward model comparison](artifacts/model_comparison.md) for a like-for-like evaluation of Lasso, Random Forest, LightGBM, and the baseline.

## Model evaluation

The deployed model is a **LightGBM regression model**. It predicts continuous primary energy consumption in TWh rather than a class label, so regression metrics are the appropriate way to assess its numerical forecasts.

The latest evaluation uses a chronological holdout: the model was trained on observations from **2000–2018** and tested on **2019–2023**. The test set contains **1,090 country-year observations from 218 countries**. The model uses a year-over-year residual forecast when both prior-year lags are available and an absolute-consumption fallback otherwise.

### Latest holdout results

| Metric | Result | Interpretation |
| --- | ---: | --- |
| MAE | **18.3476 TWh** | Predictions were off by about 18.35 TWh on average. |
| RMSE | **52.5810 TWh** | Large errors are penalized more heavily; the higher value indicates some substantial misses. |
| WAPE | **3.5094%** | Total absolute error was about 3.51% of total observed consumption. |
| MASE | **1.3678** | Error was 1.37 times the training-period average one-year change scale. |
| R² | **0.9986** | Predictions explain most pooled variation, but this is not 99.86% forecast accuracy. |

### What the metrics mean

- **MAE (Mean Absolute Error):** the average absolute difference between predicted and observed consumption. It is the clearest measure of a typical error and is expressed in TWh.
- **RMSE (Root Mean Squared Error):** an error measure in TWh that gives more weight to unusually large misses. Comparing it with MAE helps reveal whether a small number of forecasts are far from the observed value.
- **WAPE (Weighted Absolute Percentage Error):** total absolute error divided by total absolute observed consumption. It is useful for judging aggregate system-level error, but high-consumption countries contribute more to the result.
- **MASE (Mean Absolute Scaled Error):** the model MAE divided by the training-period mean absolute one-year change. It provides a scale-free context for the error. This implementation uses the training change scale, so it should not be read as a direct replacement for the held-out carry-forward comparison.
- **R² (R-squared):** how much pooled variation in the target is explained compared with predicting the test-set mean. It can be very high when countries have very different consumption levels and should not be described as a percentage accuracy.

### Comparison with the carry-forward baseline

The carry-forward baseline predicts that each country's next-year consumption will equal its latest observed consumption. On the same holdout:

| Model | MAE (TWh) | RMSE (TWh) | WAPE | MASE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM app path | **18.3476** | **52.5810** | **3.5094%** | **1.3678** | **0.9986** |
| Carry-forward baseline | 18.6546 | 52.7090 | 3.5682% | 1.3907 | 0.9986 |

The LightGBM model is slightly better on this pooled holdout: MAE improves by approximately **1.6%** relative to carry-forward. This is a modest improvement, not evidence that every country's individual forecast is highly accurate. Country-level results vary, and the aggregate WAPE is influenced by large-consuming countries.

For the full comparison methodology and graphs, see [artifacts/model_comparison.md](artifacts/model_comparison.md), [artifacts/model_comparison.svg](artifacts/model_comparison.svg), and the individual MAE, RMSE, WAPE, baseline-relative, year-by-year, and actual-versus-predicted plots linked from the comparison report.

## API routes

| Route | Purpose |
| --- | --- |
| `GET /api/health` | Check API status |
| `GET /api/options` | Load country choices and model ranges |
| `GET /api/country/{iso3}` | Load a country's recent data |
| `POST /api/predict` | Generate a forecast |
| `POST /api/upload/inspect` | Inspect an uploaded dataset |
| `POST /api/upload/predict` | Generate predictions from an uploaded dataset |

## Data and large files

The repository includes the Excel workbooks used by the project and the LightGBM model used by the web app. Some additional Random Forest model artifacts exceed GitHub's 100 MB regular-file limit and are stored with Git LFS. Install Git LFS before cloning if you need those files:

```bash
git lfs install
git clone https://github.com/vishaal377-byte/Global-electricity-consumption-predictor.git
```

The web app accepts CSV, `.xlsx`, and SQLite (`.db`, `.sqlite`, `.sqlite3`) uploads. Uploads are limited to 5,000 rows and are processed for prediction without being saved by the API.
