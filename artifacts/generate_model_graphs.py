"""Generate report-ready comparison plots for the common chronological holdout."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ISDBcleaned_DB.xlsx"
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET = "primary_energy_consumption"
TEST_START_YEAR = 2019


def add_lags(data: pd.DataFrame) -> pd.DataFrame:
    result = data.sort_values(["country", "year"], kind="mergesort").reset_index(drop=True)
    grouped = result.groupby("country", sort=False)
    for lag in (1, 2):
        previous = grouped[TARGET].shift(lag)
        previous_year = grouped["year"].shift(lag)
        result[f"lag{lag}"] = previous.where(result["year"].sub(previous_year).eq(lag))
    result["delta_last_year"] = result["lag1"] - result["lag2"]
    return result


def preprocessing(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def fit_predictions() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    data = pd.read_excel(DATA_PATH)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = add_lags(data).dropna(subset=[TARGET]).copy()
    train = data.loc[data["year"] < TEST_START_YEAR].copy()
    test = data.loc[data["year"] >= TEST_START_YEAR].copy()

    categorical = ["iso3", "country"]
    numeric = [
        "year",
        "renewables_share_elec",
        "greenhouse_gas_emissions",
        "electricity_access_pct",
        "gdp_per_capita",
        "urban_pop_growth",
        "lag1",
        "lag2",
        "delta_last_year",
    ]
    features = categorical + numeric
    x_train = train[features]
    x_test = test[features]
    y_train = train[TARGET].to_numpy(dtype=float)

    predictions: dict[str, np.ndarray] = {
        "Carry-forward baseline": test["lag1"].to_numpy(dtype=float)
    }

    lasso = Pipeline(
        [
            ("preprocessing", preprocessing(categorical, numeric)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "lasso",
                LassoCV(
                    cv=5,
                    random_state=42,
                    max_iter=50000,
                    alphas=100,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lasso.fit(x_train, y_train)
    predictions["LassoCV"] = lasso.predict(x_test)

    random_forest = Pipeline(
        [
            ("preprocessing", preprocessing(categorical, numeric)),
            (
                "random_forest",
                RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            ),
        ]
    )
    random_forest.fit(x_train, y_train)
    predictions["Random Forest"] = random_forest.predict(x_test)

    ordinal = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        dtype=np.int32,
    )
    encoded_train = x_train.copy()
    encoded_test = x_test.copy()
    encoded_train[categorical] = ordinal.fit_transform(encoded_train[categorical])
    encoded_test[categorical] = ordinal.transform(encoded_test[categorical])

    absolute = LGBMRegressor(
        objective="regression", random_state=42, n_jobs=-1, verbosity=-1
    )
    absolute.fit(encoded_train, y_train, categorical_feature=categorical)
    predictions["LightGBM absolute"] = absolute.predict(encoded_test)

    residual_mask = train["lag1"].notna() & train["lag2"].notna()
    residual = LGBMRegressor(
        objective="regression_l1",
        random_state=42,
        n_jobs=-1,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=25,
        verbosity=-1,
    )
    residual.fit(
        encoded_train.loc[residual_mask],
        y_train[residual_mask.to_numpy()] - train.loc[residual_mask, "lag1"].to_numpy(),
        categorical_feature=categorical,
    )
    app_prediction = predictions["LightGBM absolute"].copy()
    test_history = test["lag1"].notna() & test["lag2"].notna()
    app_prediction[test_history.to_numpy()] = (
        test.loc[test_history, "lag1"].to_numpy()
        + residual.predict(encoded_test.loc[test_history])
    )
    predictions["LightGBM app path"] = app_prediction
    return test, predictions


def save_bar_chart(
    values: dict[str, float],
    title: str,
    ylabel: str,
    filename: str,
    lower_is_better: bool = True,
) -> None:
    names = list(values)
    colors = ["#6b7280", "#d97706", "#2f8f83", "#246e55", "#9b6bd3"]
    order = np.argsort(list(values.values()))
    if not lower_is_better:
        order = order[::-1]
    names = [names[i] for i in order]
    plotted = [values[name] for name in names]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(names, plotted, color=[colors[list(values).index(name)] for name in names])
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, plotted):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / filename.replace(".svg", ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    test, predictions = fit_predictions()
    actual = test[TARGET].to_numpy(dtype=float)
    metrics = {
        name: {
            "MAE": mean_absolute_error(actual, prediction),
            "RMSE": np.sqrt(np.mean((actual - prediction) ** 2)),
            "WAPE": np.abs(actual - prediction).sum() / np.abs(actual).sum() * 100,
        }
        for name, prediction in predictions.items()
    }
    save_bar_chart(
        {name: values["MAE"] for name, values in metrics.items()},
        "Mean Absolute Error by Model",
        "MAE (TWh, lower is better)",
        "model_mae.svg",
    )
    save_bar_chart(
        {name: values["RMSE"] for name, values in metrics.items()},
        "Root Mean Squared Error by Model",
        "RMSE (TWh, lower is better)",
        "model_rmse.svg",
    )
    save_bar_chart(
        {name: values["WAPE"] for name, values in metrics.items()},
        "Weighted Absolute Percentage Error by Model",
        "WAPE (%) (lower is better)",
        "model_wape.svg",
    )

    baseline_mae = metrics["Carry-forward baseline"]["MAE"]
    relative_mae = {
        name: (baseline_mae - values["MAE"]) / baseline_mae * 100
        for name, values in metrics.items()
    }
    save_bar_chart(
        relative_mae,
        "MAE Improvement Relative to Carry-forward Baseline",
        "Improvement (%) (higher is better)",
        "model_relative_to_baseline.svg",
        lower_is_better=False,
    )

    yearly = pd.DataFrame({"year": test["year"].to_numpy(dtype=int), "actual": actual})
    for name, prediction in predictions.items():
        yearly[name] = np.abs(actual - prediction)
    yearly_mean = yearly.groupby("year").mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name in predictions:
        ax.plot(
            yearly_mean.index,
            yearly_mean[name],
            marker="o",
            linewidth=2,
            label=name,
        )
    ax.set_title("Mean Absolute Error by Test Year", fontsize=15, weight="bold")
    ax.set_xlabel("Test year")
    ax.set_ylabel("MAE (TWh, lower is better)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_mae_by_year.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "model_mae_by_year.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sample = np.argsort(actual)
    for name, prediction in predictions.items():
        ax.scatter(
            actual[sample],
            prediction[sample],
            s=9,
            alpha=0.28,
            label=name,
        )
    limits = [0, max(actual.max(), max(prediction.max() for prediction in predictions.values()))]
    ax.plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
    ax.set_title("Actual versus Predicted Consumption", fontsize=15, weight="bold")
    ax.set_xlabel("Actual consumption (TWh)")
    ax.set_ylabel("Predicted consumption (TWh)")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
