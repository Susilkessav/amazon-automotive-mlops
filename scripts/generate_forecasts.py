#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

def main(actuals_path: Path, output_dir: Path, steps: int):
    # 1) load your existing actuals
    df = pd.read_csv(actuals_path, parse_dates=["date"])

    # allow either 'value' or 'y' for backward compatibility
    if "y" in df.columns:
        df = df.rename(columns={"y": "value"})
    elif "value" not in df.columns:
        raise ValueError(f"{actuals_path} must contain 'date' and 'value' (or 'y') columns")

    # ensure output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2) write out a clean copy of actuals
    df.to_csv(output_dir/"actuals.csv", index=False)

    # prepare for modeling
    series = df.rename(columns={"date":"ds","value":"y"})[["ds","y"]]

    # 3) ARIMA forecast
    arima_mod = ARIMA(series["y"], order=(1,1,1)).fit()
    arima_fc = arima_mod.get_forecast(steps=steps)
    arima_index = pd.date_range(
        start=series["ds"].iloc[-1] + pd.Timedelta(days=1),
        periods=steps, freq="D"
    )
    df_arima = pd.DataFrame({
        "date": arima_index,
        "forecast": arima_fc.predicted_mean.values
    })
    df_arima.to_csv(output_dir/"forecast_arima.csv", index=False)

    # 4) Prophet forecast
    m = Prophet()
    m.fit(series)
    future = m.make_future_dataframe(periods=steps, freq="D", include_history=False)
    fc = m.predict(future)
    df_prophet = (
        fc[["ds","yhat"]]
        .rename(columns={"ds":"date","yhat":"forecast"})
    )
    df_prophet.to_csv(output_dir/"forecast_prophet.csv", index=False)

    print("✅ Forecasts written to", output_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate ARIMA & Prophet forecasts")
    p.add_argument(
        "--actuals",
        type=Path,
        default=Path("data/processed/actuals.csv"),
        help="Path to your actuals.csv (must have date,value or y)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where to write actuals.csv, forecast_arima.csv, forecast_prophet.csv"
    )
    p.add_argument(
        "--steps", type=int, default=30,
        help="How many days to forecast into the future"
    )
    args = p.parse_args()
    main(args.actuals, args.output_dir, args.steps)
