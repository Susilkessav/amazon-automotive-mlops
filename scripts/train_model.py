import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import pathlib

def main():
    df = pd.read_csv("data/processed/processed.csv", parse_dates=["date"])
    # 1) Aggregate to daily counts
    daily = df.groupby("date").size().reset_index(name="y").sort_values("date")
    # 2) Fill missing dates with 0 count
    full_index = pd.date_range(start=daily["date"].min(), end=daily["date"].max(), freq='D')
    daily_full = daily.set_index("date").reindex(full_index, fill_value=0)
    daily_full = daily_full.rename_axis("date").reset_index()  # now has 'date' and 'y'
    # 3) Anomaly detection: find outlier days
    mean_y = daily_full["y"].mean()
    std_y = daily_full["y"].std()
    high_threshold = mean_y + 3*std_y
    outliers = daily_full[daily_full["y"] > high_threshold]
    if not outliers.empty:
        print(f"Anomaly detected: {len(outliers)} days with review count > {high_threshold:.1f}:")
        for _, row in outliers.iterrows():
            print(f"   {row['date'].date()} had {row['y']} reviews.")
    # 4) Train/test split by date
    TEST_DAYS = 30
    if len(daily_full) <= TEST_DAYS:
        TEST_DAYS = max(1, int(0.2 * len(daily_full)))
    train_full = daily_full.iloc[:-TEST_DAYS]
    test_full = daily_full.iloc[-TEST_DAYS:]
    print(f"Using last {len(test_full)} days as test (from {test_full.iloc[0]['date'].date()} to {test_full.iloc[-1]['date'].date()}).")
    # 5) ARIMA model on training data
    arima = ARIMA(train_full["y"], order=(1,1,1)).fit()
    arima_fc = arima.get_forecast(steps=len(test_full))
    arima_pred = arima_fc.predicted_mean
    # 6) Prophet model on training data
    prophet_df = train_full.rename(columns={"date":"ds", "y":"y"})
    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=len(test_full), freq='D', include_history=False)
    forecast_df = m.predict(future)
    prophet_pred = forecast_df["yhat"]
    # 7) Save actuals and forecasts
    out_dir = pathlib.Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_full.to_csv(out_dir/"actuals.csv", index=False)
    pd.DataFrame({
        "date": future["ds"], 
        "arima": arima_pred.values, 
        "prophet": prophet_pred.values
    }).to_csv(out_dir/"forecasts.csv", index=False)
    # 8) Save models
    import joblib
    joblib.dump(arima, "models/arima.pkl")
    joblib.dump(m, "models/prophet.pkl")
    print("Training complete. Models saved and forecasts generated.")

if __name__ == "__main__":
    main()

