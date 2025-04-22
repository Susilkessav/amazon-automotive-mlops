import pandas as pd
from sklearn.metrics import mean_absolute_error

def main():
    df_pred = pd.read_csv("data/processed/forecasts.csv", parse_dates=["date"])
    df_act = pd.read_csv("data/processed/actuals.csv", parse_dates=["date"])
    merged = df_act.merge(df_pred, on="date", how="inner")
    for model in ("arima", "prophet"):
        mae = mean_absolute_error(merged["y"], merged[model])
        print(f"{model}: MAE = {mae:.2f}")
    
if __name__ == "__main__":
    main()
