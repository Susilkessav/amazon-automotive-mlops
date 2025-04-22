# app/streamlit_dashboard.py

import streamlit as st
import pandas as pd
import glob
from pathlib import Path
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────────────────────────────────────────────────────────
# 1) Point to your real data folder
# ──────────────────────────────────────────────────────────
PROC_DIR     = Path(r"C:\Users\susil\amazon-automotive-mlops\data\processed")
ACTUALS_PATH = PROC_DIR / "actuals.csv"

# ──────────────────────────────────────────────────────────
# 2) Sanity check
# ──────────────────────────────────────────────────────────
if not ACTUALS_PATH.exists():
    st.error(
        f"❌ Cannot find:\n  {ACTUALS_PATH}\n\n"
        "Please generate it with:\n"
        "  python train_model.py\n"
        "  python generate_forecasts.py"
    )
    st.stop()

# ──────────────────────────────────────────────────────────
# 3) Caching loaders
# ──────────────────────────────────────────────────────────
@st.cache_data
def load_actuals() -> pd.DataFrame:
    df = pd.read_csv(ACTUALS_PATH, parse_dates=["date"])
    # rename y or value → actual
    if "y" in df.columns:
        df = df.rename(columns={"y": "actual"})
    elif "value" in df.columns:
        df = df.rename(columns={"value": "actual"})
    return df

@st.cache_data
def load_forecasts() -> dict[str, pd.DataFrame]:
    forecasts = {}
    pattern = str(PROC_DIR / "forecast_*.csv")
    for fpath in glob.glob(pattern):
        model_name = Path(fpath).stem.replace("forecast_", "")
        df = pd.read_csv(fpath, parse_dates=["date"])
        # if column 'forecast' exists, rename it; else assume second column is the preds
        if "forecast" in df.columns:
            df = df.rename(columns={"forecast": model_name})
        else:
            c0, c1 = df.columns[:2]
            df = df.rename(columns={c0: "date", c1: model_name})
        forecasts[model_name] = df[["date", model_name]]
    return forecasts

# ──────────────────────────────────────────────────────────
# 4) Main dashboard
# ──────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Forecasting Dashboard", layout="wide")
    st.title("📈 Forecasting: Model Comparison Dashboard")

    actuals   = load_actuals()
    forecasts = load_forecasts()

    # Sidebar controls
    st.sidebar.header("Controls")
    start_date, end_date = st.sidebar.date_input(
        "Date range",
        value=[actuals["date"].min(), actuals["date"].max()],
        min_value=actuals["date"].min(),
        max_value=actuals["date"].max()
    )
    if start_date > end_date:
        st.sidebar.error("Start date must be before end date")
        st.stop()

    model_names = list(forecasts.keys())
    default = model_names[:2] if len(model_names) > 1 else model_names
    selected = st.sidebar.multiselect("Select model(s) to compare", model_names, default=default)
    if not selected:
        st.sidebar.error("Pick at least one model")
        st.stop()

    # Merge actuals + selected forecasts, then filter by date
    df = actuals.copy()
    for m in selected:
        df = df.merge(forecasts[m], on="date", how="left")
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    df = df.loc[mask]

    # Melt for plotting
    plot_df = df.melt(
        id_vars="date",
        value_vars=["actual"] + selected,
        var_name="series",
        value_name="value"
    )

    # Plot Actual vs. Forecast
    st.subheader("Actual vs. Forecast")
    fig = px.line(
        plot_df,
        x="date",
        y="value",
        color="series",
        labels={"date": "Date", "value": "Value", "series": "Series"},
        title="True series (solid) vs. model forecasts"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Compute & display error metrics
    st.subheader("Error Metrics")
    metrics = []
    for m in selected:
        dfm = df[["actual", m]].dropna()
        if dfm.empty:
            metrics.append({"Model": m, "MAE": "N/A", "RMSE": "N/A"})
        else:
            mae  = mean_absolute_error(dfm["actual"], dfm[m])
            rmse = mean_squared_error(dfm["actual"], dfm[m], squared=False)
            metrics.append({
                "Model": m,
                "MAE":  f"{mae:.2f}",
                "RMSE": f"{rmse:.2f}"
            })
    st.table(pd.DataFrame(metrics).set_index("Model"))

if __name__ == "__main__":
    main()
