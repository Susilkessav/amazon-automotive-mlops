import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd

@st.cache_data
def load_data():
    # adjust these paths & column names as needed
    actuals_path   = "data/processed/actuals.csv"
    forecasts_path = "data/processed/forecasts.csv"

    # use the right date column name here: 'ds' in this example
    actuals   = pd.read_csv(actuals_path,   parse_dates=["ds"])
    forecasts = pd.read_csv(forecasts_path, parse_dates=["ds"])

    # rename for consistency
    actuals.rename(columns={"ds": "date", "value": "actual"}, inplace=True)
    forecasts.rename(columns={"ds": "date"}, inplace=True)

    # now forecasts has columns ['date','arima','prophet'], etc.
    return actuals, forecasts

def main():
    st.title("📈 Forecasting: Model Comparison Dashboard")
    actuals, forecasts = load_data()

    # merge on date
    df = actuals.merge(forecasts, on="date", how="left")

    st.markdown(
        """
        This dashboard shows the **true** series and overlaid **predictions** 
        from different models. Use the controls below to:
        1. Pick which models to compare  
        2. Zoom in on any date range  
        3. Inspect point‑wise errors  
        """
    )

    # — sidebar: model selector & date filter —
    st.sidebar.header("Controls")
    models = list(preds.keys())
    selected = st.sidebar.multiselect("Select models to display:", models, default=models[:2])
    dmin, dmax = st.sidebar.date_input(
        "Date range",
        value=[actual.date.min(), actual.date.max()],
        min_value=actual.date.min(),
        max_value=actual.date.max(),
    )

    # filter data
    mask = (actual.date >= pd.to_datetime(dmin)) & (actual.date <= pd.to_datetime(dmax))
    actual_filt = actual.loc[mask]
    df_plot = actual_filt.copy()
    for m in selected:
        df_plot = df_plot.merge(preds[m], on="date", how="left")

    # — line chart of actual vs preds —
    st.subheader("Actual vs. Predictions")
    fig = px.line(
        df_plot.melt(id_vars="date", value_vars=["value"] + selected,
                     var_name="series", value_name="y"),
        x="date", y="y", color="series",
        labels={"date":"Date", "y":"Value", "series":"Series"},
        title="True values (solid) vs. model forecasts (dashed)"
    )
    # make actual a solid line, preds dashed
    fig.for_each_trace(lambda t: t.update(line=dict(dash="dash")) if t.name in selected else None)
    fig.for_each_trace(lambda t: t.update(line=dict(dash="solid", width=2)) if t.name=="value" else None)
    st.plotly_chart(fig, use_container_width=True)

    # — error metrics table —
    st.subheader("Error Metrics")
    metrics = []
    for m in selected:
        df_m = df_plot.dropna(subset=[m])
        mse = ((df_m[m] - df_m["value"])**2).mean()
        mae = (df_m[m] - df_m["value"]).abs().mean()
        metrics.append({"model": m, "MAE": f"{mae:.2f}", "MSE": f"{mse:.2f}"})
    st.table(pd.DataFrame(metrics).set_index("model"))

    # — point‑wise error scatter —
    st.subheader("Prediction Errors Over Time")
    err_df = df_plot[["date", "value"] + selected].copy()
    for m in selected:
        err_df[f"err_{m}"] = err_df[m] - err_df["value"]
    melted = err_df.melt(id_vars="date", value_vars=[f"err_{m}" for m in selected],
                         var_name="model", value_name="error")
    # clean up model names
    melted["model"] = melted["model"].str.replace("err_", "")
    fig2 = px.scatter(
        melted, x="date", y="error", color="model",
        labels={"error":"Prediction Error", "date":"Date"},
        title="Point‑wise Forecast Error"
    )
    st.plotly_chart(fig2, use_container_width=True)

        # …continuing in app/streamlit_app.py…

    actual, preds = load_data()

    st.title("🚗 Amazon Automotive: Forecasts Comparison")
    st.markdown(
        """
        Use the controls in the sidebar to pick which model forecasts you want
        to compare against the true historical values.
        """
    )

    # sidebar: choose date range and models
    with st.sidebar:
        st.header("Controls")
        # date slider
        min_date = actual["date"].min()
        max_date = actual["date"].max()
        date_range = st.date_input(
            "Date range", [min_date, max_date], min_value=min_date, max_value=max_date
        )
        # model multiselect
        model_names = list(preds.keys())
        chosen = st.multiselect("Which model(s)?", model_names, default=model_names)

    # filter actuals
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask = (actual["date"] >= start) & (actual["date"] <= end)
    actual_filt = actual.loc[mask]

    # build combined df for plotting
    plot_df = actual_filt.set_index("date").rename(columns={"value": "Actual"})
    for m in chosen:
        dfm = preds[m].set_index("date")
        plot_df = plot_df.join(dfm[m], how="left")

    st.subheader("Time‑Series Plot")
    fig = px.line(
        plot_df.reset_index(),
        x="date",
        y=["Actual"] + chosen,
        labels={"value": "Value", "date": "Date", "variable": "Series"},
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

    # error metrics
    st.subheader("Error Metrics")
    metrics = []
    for m in chosen:
        # align on dropna
        tmp = plot_df[["Actual", m]].dropna()
        mae = (tmp["Actual"] - tmp[m]).abs().mean()
        rmse = ((tmp["Actual"] - tmp[m]) ** 2).mean() ** 0.5
        metrics.append({"Model": m, "MAE": mae, "RMSE": rmse})
    metrics_df = pd.DataFrame(metrics).set_index("Model")
    st.table(metrics_df.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}"}))

if __name__ == "__main__":
    main()

