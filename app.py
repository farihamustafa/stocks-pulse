import io
import time
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor  # ✅ extra model
import yfinance as yf
from ta.momentum import RSIIndicator
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt  # ✅ for RMSE
import requests
from bs4 import BeautifulSoup
from twilio.rest import Client
import tempfile, os, json
st.set_page_config(page_title="StockPulse – Anomaly & Forecasting", layout="wide")



# -----------------------------
# Helpers with Caching
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_df(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns.values]
    df.columns = [str(c) for c in df.columns]
    return df.reset_index()

@st.cache_data(show_spinner=False)
def cached_feature_engineer(df, date_col="Date"):
    return feature_engineer(df, date_col)

@st.cache_resource(show_spinner=False)
def cached_anomaly_model(X_train_s, algo, params):
    if algo == "IsolationForest":
        model = IsolationForest(**params, random_state=42)
    elif algo == "OneClassSVM":
        model = OneClassSVM(**params)
    else:
        model = LocalOutlierFactor(**params, novelty=True)
    model.fit(X_train_s)
    return model

@st.cache_resource(show_spinner=False)
def cached_prophet_train(df, horizon, yearly=True, weekly=True):
    model = Prophet(daily_seasonality=True, yearly_seasonality=yearly, weekly_seasonality=weekly)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return model, forecast

# -----------------------------
# Core functions
# -----------------------------
def basic_preprocess(df, date_col="Date"):
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    if date_col not in df.columns:
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    return df, date_col

def feature_engineer(df, date_col="Date"):
    out = df.copy()
    candidates = [c for c in out.columns if "close" in c.lower()]
    price_col = candidates[0] if candidates else out.select_dtypes(include=np.number).columns[0]

    out["return"] = out[price_col].pct_change().fillna(0.0)
    out["ma_10"] = out[price_col].rolling(10, min_periods=1).mean()
    out["ma_20"] = out[price_col].rolling(20, min_periods=1).mean()
    out["vol_roll_10"] = out["return"].rolling(10, min_periods=1).std().fillna(0.0)

    try:
        out["rsi_14"] = RSIIndicator(close=out[price_col], window=14).rsi()
    except Exception:
        out["rsi_14"] = np.nan
    out["rsi_14"] = out["rsi_14"].fillna(method="bfill").fillna(method="ffill")

    out["ma_spread"] = out["ma_10"] - out["ma_20"]

    features = ["return", "vol_roll_10", "rsi_14", "ma_spread"]
    if "Volume" in out.columns:
        out["vol_norm"] = (out["Volume"] - out["Volume"].rolling(20, min_periods=1).mean()) / (
            out["Volume"].rolling(20, min_periods=1).std().replace(0, np.nan)
        )
        out["vol_norm"] = out["vol_norm"].fillna(0.0)
        features.append("vol_norm")

    X = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, X, features, price_col

def plot_series_with_anoms(df, date_col, price_col, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[price_col], mode="lines", name="Price", line=dict(color="blue")))
    anom_idx = np.where(preds == -1)[0]
    if len(anom_idx) > 0:
        fig.add_trace(go.Scatter(
            x=df.iloc[anom_idx][date_col],
            y=df.iloc[anom_idx][price_col],
            mode="markers", name="Anomaly", marker=dict(size=8, symbol="x", color="red")
        ))
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
    return fig

def download_csv_button(df, filename, label):
    csv = df.to_csv(index=False).encode()
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

def safe_has_ohlc(df):
    cols = set([c.lower() for c in df.columns])
    return all(k in cols for k in ["open", "high", "low"])
# -----------------------------
# Sidebar – Data Sources
# -----------------------------
st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.header("Data Source")
source = st.sidebar.radio("Load data from", ["Yahoo Finance (recommended)", "Upload CSV"])
date_default_start = pd.Timestamp.today() - pd.DateOffset(years=2)
date_default_end = pd.Timestamp.today()

if source == "Yahoo Finance (recommended)":
    ticker = st.sidebar.text_input("Ticker (e.g., AAPL, MSFT, GOOG)", value="AAPL")
    start = st.sidebar.date_input("Start date", value=date_default_start)
    end = st.sidebar.date_input("End date", value=date_default_end)
    if st.sidebar.button("Fetch"):
        st.session_state["data"] = fetch_ticker_df(ticker, start, end)
        st.session_state["data_name"] = f"{ticker}.yfinance"
else:
    uploaded = st.sidebar.file_uploader("Upload CSV with Date, Open, High, Low, Close, Volume", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.session_state["data"] = df_up
        st.session_state["data_name"] = uploaded.name

# -----------------------------
# Sidebar – Portfolio Loader
# -----------------------------
st.sidebar.subheader("📊 Portfolio")
portfolio_input = st.sidebar.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOG")

if st.sidebar.button("Load Portfolio"):
    tickers = [t.strip().upper() for t in portfolio_input.split(",") if t.strip()]
    portfolio_data = {}
    for t in tickers:
        try:
            df_t = fetch_ticker_df(t, start=date_default_start, end=date_default_end)
            df_t, date_col_t = basic_preprocess(df_t)
            df_feat_t, X_t, features_t, price_col_t = feature_engineer(df_t, date_col=date_col_t)
            portfolio_data[t] = (df_t, df_feat_t, X_t, features_t, price_col_t)
        except Exception as e:
            st.sidebar.error(f"⚠️ Failed to load {t}: {e}")
    st.session_state["portfolio_data"] = portfolio_data

# -----------------------------
# Main Tabs (2 tabs only)
# -----------------------------
st.title("📈 StockPulse — Stock Price Anomaly & Forecasting System")
st.caption("Detect unusual spikes/dips & forecast future prices with ML-powered models.")

data = st.session_state.get("data")
if data is None:
    st.info("Load a dataset from the sidebar to begin.")
    st.stop()

df, date_col = basic_preprocess(data)
df_feat, X, features, price_col = feature_engineer(df, date_col=date_col)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚨 Anomaly Detection",
    "🔮 Forecasting",
    "📊 Portfolio Dashboard",
    "ℹ️ About / Tech Stack",
    "📰 News & Insights",
    "📧 Email Report"
])


# ----------------------------- TAB 1 -----------------------------
with tab1:
    st.subheader("Anomaly Detection")

    with st.expander("ℹ️ About this tab"):
        st.markdown("""
        This module identifies unusual patterns in the stock price data:
        - Uses **machine learning algorithms** (Isolation Forest, One-Class SVM, Local Outlier Factor).
        - Flags points that differ significantly from the normal trend.
        - Provides multiple visualizations (line chart, candlestick, histograms, scatter plots).
        - Offers KPIs and drill-down to understand anomalies in context.
        """)

    # Train/test split
    test_size = st.slider("Test size (last % for testing)", 10, 50, 20, 5)
    split_idx = int(len(X) * (1 - test_size/100))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    df_train, df_test = df_feat.iloc[:split_idx], df_feat.iloc[split_idx:]

    # Algorithm choice (added LOF)
    algo = st.selectbox("Algorithm", ["IsolationForest", "OneClassSVM", "LocalOutlierFactor"])
    if algo == "IsolationForest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        contamination = st.slider("contamination", 0.01, 0.2, 0.03, 0.01)
        model = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    elif algo == "OneClassSVM":
        nu = st.slider("nu", 0.01, 0.2, 0.05, 0.01)
        gamma = st.selectbox("gamma", ["scale", "auto"])
        model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    else:
        n_neighbors = st.slider("n_neighbors (LOF)", 10, 50, 20, 5)
        contamination = st.slider("contamination (LOF)", 0.01, 0.2, 0.05, 0.01)
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)

    # Scale + fit
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    with st.spinner("Training model…"):
        model.fit(X_train_s)

    # Predictions + scores
    preds_test = model.predict(X_test_s)                      # 1 normal, -1 anomaly
    if hasattr(model, "decision_function"):
        scores_test = model.decision_function(X_test_s)
    else:
        scores_test = -model.score_samples(X_test_s)

    df_plot = df_test.copy().reset_index(drop=True)
    df_plot["pred"], df_plot["score"] = preds_test, scores_test

    # ------- KPI Cards -------
    anomalies_all = df_plot[df_plot["pred"] == -1]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Anomalies", len(anomalies_all))
    col2.metric("Anomaly %", f"{(100*len(anomalies_all)/len(df_plot)):.2f}%")
    col3.metric("Strongest Anomaly Score",
                f"{anomalies_all['score'].min():.3f}" if not anomalies_all.empty else "—")
    col4.metric("Mean Score", f"{df_plot['score'].mean():.3f}")
    first_anom = anomalies_all[date_col].iloc[0].date().isoformat() if not anomalies_all.empty else "—"
    col5.metric("First Anomaly Date", first_anom)

    # ------- Score filter -------
    st.markdown("**Anomaly strength filter** (lower score = stronger anomaly)")
    pct = st.slider("Show anomalies below this score percentile", 1, 50, 20, 1)
    threshold = np.percentile(df_plot["score"], pct)
    anomalies = df_plot[(df_plot["pred"] == -1) & (df_plot["score"] <= threshold)]

    # ------- Chart selector -------
    chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True)
    if chart_type == "Line" or not safe_has_ohlc(df_plot):
        fig_main = plot_series_with_anoms(df_plot, date_col, price_col, df_plot["pred"].values)
        if len(anomalies) > 0:
            fig_main.add_trace(go.Scatter(
                x=anomalies[date_col], y=anomalies[price_col],
                mode="markers", name=f"Top {pct}%-tile anomalies",
                marker=dict(size=10, symbol="diamond", color="orange")
            ))
    else:
        def find_col(name):
            for c in df_plot.columns:
                if c.lower() == name:
                    return c
            return None
        open_c, high_c, low_c = find_col("open"), find_col("high"), find_col("low")
        fig_main = go.Figure(data=[go.Candlestick(
            x=df_plot[date_col],
            open=df_plot.get(open_c, df_plot[price_col]),
            high=df_plot.get(high_c, df_plot[price_col]),
            low=df_plot.get(low_c,  df_plot[price_col]),
            close=df_plot[price_col],
            name="Price"
        )])
        if len(anomalies_all) > 0:
            fig_main.add_trace(go.Scatter(
                x=anomalies_all[date_col], y=anomalies_all[price_col],
                mode="markers", name="Anomaly", marker=dict(size=8, color="red", symbol="x")
            ))
        if len(anomalies) > 0:
            fig_main.add_trace(go.Scatter(
                x=anomalies[date_col], y=anomalies[price_col],
                mode="markers", name=f"Top {pct}%-tile anomalies",
                marker=dict(size=10, color="orange", symbol="diamond")
            ))
    fig_main.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_main, use_container_width=True)

    # ------- Analytics -------
    st.markdown("### Anomaly Analytics")
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=df_plot["score"], nbinsx=30, name="Scores"))
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        colors = np.where(df_plot["pred"] == -1, "red", "blue")
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=df_plot["rsi_14"], y=df_plot["return"],
                                         mode="markers", marker=dict(color=colors)))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("#### Volatility vs Return (Bubble = Volume)")
    bubble_size = np.log1p(df_plot.get("Volume", pd.Series(10, index=df_plot.index)))
    fig_bubble = go.Figure()
    fig_bubble.add_trace(go.Scatter(
        x=df_plot["vol_roll_10"], y=df_plot["return"], mode="markers",
        marker=dict(size=bubble_size, color=np.where(df_plot["pred"] == -1, "red", "blue"))
    ))
    st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("#### Monthly Anomaly Frequency")
    df_plot["month"] = df_plot[date_col].dt.to_period("M")
    month_counts = df_plot.groupby("month")["pred"].apply(lambda x: (x == -1).sum())
    if len(month_counts) > 0:
        fig_heat = go.Figure(data=go.Heatmap(
            z=month_counts.values.reshape(1, -1),
            x=month_counts.index.astype(str),
            y=["Anomaly Count"], colorscale="Reds"
        ))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ------- Drill-down -------
    st.markdown("### 🔎 Drill-down Explorer")
    if len(anomalies_all) > 0:
        sel = st.selectbox("Select an anomaly date",
                           anomalies_all[date_col].dt.strftime("%Y-%m-%d").unique())
        sel_dt = pd.to_datetime(sel)
        window = st.slider("Context window (days before/after)", 5, 30, 10, 1)
        df_zoom = df_plot[(df_plot[date_col] >= sel_dt - pd.Timedelta(days=window)) &
                          (df_plot[date_col] <= sel_dt + pd.Timedelta(days=window))]
        fig_zoom = go.Figure()
        fig_zoom.add_trace(go.Scatter(x=df_zoom[date_col], y=df_zoom[price_col],
                                      mode="lines+markers", name="Price"))
        z_anoms = df_zoom[df_zoom["pred"] == -1]
        if len(z_anoms) > 0:
            fig_zoom.add_trace(go.Scatter(x=z_anoms[date_col], y=z_anoms[price_col],
                                          mode="markers", marker=dict(color="red", size=10)))
        st.plotly_chart(fig_zoom, use_container_width=True)

    # ------- Export anomalies -------
    st.markdown("### 📤 Export")
    st.write(f"Flagged anomalies shown: **{len(anomalies)}**")
    st.dataframe(anomalies[[date_col, price_col] + features + ["score"]].tail())
    download_csv_button(anomalies[[date_col, price_col] + features + ["score"]],
                        "anomalies.csv", "📥 Download anomalies (CSV)")


# -----------------------------
# Tab 2 - Forecasting
# -----------------------------
with tab2:
    st.subheader("Future Price Forecasting")

    with st.expander("ℹ️ About this tab"):
        st.markdown("""
        This section predicts **future stock prices** using time series models:
        - Choose between **Prophet** and **ARIMA**.
        - Forecast daily prices for a user-defined horizon.
        - Evaluate accuracy with multiple metrics (MAPE, RMSE, MAE, R²).
        - Visualize **forecast, residuals, error distribution, and seasonality**.
        - Export forecast results and charts for reporting.
        """)

    # User options
    horizon = st.slider("Forecast horizon (days)", 7, 180, 30, 7)
    model_choice = st.selectbox("Choose forecasting model", ["Prophet", "ARIMA"])

    df_prophet = df[[date_col, price_col]].rename(columns={date_col: "ds", price_col: "y"})

    if model_choice == "Prophet":
        # Seasonality controls
        yearly = st.checkbox("Include yearly seasonality", True)
        weekly = st.checkbox("Include weekly seasonality", True)

        model_f = Prophet(daily_seasonality=True,
                          yearly_seasonality=yearly,
                          weekly_seasonality=weekly)
        model_f.fit(df_prophet)
        future = model_f.make_future_dataframe(periods=horizon)
        forecast = model_f.predict(future)

        # Main forecast plot
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
        fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines",
                                          name="Upper bound", line=dict(dash="dot")))
        fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines",
                                          name="Lower bound", line=dict(dash="dot")))
        fig_forecast.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name="History"))
        fig_forecast.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Train/test split for evaluation
        train_size = int(len(df_prophet) * 0.8)
        train, test = df_prophet[:train_size], df_prophet[train_size:]
        model_f2 = Prophet(daily_seasonality=True)
        model_f2.fit(train)
        forecast_test = model_f2.predict(test[["ds"]])

        # Metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        mape = mean_absolute_percentage_error(test["y"], forecast_test["yhat"])
        rmse = sqrt(mean_squared_error(test["y"], forecast_test["yhat"]))
        mae = mean_absolute_error(test["y"], forecast_test["yhat"])
        r2 = r2_score(test["y"], forecast_test["yhat"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAPE", f"{mape:.2%}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("MAE", f"{mae:.2f}")
        c4.metric("R²", f"{r2:.3f}")

        # Residuals plot
        residuals = test["y"].values - forecast_test["yhat"].values
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(x=test["ds"], y=residuals, mode="lines+markers", name="Residuals"))
        fig_resid.update_layout(title="Residuals over Time")
        st.plotly_chart(fig_resid, use_container_width=True)

        # Scatter actual vs forecast
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=test["y"], y=forecast_test["yhat"], mode="markers", name="Actual vs Predicted"
        ))
        fig_scatter.add_trace(go.Line(x=test["y"], y=test["y"], name="Ideal Fit", line=dict(color="red")))
        fig_scatter.update_layout(title="Actual vs Predicted")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Seasonality plots
        with st.expander("🔍 Seasonality Components"):
            from prophet.plot import plot_components_plotly
            fig_season = plot_components_plotly(model_f, forecast)
            st.plotly_chart(fig_season, use_container_width=True)

        # Export forecast CSV
        st.markdown("### 📤 Export Forecast")
        out_forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={"ds": str(date_col), "yhat": f"{price_col}_forecast",
                     "yhat_lower": "lower", "yhat_upper": "upper"}
        )
        st.dataframe(out_forecast.tail())
        download_csv_button(out_forecast, "forecast.csv", "📥 Download forecast (CSV)")

    else:  # ARIMA
        st.warning("ARIMA support coming soon. Prophet is currently enabled.")

# -----------------------------
# Tab 3 - Portfolio Dashboard
# -----------------------------
with tab3:
    st.subheader("📊 Portfolio Dashboard")

    # Expecting portfolio_data as a dictionary like:
    # { "AAPL": (df, df_feat, X, features, price_col), "MSFT": (...), ... }

    portfolio_data = st.session_state.get("portfolio_data", {})

    if portfolio_data:
        # ---- KPI Summary ----
        st.markdown("### 📌 Portfolio Overview")
        kpi_cols = st.columns(len(portfolio_data))

        for i, (ticker, (df_t, df_feat, X, features, price_col)) in enumerate(portfolio_data.items()):
            with kpi_cols[i]:
                st.metric(f"{ticker} Rows", len(df_feat))
                st.metric("Latest Price", f"{df_feat[price_col].iloc[-1]:.2f}")

        # ---- Charts & Forecast per Ticker ----
        for ticker, (df_t, df_feat, X, features, price_col) in portfolio_data.items():
            st.markdown(f"## 📈 {ticker} Price History")

            # Price history
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_feat[date_col],
                y=df_feat[price_col],
                mode="lines",
                name=f"{ticker} Price"
            ))
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Quick forecast with Prophet
            st.markdown(f"🔮 **30-day Forecast for {ticker}**")

            df_prophet = df_feat[[date_col, price_col]].rename(
                columns={date_col: "ds", price_col: "y"}
            )
            model_f = Prophet(daily_seasonality=True)
            model_f.fit(df_prophet)
            future = model_f.make_future_dataframe(periods=30)
            forecast = model_f.predict(future)

            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"
            ))
            fig_forecast.add_trace(go.Scatter(
                x=df_prophet["ds"], y=df_prophet["y"], mode="lines", name="History"
            ))
            fig_forecast.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Optional: anomaly highlight in portfolio
            st.markdown(f"🚨 **Recent Anomalies in {ticker}**")
            if "pred" in df_feat.columns:
                anomalies = df_feat[df_feat["pred"] == -1]
                if not anomalies.empty:
                    st.dataframe(anomalies[[date_col, price_col]].tail())
                else:
                    st.success("No anomalies detected for this ticker.")
            else:
                st.info("Run anomaly detection tab first to populate anomalies.")

    else:
        st.info("⚠️ No portfolio loaded. Fetch a portfolio from the sidebar to see the dashboard.")



# -----------------------------
# Tab 4 - About / Tech Stack
# -----------------------------
with tab4:
    st.markdown("## ℹ️ About StocksPulse")
    st.markdown(
        "<p style='font-size:18px;'>"
        "🚀 <b>StocksPulse</b> is a machine learning–powered system for "
        "<b>stock price anomaly detection</b> and <b>forecasting future trends</b>. "
        "It was designed as part of a competition project to showcase data science, ML, and UI/UX skills."
        "</p>",
        unsafe_allow_html=True
    )

    # --- Tech Stack ---
    st.markdown("### 🛠 Tech Stack")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Frontend & UI**")
        st.markdown("- Streamlit  \n- Plotly")
    with col2:
        st.markdown("**Data & Features**")
        st.markdown("- yFinance  \n- pandas / numpy  \n- ta (technical indicators)")
    with col3:
        st.markdown("**ML & Forecasting**")
        st.markdown("- scikit-learn (Isolation Forest, One-Class SVM, LOF)  \n- Prophet  \n- joblib")

    # --- Features ---
    st.markdown("### 📊 Key Features")
    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.info("🔎 Anomaly detection with multiple algorithms")
        st.info("📈 KPI summary cards for quick insights")
        st.info("🖼 Multiple anomaly visualizations (line, candlestick, heatmap, scatter, bubble)")
        st.info("🧭 Drill-down anomaly explorer with zoom")
    with feat_col2:
        st.success("🔮 Prophet-based forecasting with seasonality")
        st.success("📏 Forecast metrics: MAPE, RMSE, MAE, R²")
        st.success("📅 Residual & seasonality analysis")
        st.success("📤 Export anomalies & forecasts (CSV)")

    # --- Footer ---
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:16px;'>"
        "✅ Built with ❤️ using <b>Python</b> & <b>Streamlit</b> • "
        "Powered by <b>Machine Learning</b> & <b>Data Science</b>"
        "</p>",
        unsafe_allow_html=True
    )


import streamlit.components.v1 as components

with tab5:
    st.subheader("📰 Latest Stock News")

    ticker_symbol = st.session_state.get("data_name", "AAPL").split(".")[0]

    try:
        url = f"https://news.google.com/rss/search?q={ticker_symbol}+stock&hl=en-US&gl=US&ceid=US:en"
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")[:8]

        news_html = """
        <div style='display:flex; flex-wrap:wrap; gap:20px;'>
        """

        for item in items:
            title = item.title.text
            link = item.link.text
            pub_date = item.pubDate.text

            desc_html = item.description.text if item.description else ""
            desc = BeautifulSoup(desc_html, "html.parser").get_text()

            news_html += f"""
            <div style="
                flex:1 1 calc(50% - 20px);
                border:1px solid #ddd; border-radius:12px;
                padding:15px; background:white;
                box-shadow:0 4px 12px rgba(0,0,0,0.1);
                transition:transform 0.2s; min-width:300px;
            " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
                
                <h4 style="margin:0; color:#003366; font-size:16px;">
                    <a href="{link}" target="_blank" style="text-decoration:none; color:#003366;">
                        📌 {title}
                    </a>
                </h4>
                <p style="font-size:14px; color:#444; margin-top:8px;">{desc}</p>
                <p style="font-size:12px; color:#777; margin-top:10px;">
                    📅 {pub_date}
                </p>
            </div>
            """

        news_html += "</div>"

        # 🚨 Auto-height and no scrollbars
        components.html(news_html, height=1500, scrolling=False)

    except Exception as e:
        st.error("⚠️ Could not fetch news.")
        st.exception(e)

from io import BytesIO

with tab6:
    st.subheader("📧 Send StocksPulse Report")

    st.markdown("""
    Enter your email below to receive **anomalies, forecasts, and portfolio summaries**  
    directly in your inbox.  
    """)

    receiver_email = st.text_input("Recipient Email", "")

    if st.button("Send Report"):
        if not receiver_email:
            st.error("⚠️ Please enter your email.")
        else:
            try:
                import smtplib
                from email.mime.multipart import MIMEMultipart
                from email.mime.text import MIMEText
                from email.mime.base import MIMEBase
                from email import encoders

                # --- FIXED sender account (add to Streamlit secrets) ---
                sender_email = st.secrets["sender_email"]
                sender_pass = st.secrets["sender_pass"]

                # Collect in-memory attachments
                attachments = []

                if "anomalies" in locals() and not anomalies.empty:
                    buf = BytesIO()
                    anomalies.to_csv(buf, index=False)
                    buf.seek(0)
                    attachments.append(("Anomalies.csv", buf.read()))

                if "out_forecast" in locals():
                    buf = BytesIO()
                    out_forecast.to_csv(buf, index=False)
                    buf.seek(0)
                    attachments.append(("Forecast.csv", buf.read()))

                if "portfolio_data" in st.session_state and st.session_state["portfolio_data"]:
                    # Example: save summary portfolio table
                    portfolio_summary = pd.DataFrame({
                        "Ticker": list(st.session_state["portfolio_data"].keys()),
                        "Latest Price": [df_feat.iloc[-1][price_col] for _, (df_t, df_feat, X, fts, price_col) in st.session_state["portfolio_data"].items()]
                    })
                    buf = BytesIO()
                    portfolio_summary.to_csv(buf, index=False)
                    buf.seek(0)
                    attachments.append(("Portfolio.csv", buf.read()))

                # Create email
                msg = MIMEMultipart()
                msg["From"] = sender_email
                msg["To"] = receiver_email
                msg["Subject"] = "📊 StocksPulse Report"

                body = f"""
                <h2>📈 StocksPulse Report</h2>
                <p>Attached are your latest anomaly detections, forecasts, and portfolio summaries from StocksPulse.</p>
                <p>✅ Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <br>
                <p style="color:gray; font-size:12px;">
                Built with ❤️ using Python & Streamlit
                </p>
                """
                msg.attach(MIMEText(body, "html"))

                # Attach files
                for fname, fbytes in attachments:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(fbytes)
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={fname}")
                    msg.attach(part)

                # Send via Gmail SMTP
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, sender_pass)
                server.send_message(msg)
                server.quit()

                st.success(f"📩 Report sent successfully to {receiver_email}!")

            except Exception as e:
                st.error("⚠️ Failed to send email.")
                st.exception(e)

