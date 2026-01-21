import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly.graph_objects as go
import plotly.express as px

# Optional LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Apple Stock Forecast | Executive Dashboard",
    page_icon="🍎",
    layout="wide"
)

DATA_PATH = "AAPL (5).csv"
FORECAST_DAYS = 30


# ---------------- LIGHT THEME CSS ----------------
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #F7F9FC;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E6E8EF;
    }

    /* Top Titles */
    .brand-title {
        font-size: 34px;
        font-weight: 900;
        color: #0B1F44;
        margin-bottom: 0px;
    }
    .brand-sub {
        font-size: 14px;
        color: #55657A;
        margin-top: -3px;
        margin-bottom: 18px;
    }

    /* Section Titles */
    .section-title {
        font-size: 20px;
        font-weight: 900;
        color: #0B1F44;
        margin-top: 10px;
        margin-bottom: 8px;
    }
    .small-note {
        font-size: 13px;
        color: #55657A;
        margin-bottom: 8px;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E6E8EF;
        border-radius: 18px;
        padding: 16px;
        height: 115px;
        box-shadow: 0 6px 20px rgba(12, 36, 97, 0.06);
    }
    .kpi-title {
        font-size: 12px;
        font-weight: 700;
        color: #6B7A90;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 900;
        color: #0B1F44;
        margin-top: 6px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #6B7A90;
        margin-top: 4px;
    }

    /* Summary Box */
    .summary-box {
        background: #FFFFFF;
        border: 1px solid #E6E8EF;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 6px 20px rgba(12, 36, 97, 0.06);
    }
    .summary-title {
        font-size: 15px;
        font-weight: 900;
        color: #0B1F44;
        margin-bottom: 8px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------
def kpi_card(title, value, subtitle=""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(r"\\s+", " ", regex=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def detect_date_column(df):
    possible = ["date", "datetime", "timestamp", "time"]
    for col in df.columns:
        c = col.lower().replace(" ", "").replace("_", "")
        if c in possible:
            return col
    return df.columns[0]

def detect_target_column(df):
    for col in df.columns:
        c = col.lower().replace(" ", "").replace("_", "")
        if c in ["adjclose", "adj_close", "adjustedclose", "adjustedcloseprice"]:
            return col
    for col in df.columns:
        if col.lower().strip() == "close":
            return col
    return None

def add_indicators(df, target_col):
    df["Daily_Return"] = df[target_col].pct_change()
    df["MA20"] = df[target_col].rolling(20).mean()
    df["MA50"] = df[target_col].rolling(50).mean()
    df["MA200"] = df[target_col].rolling(200).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    return df

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return rmse, mae, mp


# ---------------- SIDEBAR BRANDING ----------------
st.sidebar.markdown("### 📊 CDIT Analytics")
st.sidebar.caption("Executive Forecast Dashboard")
st.sidebar.divider()

show_table = st.sidebar.checkbox("Show cleaned dataset table", False)

st.sidebar.divider()
st.sidebar.markdown("### Models Used")
st.sidebar.write("✅ ARIMA")
st.sidebar.write("✅ SARIMA")
if TF_AVAILABLE:
    st.sidebar.write("✅ LSTM (optional)")
else:
    st.sidebar.write("❌ LSTM disabled")


# ---------------- HEADER ----------------
st.markdown('<div class="brand-title">🍎 Apple Stock Forecast — Executive Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Task 1: Data Quality + EDA | Task 2: Forecasting + 30-Day Outlook</div>', unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except Exception:
    st.error(f"❌ Dataset not found: {DATA_PATH}. Put CSV in same folder as app.py.")
    st.stop()

df = clean_columns(df)

date_col = detect_date_column(df)
target_col = detect_target_column(df)

if target_col is None:
    st.error("❌ Target column not found (Adj_Close or Close).")
    st.stop()

# Rename date
df.rename(columns={date_col: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

# Target numeric fix
df[target_col] = df[target_col].astype(str).str.replace(",", "").str.replace("$", "")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])

# Business-day frequency
df = df.asfreq("B")
df[target_col] = df[target_col].ffill()

# indicators
df = add_indicators(df, target_col)
ts = df[target_col].dropna()

# ---------------- KPI SECTION ----------------
latest = ts.iloc[-1]
overall_return = ((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0]) * 100
high_price = ts.max()
low_price = ts.min()
vol_last = df["Volatility_20"].dropna().iloc[-1] if len(df["Volatility_20"].dropna()) else np.nan

k1, k2, k3, k4 = st.columns(4)
with k1: kpi_card("Latest Price", f"${latest:,.2f}", "Last business day")
with k2: kpi_card("Overall Return", f"{overall_return:.2f}%", "Start → Latest")
with k3: kpi_card("All-time High", f"${high_price:,.2f}", "Peak in dataset")
with k4: kpi_card("Volatility (20D)", f"{vol_last:.2f}", "Annualized")

st.divider()


# ---------------- EXEC SUMMARY ----------------
st.markdown('<div class="section-title">🧾 Executive Summary</div>', unsafe_allow_html=True)
trend = "Uptrend ✅" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Downtrend ⚠️"
risk = "High Risk ⚠️" if vol_last > 0.35 else "Moderate Risk ✅"

left, right = st.columns(2)
with left:
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">Key Insights</div>
        <ul>
            <li><b>Trend Signal:</b> {trend}</li>
            <li><b>Risk Level:</b> {risk}</li>
            <li><b>Overall Return:</b> {overall_return:.2f}%</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">Forecast Objective</div>
        <ul>
            <li>Predict adjusted closing price for next <b>{FORECAST_DAYS} business days</b>.</li>
            <li>Compare ARIMA vs SARIMA (and LSTM if enabled).</li>
            <li>Select best model using <b>RMSE</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ================= TASK 1: EDA =================
st.markdown('<div class="section-title">✅ Task 1: Data Quality & EDA</div>', unsafe_allow_html=True)
st.markdown('<div class="small-note">Interactive charts: zoom/hover/legend toggle.</div>', unsafe_allow_html=True)

# Price chart
fig_price = go.Figure()

if set(["Open","High","Low","Close"]).issubset(df.columns):
    fig_price.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candlestick"
    ))
else:
    fig_price.add_trace(go.Scatter(x=df.index, y=df[target_col], mode="lines", name="Price"))

fig_price.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

fig_price.update_layout(
    template="plotly_white",
    height=520,
    title="Stock Price with Moving Averages",
    legend=dict(orientation="h"),
)
st.plotly_chart(fig_price, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(df.dropna(), x="Daily_Return", nbins=60, title="Daily Returns Distribution", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.line(df, y="Volatility_20", title="20-Day Rolling Volatility", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

if show_table:
    st.subheader("📄 Cleaned Dataset Preview")
    st.dataframe(df.tail(30), use_container_width=True)

st.divider()


# ================= TASK 2: FORECASTING =================
st.markdown('<div class="section-title">🔮 Task 2: Forecasting (30-Day Outlook)</div>', unsafe_allow_html=True)

split_idx = int(len(ts) * 0.8)
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

# ARIMA
arima_fit = ARIMA(train, order=(5,1,0)).fit()
arima_pred = arima_fit.forecast(steps=len(test))
arima_rmse, arima_mae, arima_mape = eval_metrics(test, arima_pred)

# SARIMA
sarima_fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
sarima_pred = sarima_fit.forecast(steps=len(test))
sarima_rmse, sarima_mae, sarima_mape = eval_metrics(test, sarima_pred)

results = [
    ["ARIMA", arima_rmse, arima_mae, arima_mape],
    ["SARIMA", sarima_rmse, sarima_mae, sarima_mape],
]

res_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "MAPE (%)"])
st.subheader("📌 Model Evaluation")
st.dataframe(res_df, use_container_width=True)

best_model = res_df.sort_values("RMSE").iloc[0]["Model"]
st.success(f"✅ Best Model (RMSE): **{best_model}**")

future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="B")

if best_model == "ARIMA":
    future_forecast = arima_fit.forecast(steps=FORECAST_DAYS).values
else:
    future_forecast = sarima_fit.forecast(steps=FORECAST_DAYS).values

forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast})

# Forecast plot test
fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Actual"))
fig_test.add_trace(go.Scatter(x=test.index, y=arima_pred.values, mode="lines", name="ARIMA"))
fig_test.add_trace(go.Scatter(x=test.index, y=sarima_pred.values, mode="lines", name="SARIMA"))
fig_test.update_layout(template="plotly_white", height=440, title="Test Forecast: Actual vs Models")
st.plotly_chart(fig_test, use_container_width=True)

# Future forecast plot
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=ts.index[-250:], y=ts.values[-250:], mode="lines", name="Historical"))
fig_future.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"))
fig_future.update_layout(template="plotly_white", height=440, title="Next 30 Business Days Forecast")
st.plotly_chart(fig_future, use_container_width=True)

st.subheader("📄 Forecast Output")
st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="AAPL_30Day_Forecast.csv", mime="text/csv")
