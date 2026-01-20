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

# ---------- Optional LSTM ----------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# ---------- CONFIG ----------
st.set_page_config(page_title="AAPL Forecast | Executive Dashboard", page_icon="🍎", layout="wide")
DATA_PATH = "AAPL (5).csv"
FORECAST_DAYS = 30

# ---------- PREMIUM COMPANY CSS ----------
st.markdown("""
<style>
    .main {background: #050A16;}
    section[data-testid="stSidebar"] {background-color: #0A1224;}
    .brand-title{font-size:34px;font-weight:900;color:white;margin-bottom:0px}
    .brand-sub{font-size:14px;color:#B5B5B5;margin-top:-4px;margin-bottom:18px}
    .section-title{font-size:22px;font-weight:900;color:white;margin-top:8px;margin-bottom:6px}
    .sub-note{font-size:13px;color:#CFCFCF;margin-bottom:10px}

    .card{
        background: linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 16px;
        color: white;
        height: 115px;
    }
    .card-title{font-size:13px;color:#9CA3AF;font-weight:700;}
    .card-value{font-size:28px;font-weight:900;margin-top:6px;}
    .card-sub{font-size:12px;color:#9CA3AF;margin-top:2px;}

    .summary-box{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 16px;
        color: white;
    }
    .summary-title{font-size:16px;font-weight:900;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def metric_card(title, value, subtitle=""):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(r"\s+", " ", regex=True)
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

# ---------- SIDEBAR BRANDING ----------
st.sidebar.markdown("## 🏢 CDIT Analytics")
st.sidebar.caption("Executive Forecasting Dashboard")
st.sidebar.divider()

st.sidebar.markdown("### Controls")
show_table = st.sidebar.checkbox("Show cleaned dataset table", False)
st.sidebar.caption("Model params are fixed in this executive dashboard.")

st.sidebar.divider()
st.sidebar.markdown("### Models Used")
st.sidebar.write("✅ ARIMA")
st.sidebar.write("✅ SARIMA")
if TF_AVAILABLE:
    st.sidebar.write("✅ LSTM (optional)")
else:
    st.sidebar.write("❌ LSTM disabled (TensorFlow missing)")

# ---------- HEADER ----------
st.markdown('<div class="brand-title">🍎 Apple Stock Forecast — Executive Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Task 1: Data Quality + EDA | Task 2: Forecasting + 30-Day Outlook | Built for Business Stakeholders</div>', unsafe_allow_html=True)

# ---------- LOAD DATA ----------
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except Exception:
    st.error(f"❌ Dataset not found: {DATA_PATH}. Put CSV in same folder as app.py")
    st.stop()

df = clean_columns(df)
date_col = detect_date_column(df)
target_col = detect_target_column(df)

if target_col is None:
    st.error("❌ Target column not found (Adj_Close or Close).")
    st.stop()

df.rename(columns={date_col: "Date"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

df[target_col] = df[target_col].astype(str).str.replace(",", "").str.replace("$", "")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])

df = df.asfreq("B")
df[target_col] = df[target_col].ffill()
df = add_indicators(df, target_col)

ts = df[target_col].dropna()

# ---------- KPI ----------
latest = ts.iloc[-1]
overall_return = ((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0]) * 100
high_price = ts.max()
low_price = ts.min()
vol_last = df["Volatility_20"].dropna().iloc[-1] if len(df["Volatility_20"].dropna()) else np.nan

k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Latest Price", f"${latest:,.2f}", "Last Business Day")
with k2: metric_card("Overall Return", f"{overall_return:.2f}%", "Start → Latest")
with k3: metric_card("All-Time High", f"${high_price:,.2f}", "Peak Value")
with k4: metric_card("Volatility (20D)", f"{vol_last:.2f}", "Annualized")

st.divider()

# ---------- EXECUTIVE SUMMARY ----------
st.markdown('<div class="section-title">🧾 Executive Summary</div>', unsafe_allow_html=True)

trend = "Uptrend ✅" if df["MA20"].iloc[-1] > df["MA50"].iloc[-1] else "Downtrend ⚠️"
risk = "High risk ⚠️" if vol_last > 0.35 else "Moderate risk ✅"

summary_left, summary_right = st.columns([1.2, 1])

with summary_left:
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">Key Findings</div>
        <ul>
            <li><b>Trend Signal:</b> {trend}</li>
            <li><b>Risk Level:</b> {risk}</li>
            <li><b>Return:</b> {overall_return:.2f}% over dataset timeline</li>
            <li><b>Decision Use-case:</b> Supports investment planning and portfolio projection</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with summary_right:
    st.markdown(f"""
    <div class="summary-box">
        <div class="summary-title">Forecast Strategy</div>
        <ul>
            <li>ARIMA & SARIMA used for statistical forecasting.</li>
            <li>LSTM used for deep learning forecasting (if TensorFlow enabled).</li>
            <li>Best model selected using <b>RMSE</b>.</li>
            <li>Forecast horizon: <b>{FORECAST_DAYS} business days</b>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ======================= TASK 1: EDA =======================
st.markdown('<div class="section-title">✅ Task 1: Data Quality & EDA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-note">Charts are interactive (zoom, hover, legends). Moving averages help identify long-term momentum.</div>', unsafe_allow_html=True)

# Candlestick + MA
fig = go.Figure()

if set(["Open","High","Low","Close"]).issubset(df.columns):
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candlestick"
    ))
else:
    fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode="lines", name="Price"))

fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

fig.update_layout(template="plotly_dark", height=540, title="Price Movement with Moving Averages")
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(df.dropna(), x="Daily_Return", nbins=60, title="Daily Returns Distribution", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.line(df, y="Volatility_20", title="20-Day Rolling Volatility", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

if show_table:
    st.subheader("📄 Cleaned Dataset (Preview)")
    st.dataframe(df.tail(30), use_container_width=True)

st.divider()

# ======================= TASK 2: FORECASTING =======================
st.markdown('<div class="section-title">🔮 Task 2: Forecasting Models & 30-Day Outlook</div>', unsafe_allow_html=True)

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

# Optional LSTM
if TF_AVAILABLE and len(ts) > 200:
    try:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(ts.values.reshape(-1,1))

        seq_len = 60
        X, y = [], []
        for i in range(seq_len, len(scaled)):
            X.append(scaled[i-seq_len:i, 0])
            y.append(scaled[i, 0])

        X = np.array(X).reshape(-1, seq_len, 1)
        y = np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        pred_scaled = model.predict(X_test, verbose=0)
        lstm_pred = scaler.inverse_transform(pred_scaled).flatten()

        y_actual = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        lstm_rmse, lstm_mae, lstm_mape = eval_metrics(y_actual, lstm_pred)

        results.append(["LSTM", lstm_rmse, lstm_mae, lstm_mape])
    except:
        pass

res_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "MAPE (%)"])
st.subheader("📌 Model Evaluation Table")
st.dataframe(res_df, use_container_width=True)

best_model = res_df.sort_values("RMSE").iloc[0]["Model"]
st.success(f"✅ Best Model Selected (RMSE): **{best_model}**")

# future forecast
future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="B")

if best_model == "ARIMA":
    future_forecast = arima_fit.forecast(steps=FORECAST_DAYS).values
elif best_model == "SARIMA":
    future_forecast = sarima_fit.forecast(steps=FORECAST_DAYS).values
else:
    # if LSTM selected but future not trained in this executive mode -> fallback SARIMA
    future_forecast = sarima_fit.forecast(steps=FORECAST_DAYS).values

forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast})

# test forecast chart
fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Actual"))
fig_test.add_trace(go.Scatter(x=test.index, y=arima_pred.values, mode="lines", name="ARIMA"))
fig_test.add_trace(go.Scatter(x=test.index, y=sarima_pred.values, mode="lines", name="SARIMA"))
fig_test.update_layout(template="plotly_dark", height=460, title="Test Forecast: Actual vs ARIMA vs SARIMA")
st.plotly_chart(fig_test, use_container_width=True)

# future chart
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=ts.index[-250:], y=ts.values[-250:], mode="lines", name="Historical"))
fig_future.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"))
fig_future.update_layout(template="plotly_dark", height=460, title="Future Forecast (30 Business Days)")
st.plotly_chart(fig_future, use_container_width=True)

st.subheader("📄 Forecast Output Table")
st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="AAPL_30Day_Forecast.csv", mime="text/csv")
