import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import plotly.graph_objects as go
import plotly.express as px

# ---- TensorFlow optional ----
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------- CONFIG ----------------
st.set_page_config(page_title="Apple Stock Forecast Dashboard", page_icon="🍎", layout="wide")
DATA_PATH = "AAPL (5).csv"
FORECAST_DAYS = 30

# ---------------- STYLE ----------------
st.markdown("""
<style>
    .main {background: #070B14;}
    section[data-testid="stSidebar"] {background-color: #0B1220;}
    .title-main{font-size:42px;font-weight:900;color:white;margin-bottom:0px}
    .subtitle{font-size:15px;color:#B5B5B5;margin-top:-6px;margin-bottom:20px}
    .card{
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
        color: white;
        height: 110px;
    }
    .card-title{font-size:13px;color:#9CA3AF;font-weight:600;}
    .card-value{font-size:28px;font-weight:900;margin-top:6px;}
    .card-sub{font-size:12px;color:#9CA3AF;margin-top:2px;}
</style>
""", unsafe_allow_html=True)


# ---------------- HELPERS ----------------
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


# ---------------- HEADER ----------------
st.markdown('<div class="title-main">🍎 Apple Stock Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Task 1: Data Quality + EDA | Task 2: Forecasting (ARIMA, SARIMA, LSTM)</div>', unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except Exception:
    st.error(f"❌ Dataset not found: {DATA_PATH}. Put file with app.py")
    st.stop()

df = clean_columns(df)
date_col = detect_date_column(df)
target_col = detect_target_column(df)

if target_col is None:
    st.error("❌ Adj_Close / Close column not found.")
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

# ---------------- KPI SECTION ----------------
latest = ts.iloc[-1]
overall_return = ((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0]) * 100
high_price = ts.max()
low_price = ts.min()

k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Latest Price", f"${latest:,.2f}", "Last Business Day")
with k2: metric_card("Overall Return", f"{overall_return:.2f}%", "From first → latest")
with k3: metric_card("All Time High", f"${high_price:,.2f}", "Peak in dataset")
with k4: metric_card("All Time Low", f"${low_price:,.2f}", "Minimum in dataset")

st.divider()

# ======================= TASK 1: EDA =======================
st.subheader("✅ Task 1: Data Quality & EDA")

# Price chart + MA
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode="lines", name="Price"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))
fig.update_layout(template="plotly_dark", height=520, title="Price with Moving Averages")
st.plotly_chart(fig, use_container_width=True)

# Returns + Volatility
c1, c2 = st.columns(2)

with c1:
    fig = px.histogram(df.dropna(), x="Daily_Return", nbins=60, title="Daily Returns Distribution", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(df, y="Volatility_20", title="20-Day Rolling Volatility (Annualized)", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ======================= TASK 2: FORECASTING =======================
st.subheader("🔮 Task 2: Forecasting (ARIMA / SARIMA / LSTM)")

split_idx = int(len(ts) * 0.8)
train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

st.write(f"Train records: **{len(train)}** | Test records: **{len(test)}**")

# -------- ARIMA --------
arima_fit = ARIMA(train, order=(5,1,0)).fit()
arima_pred = arima_fit.forecast(steps=len(test))
arima_rmse, arima_mae, arima_mape = eval_metrics(test, arima_pred)

# -------- SARIMA --------
sarima_fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
sarima_pred = sarima_fit.forecast(steps=len(test))
sarima_rmse, sarima_mae, sarima_mape = eval_metrics(test, sarima_pred)

# -------- LSTM (optional) --------
lstm_available = False
if TF_AVAILABLE and len(ts) > 200:
    lstm_available = True
    data = ts.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

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

    lstm_pred_scaled = model.predict(X_test, verbose=0)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    lstm_rmse, lstm_mae, lstm_mape = eval_metrics(y_test_actual, lstm_pred)

# Metrics Table
results = [
    ["ARIMA", arima_rmse, arima_mae, arima_mape],
    ["SARIMA", sarima_rmse, sarima_mae, sarima_mape],
]
if lstm_available:
    results.append(["LSTM", lstm_rmse, lstm_mae, lstm_mape])

res_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "MAPE (%)"])
st.dataframe(res_df, use_container_width=True)

# Select Best Model
best_model = res_df.sort_values("RMSE").iloc[0]["Model"]
st.success(f"✅ Best model based on RMSE: **{best_model}**")

# Forecast future 30 days using best model
future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="B")

if best_model == "ARIMA":
    future_forecast = arima_fit.forecast(steps=FORECAST_DAYS).values

elif best_model == "SARIMA":
    future_forecast = sarima_fit.forecast(steps=FORECAST_DAYS).values

else:
    # LSTM future forecast
    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    future_scaled = []
    for _ in range(FORECAST_DAYS):
        p_ = model.predict(last_seq, verbose=0)[0][0]
        future_scaled.append(p_)
        last_seq = np.append(last_seq[:, 1:, :], [[[p_]]], axis=1)
    future_forecast = scaler.inverse_transform(np.array(future_scaled).reshape(-1,1)).flatten()

forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_forecast})

# Plot actual vs predicted (ARIMA + SARIMA)
fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Actual"))
fig_test.add_trace(go.Scatter(x=test.index, y=arima_pred.values, mode="lines", name="ARIMA Pred"))
fig_test.add_trace(go.Scatter(x=test.index, y=sarima_pred.values, mode="lines", name="SARIMA Pred"))
fig_test.update_layout(template="plotly_dark", height=450, title="Test Forecast Comparison")
st.plotly_chart(fig_test, use_container_width=True)

# Plot future forecast
fig_future = go.Figure()
fig_future.add_trace(go.Scatter(x=ts.index[-250:], y=ts.values[-250:], mode="lines", name="Historical"))
fig_future.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"))
fig_future.update_layout(template="plotly_dark", height=450, title="Next 30 Business Days Forecast")
st.plotly_chart(fig_future, use_container_width=True)

st.subheader("📄 Forecast Output")
st.dataframe(forecast_df, use_container_width=True)

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="AAPL_forecast_30days.csv", mime="text/csv")
