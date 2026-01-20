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

# ---- Optional TensorFlow (LSTM) ----
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Stock Forecast Dashboard",
    page_icon="🍎",
    layout="wide",
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
    .main {background: #070B14;}
    section[data-testid="stSidebar"] {background-color: #0B1220;}
    .title-main{
        font-size:42px;
        font-weight:900;
        color:white;
        margin-bottom:0px;
        letter-spacing:0.5px;
    }
    .subtitle{
        font-size:15px;
        color:#B5B5B5;
        margin-top:-6px;
        margin-bottom:20px;
    }
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
    .section-title{
        font-size:22px;
        font-weight:900;
        color:white;
        margin-top:20px;
        margin-bottom:6px;
    }
    .small-note{
        font-size:13px;
        color:#C7C7C7;
        margin-bottom:10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SETTINGS ----------------
DATA_PATH = "AAPL (5).csv"


# ---------------- HELPERS ----------------
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

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mp = mape(y_true, y_pred)
    return rmse, mae, mp

def add_indicators(df, target_col):
    df["Daily_Return"] = df[target_col].pct_change()
    df["MA20"] = df[target_col].rolling(20).mean()
    df["MA50"] = df[target_col].rolling(50).mean()
    df["MA200"] = df[target_col].rolling(200).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    return df

def metric_card(title, value, subtitle=""):
    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        <div class="card-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def plotly_price_chart(df, target_col, ma20=True, ma50=True, ma200=True):
    fig = go.Figure()

    # candlestick if open/high/low/close present
    if set(["Open", "High", "Low", "Close"]).issubset(df.columns):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Candlestick"
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df[target_col], mode="lines", name=target_col))

    if ma20 and "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    if ma50 and "MA50" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    if ma200 and "MA200" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

    fig.update_layout(
        template="plotly_dark",
        height=520,
        title="Price + Moving Averages",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------- HEADER ----------------
st.markdown('<div class="title-main">🍎 Apple Stock Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">P-633 Project • Task 1 (Preprocessing) + Task 2 (Forecasting Models)</div>', unsafe_allow_html=True)

# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.title("⚙️ Dashboard Controls")

tab_select = st.sidebar.radio("Go to", ["📌 Task 1: Data Preprocessing", "🔮 Task 2: Forecasting"])

st.sidebar.divider()
st.sidebar.subheader("Forecast Controls")

forecast_days = st.sidebar.slider("Forecast days", 7, 90, 30)

model_options = ["ARIMA", "SARIMA", "Compare All"]
if TF_AVAILABLE:
    model_options.insert(2, "LSTM")
else:
    st.sidebar.warning("TensorFlow not installed → LSTM disabled")

model_choice = st.sidebar.selectbox("Model", model_options, index=0)

st.sidebar.markdown("### ARIMA Params")
p = st.sidebar.slider("p", 0, 10, 5)
d = st.sidebar.slider("d", 0, 2, 1)
q = st.sidebar.slider("q", 0, 10, 0)

st.sidebar.markdown("### SARIMA Params")
sp = st.sidebar.slider("Seasonal p", 0, 2, 1)
sd = st.sidebar.slider("Seasonal d", 0, 2, 1)
sq = st.sidebar.slider("Seasonal q", 0, 2, 1)
seasonal_period = st.sidebar.selectbox("Seasonality (period)", [5, 12, 20, 30], index=2)

if TF_AVAILABLE:
    st.sidebar.markdown("### LSTM Params")
    seq_len = st.sidebar.slider("Sequence Length", 20, 120, 60)
    epochs = st.sidebar.slider("Epochs", 5, 50, 10)
    batch_size = st.sidebar.selectbox("Batch size", [16, 32, 64], index=1)

show_data = st.sidebar.checkbox("Show cleaned data", False)


# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
except Exception:
    st.error(f"❌ Dataset not found: {DATA_PATH}\n\n✅ Put your CSV next to app.py")
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

# fix numeric values
df[target_col] = df[target_col].astype(str).str.replace(",", "").str.replace("$", "")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[target_col])

# business day freq
df = df.asfreq("B")
df[target_col] = df[target_col].ffill()

# indicators
df = add_indicators(df, target_col)
ts = df[target_col].dropna()

# ---------------- KPI SECTION ----------------
latest_price = ts.iloc[-1]
overall_return = ((ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0]) * 100
high_price = ts.max()
low_price = ts.min()

k1, k2, k3, k4 = st.columns(4)
with k1: metric_card("Latest Price", f"${latest_price:,.2f}", "Last Business Day")
with k2: metric_card("Overall Return", f"{overall_return:.2f}%", "Start → Latest")
with k3: metric_card("All Time High", f"${high_price:,.2f}", "Dataset peak")
with k4: metric_card("All Time Low", f"${low_price:,.2f}", "Dataset minimum")

st.divider()


# ---------------- TASK 1: PREPROCESSING ----------------
if tab_select == "📌 Task 1: Data Preprocessing":
    st.markdown('<div class="section-title">📌 Task 1: Data Quality & Preprocessing</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">This section validates dataset, handles missing values, converts date-time, and prepares time series.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", df.shape[0])
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values (Price)", int(df[target_col].isna().sum()))

    st.subheader("📈 Price Chart (Candlestick + MA)")
    plotly_price_chart(df, target_col, ma20=True, ma50=True, ma200=True)

    st.subheader("📊 Returns & Volatility")
    colA, colB = st.columns(2)

    with colA:
        fig = px.histogram(df.dropna(), x="Daily_Return", nbins=60, title="Daily Returns Distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.line(df, y="Volatility_20", title="20-Day Rolling Volatility (Annualized)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if "Volume" in df.columns:
        st.subheader("📌 Volume Insight")
        fig = px.line(df, y="Volume", title="Volume Over Time", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if show_data:
        st.subheader("📄 Cleaned Dataset")
        st.dataframe(df.tail(30), use_container_width=True)


# ---------------- TASK 2: FORECASTING ----------------
else:
    st.markdown('<div class="section-title">🔮 Task 2: Forecasting Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">Select model, tune parameters, and forecast next business days.</div>', unsafe_allow_html=True)

    split_idx = int(len(ts) * 0.8)
    train, test = ts.iloc[:split_idx], ts.iloc[split_idx:]

    st.write(f"✅ Train: {len(train)} rows | ✅ Test: {len(test)} rows")

    def run_arima():
        fit = ARIMA(train, order=(p, d, q)).fit()
        pred = fit.forecast(steps=len(test))
        rmse, mae, mp = eval_metrics(test, pred)
        future = fit.forecast(steps=forecast_days)
        return pred, future, rmse, mae, mp

    def run_sarima():
        fit = SARIMAX(train, order=(p, d, q), seasonal_order=(sp, sd, sq, seasonal_period)).fit(disp=False)
        pred = fit.forecast(steps=len(test))
        rmse, mae, mp = eval_metrics(test, pred)
        future = fit.forecast(steps=forecast_days)
        return pred, future, rmse, mae, mp

    def run_lstm():
        data = ts.values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        s = seq_len
        if len(scaled) <= s:
            s = max(10, len(scaled)//2)

        X, y = [], []
        for i in range(s, len(scaled)):
            X.append(scaled[i-s:i, 0])
            y.append(scaled[i, 0])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 20:
            raise ValueError("Not enough data for LSTM training.")

        X = X.reshape(X.shape[0], X.shape[1], 1)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled).flatten()

        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        rmse, mae, mp = eval_metrics(y_actual, pred)

        # future
        last_seq = scaled[-s:].reshape(1, s, 1)
        future_scaled = []
        for _ in range(forecast_days):
            p_ = model.predict(last_seq, verbose=0)[0][0]
            future_scaled.append(p_)
            last_seq = np.append(last_seq[:, 1:, :], [[[p_]]], axis=1)

        future = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        pred_series = pd.Series(pred, index=test.index[-len(pred):])
        return pred_series, future, rmse, mae, mp

    if st.button("🚀 Run Forecast"):
        results = []
        pred_final = None
        future_final = None

        if model_choice == "ARIMA":
            pred_final, future_final, rmse, mae, mp = run_arima()
            results.append(["ARIMA", rmse, mae, mp])

        elif model_choice == "SARIMA":
            pred_final, future_final, rmse, mae, mp = run_sarima()
            results.append(["SARIMA", rmse, mae, mp])

        elif model_choice == "LSTM":
            pred_final, future_final, rmse, mae, mp = run_lstm()
            results.append(["LSTM", rmse, mae, mp])

        else:
            pred_a, fut_a, rmse_a, mae_a, mp_a = run_arima()
            pred_s, fut_s, rmse_s, mae_s, mp_s = run_sarima()

            results = [
                ["ARIMA", rmse_a, mae_a, mp_a],
                ["SARIMA", rmse_s, mae_s, mp_s],
            ]

            pred_final, future_final = pred_a, fut_a
            best_rmse = rmse_a

            if rmse_s < best_rmse:
                pred_final, future_final = pred_s, fut_s
                best_rmse = rmse_s

            if TF_AVAILABLE:
                try:
                    pred_l, fut_l, rmse_l, mae_l, mp_l = run_lstm()
                    results.append(["LSTM", rmse_l, mae_l, mp_l])
                    if rmse_l < best_rmse:
                        pred_final, future_final = pred_l, fut_l
                        best_rmse = rmse_l
                except:
                    pass

        st.subheader("📌 Model Metrics (RMSE / MAE / MAPE)")
        res_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "MAPE (%)"])
        st.dataframe(res_df, use_container_width=True)

        st.subheader("✅ Test Forecast vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=pred_final.index, y=pred_final.values, mode="lines", name="Predicted"))
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"📅 Next {forecast_days} Business Days Forecast")
        future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": np.array(future_final).flatten()})

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ts.index[-250:], y=ts.values[-250:], mode="lines", name="Historical"))
        fig2.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines", name="Forecast"))
        fig2.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📄 Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast CSV", data=csv, file_name="AAPL_forecast.csv", mime="text/csv")

    else:
        st.info("👉 Click **🚀 Run Forecast** to generate prediction.")
