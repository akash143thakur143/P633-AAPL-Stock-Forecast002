import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Apple Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #F5F7FB;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL (5).csv")
    forecast = pd.read_csv("AAPL_30Day_Forecast.csv")

    df["Date"] = pd.to_datetime(df["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])

    return df, forecast

df, forecast_df = load_data()

# ---------------- SAFE FORECAST COLUMN ----------------
forecast_col = None
for col in forecast_df.columns:
    if col.lower() not in ["date"]:
        forecast_col = col
        break

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Apple Stock Dashboard")
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Historical Analysis", "Forecast (30 Days)", "Insights"]
)

# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.title("📈 Apple Inc. Stock Price Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    latest_price = df["Adj Close"].iloc[-1]
    avg_price = df["Adj Close"].mean()
    volatility = df["Adj Close"].std()
    trend = "Bullish 📈" if latest_price > avg_price else "Bearish 📉"

    col1.metric("Latest Price ($)", f"{latest_price:.2f}")
    col2.metric("Average Price ($)", f"{avg_price:.2f}")
    col3.metric("Volatility", f"{volatility:.2f}")
    col4.metric("Trend", trend)

    fig = px.line(
        df,
        x="Date",
        y="Adj Close",
        title="Apple Adjusted Close Price Trend",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- HISTORICAL ANALYSIS ----------------
elif menu == "Historical Analysis":
    st.title("📊 Historical Market Analysis")

    df["MA20"] = df["Adj Close"].rolling(20).mean()
    df["MA50"] = df["Adj Close"].rolling(50).mean()

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MA20"],
        name="MA 20",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["MA50"],
        name="MA 50",
        line=dict(color="green")
    ))

    fig.update_layout(
        title="Candlestick with Moving Averages",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    vol_fig = px.bar(
        df,
        x="Date",
        y="Volume",
        title="Trading Volume",
        template="plotly_white"
    )
    st.plotly_chart(vol_fig, use_container_width=True)

# ---------------- FORECAST ----------------
elif menu == "Forecast (30 Days)":
    st.title("🔮 30-Day Stock Price Forecast")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Adj Close"],
        name="Historical Price",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df[forecast_col],
        name="Forecast Price",
        line=dict(color="green", dash="dash")
    ))

    fig.update_layout(
        title="Apple Stock Forecast (Next 30 Days)",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Forecast Data")
    st.dataframe(forecast_df)

# ---------------- INSIGHTS ----------------
else:
    st.title("📌 Key Insights")

    st.success(
        "• Apple stock shows a consistent upward trend\n"
        "• Moving averages confirm trend stability\n"
        "• Forecast indicates moderate future growth\n"
        "• Useful for long-term investment decisions"
    )

    st.info(
        "Tools Used:\n"
        "• Python\n"
        "• Pandas\n"
        "• Streamlit\n"
        "• Plotly\n"
        "• Time Series Forecasting"
    )

    st.markdown("👨‍🎓 **Presented by:** Akash Thakur & Group")
    st.markdown("📘 **Project:** P-633 Apple Stock Forecast")

