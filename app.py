import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Apple Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background-color: #F7FAFC;
}
.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    hist = pd.read_csv("AAPL (5).csv")
    forecast = pd.read_csv("AAPL_30Day_Forecast.csv")
    hist["Date"] = pd.to_datetime(hist["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    return hist, forecast

df, forecast_df = load_data()

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Apple Stock Dashboard")
st.sidebar.markdown("**Project:** P-633 Apple Stock Forecast")
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Historical Analysis", "Forecast (30 Days)", "Insights"]
)

# -------------------- OVERVIEW --------------------
if menu == "Overview":
    st.title("📈 Apple Inc. Stock Analysis & Forecast")

    col1, col2, col3, col4 = st.columns(4)

    latest_price = df["Adj Close"].iloc[-1]
    avg_price = df["Adj Close"].mean()
    volatility = df["Adj Close"].std()
    trend = "Bullish 📈" if latest_price > avg_price else "Bearish 📉"

    col1.metric("Latest Price ($)", round(latest_price, 2))
    col2.metric("Average Price ($)", round(avg_price, 2))
    col3.metric("Volatility", round(volatility, 2))
    col4.metric("Market Trend", trend)

    st.markdown("---")

    fig = px.line(
        df,
        x="Date",
        y="Adj Close",
        title="Apple Stock Price Trend",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- HISTORICAL ANALYSIS --------------------
elif menu == "Historical Analysis":
    st.title("📊 Historical Market Analysis")

    ma_20 = df["Adj Close"].rolling(20).mean()
    ma_50 = df["Adj Close"].rolling(50).mean()

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
        x=df["Date"], y=ma_20,
        line=dict(color="blue"),
        name="MA 20"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=ma_50,
        line=dict(color="green"),
        name="MA 50"
    ))

    fig.update_layout(
        title="Candlestick Chart with Moving Averages",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📦 Trading Volume")
    vol_fig = px.bar(
        df,
        x="Date",
        y="Volume",
        template="plotly_white"
    )
    st.plotly_chart(vol_fig, use_container_width=True)

# -------------------- FORECAST --------------------
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
        y=forecast_df["Forecast"],
        name="Forecast",
        line=dict(color="green", dash="dash")
    ))

    fig.update_layout(
        title="Apple Stock Price Forecast (Next 30 Days)",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📄 Forecast Data")
    st.dataframe(forecast_df)

# -------------------- INSIGHTS --------------------
else:
    st.title("📌 Key Insights & Conclusion")

    st.success("""
    ✔ Apple stock shows a long-term upward trend  
    ✔ Moving averages confirm trend stability  
    ✔ Forecast suggests moderate price growth  
    ✔ Suitable for long-term investment analysis  
    """)

    st.info("""
    **Tools Used:**  
    - Python  
    - Pandas & NumPy  
    - Streamlit  
    - Plotly  
    - Time Series Forecasting  
    """)

    st.markdown("📘 **Project:** P-633 Apple Stock Forecast")
    st.markdown("👨‍🎓 **Presented by:** Akash Thakur")
