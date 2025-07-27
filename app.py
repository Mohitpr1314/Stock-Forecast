import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import keras

# Stock options
stocks = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms, Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BRK-B": "Berkshire Hathaway Inc. (Class B)",
    "V": "Visa Inc.",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys Ltd",
    "HDFCBANK.NS": "HDFC Bank Ltd",
    "SBIN.NS": "State Bank of India",
    "RELIANCE.NS": "Reliance Industries Ltd"
}

st.set_page_config(page_title="Stock Price Forecast", layout="wide")
st.title("📈 Stock Price Forecast Using LSTM")

# Ticker selection only (no date input)
selected_stock = st.sidebar.selectbox(
    "Choose a stock",
    options=list(stocks.keys()),
    format_func=lambda x: f"{stocks[x]} ({x})"
)

# Fixed date range
START = '2010-01-01'
END = '2025-06-30'

@st.cache_resource
def load_forecast_model():
    model = load_model("seq2seq_lstm_stock_forecast.h5", compile=False)
    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
    return model

model = load_forecast_model()

st.write(f"### Fetching data for **{stocks[selected_stock]}** from {START} to {END}...")
data = yf.download(selected_stock, start=START, end=END)
data = data[['Close']].dropna()

if data.empty:
    st.error("⚠️ No data found for this stock in that date range!")
    st.stop()

scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(data[['Close']])

lookback_days = 100
forecast_days = 90

last_sequence = scaled_close[-lookback_days:].reshape(1, lookback_days, 1)
predicted_scaled = model.predict(last_sequence)
predicted_prices = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

future_dates = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq='B'
)

st.write("## Forecast Graphs")

# --- 30 Days ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index[-200:], data['Close'][-200:], label="Historical Data")
ax1.plot(future_dates[:30], predicted_prices[:30], 'o-', color='red', label="Next 30 Days")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.xticks(rotation=45)
plt.title(f"{stocks[selected_stock]} ({selected_stock}) - 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig1)

# --- 60 Days ---
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data.index[-200:], data['Close'][-200:], label="Historical Data")
ax2.plot(future_dates[:60], predicted_prices[:60], 'o-', color='orange', label="Next 60 Days")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.xticks(rotation=45)
plt.title(f"{stocks[selected_stock]} ({selected_stock}) - 60-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig2)

# --- 90 Days ---
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(data.index[-200:], data['Close'][-200:], label="Historical Data")
ax3.plot(future_dates, predicted_prices, 'o-', color='magenta', label="Next 90 Days")
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
plt.xticks(rotation=45)
plt.title(f"{stocks[selected_stock]} ({selected_stock}) - 90-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig3)

st.write("### Predicted Prices for All 90 Days")
forecast_table = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predicted_prices
})
st.dataframe(forecast_table)
