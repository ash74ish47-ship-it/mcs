import numpy as np
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title("Monte Carlo Stock Price Simulator")

# 1. Setup
ticker = st.text_input(
    "Enter Stock Ticker Symbol (e.g., AAPL):", "AAPL").upper()

if st.button("Run Simulation"):
    try:
        # Download data
        data = yf.download(ticker, start='2020-01-01',
                           end='2026-03-31', progress=False)['Close']

        # Check if data is empty
        if data.empty:
            st.error(
                f"No data found for ticker: {ticker}. Please check the ticker symbol and try again.")
            # Exit if no data

        # Calculate statistical properties
        log_returns = np.log(1 + data.pct_change()).dropna()

        u = log_returns.mean().values if hasattr(
            log_returns.mean(), 'values') else log_returns.mean()
        var = log_returns.var().values if hasattr(
            log_returns.var(), 'values') else log_returns.var()
        stdev = log_returns.std().values if hasattr(
            log_returns.std(), 'values') else log_returns.std()

        # Drift calculation
        drift = u

        # 2. Monte Carlo Parameters
        t_intervals = 252
        iterations = 1000

        # 3. The Math Engine
        daily_returns = np.exp(
            drift + stdev * np.random.normal(0, 1, (t_intervals, iterations)))

        S0 = data.iloc[-1].item()

        price_list = np.zeros_like(daily_returns)
        price_list[0] = S0

        for t in range(1, t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        # 4. Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(price_list, color='cornflowerblue', alpha=0.02)
        ax.plot(price_list.mean(axis=1), color='red',
                linewidth=2, label='Mean Prediction')
        ax.set_title(
            f"Monte Carlo Simulation: 1000 Possible Futures for {ticker}")
        ax.set_xlabel("Days into Future")
        ax.set_ylabel("Price")
        ax.legend()

        st.pyplot(fig)

        # 5. Results
        expected_price = price_list[-1].mean()
        st.metric("Current Price", f"{S0:.2f}")
        st.metric("Expected Price in 1 Year", f"{expected_price:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
