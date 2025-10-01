import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.layers import Input, LSTM, Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import streamlit as st


# Add theme selection dropdown in sidebar

theme = st.sidebar.selectbox(
    "Select Theme", 
    [
        "Dark",
        "Matrix Terminal", 
        "Retro Terminal", 
        "Glitch Vibe", 
        "Neon Code", 
        "Cyberpunk", 
        "CodeWave", 
        "Datanaut",
        "Monochrome Pro",
        "Futurist HUD" ,
        "Purple",
        "Blue"
    ]
)

# Load stock data
ticker = st.sidebar.selectbox(
    "Select Company",
    options=["MSFT", "AAPL", "META", "GOOGL", "AMZN"],
    format_func=lambda x: {"MSFT": "Microsoft", "AAPL": "Apple", "META": "Meta", "GOOGL": "Alphabet (Google)", "AMZN": "Amazon"}[x]
)

# Then replace the hard-coded download line:
data = yf.download(
    ticker,
    start='2015-01-01',
    end=pd.Timestamp.today().strftime('%Y-%m-%d')
)

# Theme styles mapped to user selection

theme_css = {
    "Dark": """
        <style>
        body, .stApp { background-color: #0e1117; color: #ffffff; }
        </style>
    """,
    "Matrix Terminal": """
        <style>
        body, .stApp {
            background-color: #0f0f0f;
            color: #39FF14;
        }
        </style>
    """,
    "Retro Terminal": """
        <style>
        body, .stApp {
            background-color: #1a1a1a;
            color: #ffcc00;
        }
        </style>
    """,
    "Glitch Vibe": """
        <style>
        body, .stApp {
            background-color: #121212;
            color: #f702ff;
        }
        </style>
    """,
    "Neon Code": """
        <style>
        body, .stApp {
            background-color: #0d0d0d;
            color: #00ffff;
        }
        </style>
    """,
    "Cyberpunk": """
        <style>
        body, .stApp {
            background-color: #1a001a;
            color: #ff00ff;
        }
        </style>
    """,
    "CodeWave": """
        <style>
        body, .stApp {
            background-color: #111827; color: #38bdf8;
        }
        </style>
    """,
    "Datanaut": """
        <style>
        body, .stApp {
            background-color: #090b10; color: #00c3ff;
        }
        </style>
    """,
    "Monochrome Pro": """
        <style>
        body, .stApp {
            background-color: #1c1c1c; color: #ffffff;
        }
        </style>
    """,
    "Futurist HUD": """
        <style>
        body, .stApp {
            background-color: #050505; color: #0ff0fc;
        }
        </style>
    """,
    "Purple": """
        <style>
        body, .stApp { background-color: #2b1d3a; color: #ffffff; }
        </style>
    """,
    "Blue": """
        <style>
        body, .stApp { background-color: #001f3f; color: #ffffff; }
        </style>
    """,
}

# Inject the selected theme
st.markdown(theme_css[theme], unsafe_allow_html=True)

# Sidebar UI

st.sidebar.header("Forecasting Panel")
model_choice = st.sidebar.radio("Select Model", ["-- Select --", "ARIMA", "SARIMA", "Prophet", "LSTM", "HWES"])

# Landing Page

if model_choice == "-- Select --":
    st.title(f"📈 Stock Price Analysis & Forecasting — {ticker}")
    st.markdown("""
                Welcome to the **Time Series Forecasting App** for Stock Market Analysis!

                This app leverages real {ticker} stock data and powerful forecasting models

                Use the sidebar to select a model and start forecasting like a data scientist! 🧠💹
                """)

# Show Historical Stock Chart
    st.subheader(f"Historical {ticker} Stock Price")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    ax_hist.plot(data.index, data['Close'], label="Close Price", color='black')
    ax_hist.set_title(f"{ticker} Closing Price")
    ax_hist.set_xlabel("Date", fontweight='bold')
    ax_hist.set_ylabel("Price", fontweight='bold')
    st.pyplot(fig_hist)


# ARIMA 
elif model_choice == "ARIMA": 

    st.subheader("AutoRegressive Integrated Moving Average (ARIMA) Forecast")
    st.write("By default forecasting is set to 15 days")

    cols = st.columns([1, 1, 1, 1])
    
    # Default selection
    if "forecast_days" not in st.session_state:
        st.session_state.forecast_days = 15
    
    with cols[0]:
        if st.button("7 Days", use_container_width=True):
            st.session_state.forecast_days = 7
    with cols[1]:
        if st.button("15 Days", use_container_width=True):
            st.session_state.forecast_days = 15
    with cols[2]:
        if st.button("1 Month", use_container_width=True):
            st.session_state.forecast_days = 30
    with cols[3]:
        if st.button("2 Months", use_container_width=True):
            st.session_state.forecast_days = 60

    forecast_days = st.session_state.forecast_days

    train = data['Close'][:-forecast_days]
    test = data['Close'][-forecast_days:]

    import warnings
    warnings.filterwarnings("ignore")

    # Fixed ARIMA order
    p, d, q = 2, 2, 0

    with st.spinner("Fitting ARIMA Model..."):
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        forecast_arima = model_fit.forecast(steps=forecast_days)
        rmse_arima = np.sqrt(mean_squared_error(test, forecast_arima))

        future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        context_days = 365
        fig_arima, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index[-context_days:], data['Close'][-context_days:], label="Actual", color='blue')
        ax.plot(future_index, forecast_arima, label="Forecast", color='red')
        ax.axvline(x=data.index[-1], color='gray', linestyle='--', label='Forecast Start')
        ax.set_title(f"ARIMA Forecast")
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_ylabel("Price", fontweight='bold')
        for label in ax.get_xticklabels():
            label.set_rotation(45)  # or 90 for vertical
            label.set_horizontalalignment('right')
        ax.legend()
        st.pyplot(fig_arima)

        st.metric("ARIMA RMSE", f"{rmse_arima:.2f}")

        df_arima_export = pd.DataFrame({
            "Date": future_index,
            "Model": "ARIMA",
            "Actual": [None] * forecast_days,
            "Forecast": forecast_arima.values
        })

        df_arima_export.to_csv("arima_forecast.csv", index=False)
        st.download_button(
            "Download Forecast CSV",
            data=df_arima_export.to_csv(index=False),
            file_name="arima_forecast.csv",
            mime="text/csv"
            
            )

        st.caption("ARIMA is a classical statistical model ideal for linear and stationary time series data.")
    
        arima_table = pd.DataFrame({
            "Forecast Date": future_index,
            "Forecasted Price": forecast_arima.values
        })
        arima_table.index = range(1, len(arima_table) + 1)
        st.dataframe(arima_table.style.format({"Forecasted Price": "${:,.2f}"}))


# SARIMA

elif model_choice == "SARIMA":

    st.subheader("Seasonal AutoRegressive Integrated Moving Average (SARIMA) Forecast")
    st.write("By default forecasting is set to 15 days")

    cols = st.columns([1, 1, 1, 1])

    # Default selection
    if "forecast_days" not in st.session_state:
        st.session_state.forecast_days = 15

    with cols[0]:
        if st.button("7 Days", use_container_width=True):
            st.session_state.forecast_days = 7
    with cols[1]:
        if st.button("15 Days", use_container_width=True):
            st.session_state.forecast_days = 15
    with cols[2]:
        if st.button("1 Month", use_container_width=True):
            st.session_state.forecast_days = 30
    with cols[3]:
        if st.button("2 Months", use_container_width=True):
            st.session_state.forecast_days = 60

    forecast_days = st.session_state.forecast_days

    train = data['Close'][:-forecast_days]
    test = data['Close'][-forecast_days:]

    import warnings
    warnings.filterwarnings("ignore")

    # Best SARIMA order from tuning (you can change if needed)
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 5)

    with st.spinner("Fitting SARIMA Model..."):
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        forecast_sarima = model_fit.forecast(steps=forecast_days)
        rmse_sarima = np.sqrt(mean_squared_error(test, forecast_sarima))

        future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
        context_days = 365
        fig_sarima, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data.index[-context_days:], data['Close'][-context_days:], label="Actual", color='blue')
        ax.plot(future_index, forecast_sarima, label="Forecast", color='orange')
        ax.axvline(x=data.index[-1], color='gray', linestyle='--', label='Forecast Start')
        ax.set_title(f"SARIMA Forecast")
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_ylabel("Price", fontweight='bold')
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        ax.legend()
        st.pyplot(fig_sarima)

        st.metric("SARIMA RMSE", f"{rmse_sarima:.2f}")

        df_sarima_export = pd.DataFrame({
            "Date": future_index,
            "Model": "SARIMA",
            "Actual": [None] * forecast_days,
            "Forecast": forecast_sarima.values
        })

        df_sarima_export.to_csv("sarima_forecast.csv", index=False)
        st.download_button(
            "Download Forecast CSV",
            data=df_sarima_export.to_csv(index=False),
            file_name="sarima_forecast.csv",
            mime="text/csv"
        )

        st.caption("SARIMA extends ARIMA by modeling seasonality, making it ideal for cyclical stock trends.")

        sarima_table = pd.DataFrame({
            "Forecast Date": future_index,
            "Forecasted Price": forecast_sarima.values
        })
        sarima_table.index = range(1, len(sarima_table) + 1)
        st.dataframe(sarima_table.style.format({"Forecasted Price": "${:,.2f}"}))

# Prophet
elif model_choice == "Prophet":

    st.subheader("Prophet Forecast")
    st.write("By default forecasting is set to 15 days")

    cols = st.columns([1, 1, 1, 1])

    if "forecast_days" not in st.session_state:
        st.session_state.forecast_days = 15

    with cols[0]:
        if st.button("7 Days", use_container_width=True):
            st.session_state.forecast_days = 7
    with cols[1]:
        if st.button("15 Days", use_container_width=True):
            st.session_state.forecast_days = 15
    with cols[2]:
        if st.button("1 Month", use_container_width=True):
            st.session_state.forecast_days = 30
    with cols[3]:
        if st.button("2 Months", use_container_width=True):
            st.session_state.forecast_days = 60

    forecast_days = st.session_state.forecast_days

    prophet_df = data.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']

    model_prophet = Prophet()
    model_prophet.fit(prophet_df)

    future = model_prophet.make_future_dataframe(periods=forecast_days)
    forecast_prophet = model_prophet.predict(future)

    # Define forecast_only from the last n days
    forecast_only = forecast_prophet[-forecast_days:]

    context_days = 365

    rmse_prophet = np.sqrt(mean_squared_error(data['Close'][-forecast_days:], forecast_only['yhat']))

    fig_prophet, ax = plt.subplots(figsize=(10, 4))
    ax.plot(prophet_df['ds'][-context_days:], prophet_df['y'][-context_days:], label="Actual", color='blue')
    ax.plot(forecast_only['ds'], forecast_only['yhat'], label="Forecast", color='green')
    ax.set_title("Prophet Forecast")
    ax.set_xlabel("Date", fontweight='bold')
    ax.set_ylabel("Price", fontweight='bold')
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    ax.legend()
    st.pyplot(fig_prophet)

    st.metric("Prophet RMSE", f"{rmse_prophet:.2f}")
    
    df_prophet_export = pd.DataFrame({
    "Date": forecast_only['ds'],
    "Model": "Prophet",
    "Actual": [None] * forecast_days,
    "Forecast": forecast_only['yhat']
    })

    df_prophet_export.to_csv("prophet_forecast.csv", index=False)

    st.download_button(
        "Download Forecast CSV",
        data=df_prophet_export.to_csv(index=False),
        file_name="prophet_forecast.csv",
        mime="text/csv"
    )

    st.caption("Prophet is Facebook’s open-source tool for handling trends, seasonality, and holidays in time series data.")
    
    prophet_table = pd.DataFrame({
        "Forecast Date": forecast_only['ds'],
        "Forecasted Price": forecast_only['yhat']
    })
    prophet_table.index = range(1, len(prophet_table) + 1)
    st.dataframe(prophet_table.style.format({"Forecasted Price": "${:,.2f}"}))

# LSTM

elif model_choice == "LSTM":
    st.subheader("Long Short-Term Memory (LSTM) Forecast")
    st.write("By default forecasting is set to 15 days")

    # Forecast button UI
    cols = st.columns([1, 1, 1, 1])
    if "forecast_days" not in st.session_state:
        st.session_state.forecast_days = 15

    with cols[0]:
        if st.button("7 Days", use_container_width=True):
            st.session_state.forecast_days = 7
    with cols[1]:
        if st.button("15 Days", use_container_width=True):
            st.session_state.forecast_days = 15
    with cols[2]:
        if st.button("1 Month", use_container_width=True):
            st.session_state.forecast_days = 30
    with cols[3]:
        if st.button("2 Months", use_container_width=True):
            st.session_state.forecast_days = 60

    forecast_days = st.session_state.forecast_days

    # Scaling data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])

    context_days = 60
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - context_days:]

    def create_sequences(dataset, step=context_days):
        X, Y = [], []
        for i in range(step, len(dataset)):
            X.append(dataset[i - step:i, 0])
            Y.append(dataset[i, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # LSTM model
    model_lstm = Sequential([
        Input(shape=(context_days, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # Prediction on test data
    predicted_scaled = model_lstm.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse_lstm = np.sqrt(mean_squared_error(actual[-forecast_days:], predicted[-forecast_days:]))

    # Forecast future values
    future_input = test_data[-context_days:].reshape(1, context_days, 1)
    future_forecast_scaled = []

    for _ in range(forecast_days):
        next_pred = model_lstm.predict(future_input)[0][0]
        future_forecast_scaled.append([next_pred])
        future_input = np.append(future_input[:, 1:, :], [[[next_pred]]], axis=1)

    future_forecast = scaler.inverse_transform(future_forecast_scaled)

    # Future dates for forecast
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    test_dates = data.index[-len(actual):]

    # Plot actual, prediction (test), and future forecast
    fig_lstm, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_dates, actual, label="Actual", color='blue')
    ax.plot(test_dates, predicted, label="Prediction", color='black')
    ax.plot(future_dates, future_forecast, label="Future Forecast", color='red')
    ax.set_title("LSTM Forecast")
    ax.set_xlabel("Date", fontweight='bold')
    ax.set_ylabel("Price", fontweight='bold')
    for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
    ax.legend()
    st.pyplot(fig_lstm)

    st.metric("LSTM RMSE", f"{rmse_lstm:.2f}")

    df_lstm_export = pd.DataFrame({
        "Date": future_dates,
        "Model": "LSTM",
        "Actual": [None]*forecast_days,
        "Forecast": future_forecast.flatten()
    })
    df_lstm_export.to_csv("lstm_forecast.csv", index=False)
    st.download_button("Download Forecast CSV", data=df_lstm_export.to_csv(index=False), file_name="lstm_forecast.csv", mime="text/csv")

    st.caption("LSTM captures long-term dependencies in stock price trends using neural networks.")

    lstm_table = pd.DataFrame({
        "Forecast Date": future_dates,
        "Forecasted Price": future_forecast.flatten()
    })
    lstm_table.index = range(1, len(lstm_table) + 1)
    st.dataframe(lstm_table.style.format({"Forecasted Price": "${:,.2f}"}))


# HWES
elif model_choice == "HWES":
    st.subheader("Holt-Winters Exponential Smoothing (HWES) Forecast")
    st.write("By default forecasting is set to 15 days")

    # Forecast button UI
    cols = st.columns([1, 1, 1, 1])
    if "forecast_days" not in st.session_state:
        st.session_state.forecast_days = 15

    with cols[0]:
        if st.button("7 Days", use_container_width=True):
            st.session_state.forecast_days = 7
    with cols[1]:
        if st.button("15 Days", use_container_width=True):
            st.session_state.forecast_days = 15
    with cols[2]:
        if st.button("1 Month", use_container_width=True):
            st.session_state.forecast_days = 30
    with cols[3]:
        if st.button("2 Months", use_container_width=True):
            st.session_state.forecast_days = 60

    forecast_days = st.session_state.forecast_days

    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    train = data['Close'][:-forecast_days]
    test = data['Close'][-forecast_days:]

    # ✅ Moved use_boxcox into model initialization
    model_hwes = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="mul",
        seasonal_periods=21,
        use_boxcox=True  # Moved here
    ).fit(optimized=True, remove_bias=True)

    forecast_hwes = model_hwes.forecast(steps=forecast_days)

    rmse_hwes = np.sqrt(mean_squared_error(test, forecast_hwes))

    future_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    context_days = 365
    fig_hwes, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index[-context_days:], data['Close'][-context_days:], label="Actual", color='blue')
    ax.plot(future_index, forecast_hwes, label="Forecast", color='purple')
    ax.axvline(x=data.index[-1], color='gray', linestyle='--', label='Forecast Start')
    ax.set_title("HWES Forecast")
    ax.set_xlabel("Date", fontweight='bold')
    ax.set_ylabel("Price", fontweight='bold')
    for label in ax.get_xticklabels():
        label.set_rotation(45)  # or 90 for vertical
        label.set_horizontalalignment('right')

    ax.legend()
    st.pyplot(fig_hwes)
    
    st.metric("HWES RMSE", f"{rmse_hwes:.2f}")

    df_hwes_export = pd.DataFrame({
        "Date": future_index,
        "Model": ["HWES"] * forecast_days,
        "Actual": [None] * forecast_days,
        "Forecast": forecast_hwes.values
    })
    df_hwes_export.to_csv("hwes_forecast.csv", index=False)
    st.download_button("Download Forecast CSV", data=df_hwes_export.to_csv(index=False), file_name="hwes_forecast.csv", mime="text/csv")

    st.caption("HWES captures seasonality and trends in time series data using exponential smoothing.")

    hwes_table = pd.DataFrame({
        "Forecast Date": future_index,
        "Forecasted Price": forecast_hwes.values
    })
    hwes_table.index = range(1, len(hwes_table) + 1)  # To remove default date-based index
    st.dataframe(hwes_table.style.format({"Forecasted Price": "${:,.2f}"}))
