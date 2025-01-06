import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import tensorflow as tf
from keras.api.regularizers import l2
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to allow your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model for ticker
class TickerInput(BaseModel):
    ticker: str

# Train LSTM model
def train_lstm_model(model, x_train, scaled_train, epochs):
    model.fit(x_train, scaled_train, epochs=epochs, batch_size=16, verbose=0)
    return model

# Select best LSTM model with different epoch values
def select_best_lstm_model(x_train, scaled_train, x_test, test, scaler):
    epoch_values = [20, 30, 35, 40, 50]
    best_epoch = None
    best_mse = float('inf')
    best_metrics = {}
    best_model = None

    for epochs in epoch_values:
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1), kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model = train_lstm_model(model, x_train, scaled_train, epochs)
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        mae = np.mean(np.abs(test.values - predictions))
        mse = np.mean((test.values - predictions) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test.values - predictions) / test.values)) * 100

        if mse < best_mse:
            best_mse = mse
            best_epoch = epochs
            best_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}
            best_model = model

    return best_model, best_epoch, best_metrics

# Core LSTM function to handle stock data and predictions
def lstm(ticker):
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - timedelta(days=1825)).strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    data.dropna(inplace=True)

    data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    data.reset_index(drop=True, inplace=True)

    train = data[['Adj Close']][:-30]
    test = data[['Adj Close']][-30:]

    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)

    x_train = np.reshape(scaled_train, (scaled_train.shape[0], 1, 1))
    x_test = np.reshape(scaled_test, (scaled_test.shape[0], 1, 1))

    best_model, best_epoch, best_metrics = select_best_lstm_model(x_train, scaled_train, x_test, test, scaler)

    # Use the best model to predict future prices
    future_predictions = []
    last_30_days = scaled_test[-30:]  # Take the last 30 days of scaled data as input for future predictions

    # Loop to predict the next 30 days
    for _ in range(30):
        prediction = best_model.predict(np.reshape(last_30_days[-1], (1, 1, 1)))  # Predict based on the last value
        future_predictions.append(prediction[0, 0])  # Save the predicted value
        last_30_days = np.append(last_30_days, prediction).reshape(-1, 1)  # Update the input data for the next prediction

    # Inverse the scaling of predictions to get the original values
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)

    # Prepare future dates
    future_dates = pd.date_range(today + timedelta(1), periods=30).strftime("%Y-%m-%d").tolist()
    future_predictions = future_predictions.flatten().tolist()

    # Convert historical data (DataFrame) to JSON serializable format
    historical_data = data.to_dict(orient='records')


    return {
        "ticker": ticker,
        "historical_data": historical_data,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "future_predictions": dict(zip(future_dates, future_predictions))
    }

# Define the API endpoint
@app.post("/predict/")
async def predict_stock(ticker: TickerInput):
    result = lstm(ticker.ticker)
    return result

# Entry point for running FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)