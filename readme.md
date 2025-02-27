# LSTM-Based Stock Price Prediction API

This repository contains a FastAPI-based application for predicting stock prices using an LSTM (Long Short-Term Memory) neural network. The project integrates machine learning techniques, data preprocessing, and API development to provide future stock price predictions.

## Features

- **Historical Stock Data Retrieval**: Fetches historical stock data for the last five years using `yfinance`.
- **LSTM Model Training**: Trains and optimizes an LSTM model to predict stock prices based on historical data.
- **Future Predictions**: Provides future stock price predictions for the next 30 days.
- **Performance Metrics**: Displays key metrics (MAE, MSE, RMSE, MAPE) for the best LSTM model.
- **API Interface**: Offers a REST API endpoint to request predictions for a specific stock ticker.

## Technologies Used

- **FastAPI**: For building the REST API.
- **TensorFlow/Keras**: For constructing and training the LSTM model.
- **yfinance**: For fetching historical stock market data.
- **scikit-learn**: For data preprocessing.
- **NumPy & Pandas**: For data manipulation and analysis.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Hansder01/LSTM-Based-Stock-Price-Prediction-API.git
   cd LSTM-Based-Stock-Price-Prediction-API
   ```

2. Create and activate a Python virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```

5. The API will be available at `http://127.0.0.1:8000`.

## API Usage

### Endpoint

- **POST** `/predict/`

### Sample Request Body
```json
{
  "ticker": "AAPL"
}
```

### Sample Response
```json
{
  "ticker": "AAPL",
  "historical_data": [...],
  "best_epoch": 30,
  "best_metrics": {
    "MAE": 1.23,
    "MSE": 2.34,
    "RMSE": 1.53,
    "MAPE": 0.45
  },
  "future_predictions": {
    "2025-01-07": 150.23,
    "2025-01-08": 151.45,
    ...
  }
}
```

## Project Structure

```
.
├── main.py               # Main application file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

## Key Functions

- **`train_lstm_model()`**: Trains the LSTM model with the given parameters.
- **`select_best_lstm_model()`**: Tunes the model using different epochs and selects the best one based on MSE.
- **`lstm()`**: Core function to prepare the data, train the model, and generate predictions.
- **FastAPI Endpoint**: `/predict/` for stock price predictions.

## Future Enhancements

- Add frontend visualization for historical and predicted stock data.
- Support for additional model hyperparameter tuning.
- Deployment to cloud platforms like AWS, GCP, or Azure.

## Contributing

Feel free to open issues or submit pull requests for improvements and new features.


Let me know if you'd like any adjustments!
