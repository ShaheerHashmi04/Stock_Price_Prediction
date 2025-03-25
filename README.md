Apple Stock Price Prediction
A simple stock price prediction model for Apple Inc. (AAPL) using 50-day and 200-day moving averages as features in a Linear Regression model. This project uses historical stock data from 2010 to 2020 and evaluates the model using Mean Absolute Error (MAE).

📈 Stock Price Prediction
This repository contains a stock price prediction model for Apple Inc. (AAPL) that uses two common technical indicators:

50-day Moving Average (50_MA): Short-term price trend

200-day Moving Average (200_MA): Long-term price trend

✅ What It Does:
Downloads historical data for Apple Inc. from yfinance.

Calculates the 50-day and 200-day moving averages.

Trains a Linear Regression model using these features to predict the Close price.

Evaluates model performance using Mean Absolute Error (MAE).

Visualizes the actual vs predicted stock prices.

⚙️ How It Works
Downloads historical daily stock data for AAPL (2010-2020).

Calculates the 50-day and 200-day moving averages.

Splits the data into training and testing sets.

Trains a Linear Regression model on the moving averages to predict the closing price.

Evaluates the model's performance using Mean Absolute Error (MAE).

Plot actual vs predicted stock prices for visualization.

The script will:

Output the Mean Absolute Error (MAE) of the model.

Plot the actual vs predicted stock prices.

📁 File
stock_price_prediction.py: Main script for training and evaluating the model.

📊 Requirements
Python 3.x

yfinance, pandas, numpy, sklearn, matplotlib

To install dependencies, use:

pip install yfinance pandas numpy scikit-learn matplotlib
⚠️ Disclaimer
This project is for educational purposes only and does not constitute financial advice or a recommendation to buy/sell securities. Backtesting results do not guarantee future performance.

📬 Contact
For questions or collaboration, feel free to open an issue or reach out.
