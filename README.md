# Stock Price Prediction Using Machine Learning

## Overview
This project uses historical stock data to predict future stock prices of Apple (AAPL) using machine learning techniques. The model uses **Linear Regression** and **Moving Averages** as features to predict the closing stock price.

## Features
- Collects historical stock data from Yahoo Finance using the `yfinance` library.
- Calculates **50-day** and **200-day** moving averages as features.
- Implements **Linear Regression** to predict stock prices.
- Visualizes the predicted stock prices vs. actual prices on a plot.

## Requirements
To run this project, you need to install the following Python libraries:
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib
