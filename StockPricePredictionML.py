# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Data Collection: Download historical stock data for Apple (AAPL)
data = yf.download('AAPL', start='2010-01-01', end='2020-01-01')

# 2. Feature Engineering: Creating 50-day and 200-day moving averages
data['50_MA'] = data['Close'].rolling(window=50).mean()  # 50-day moving average
data['200_MA'] = data['Close'].rolling(window=200).mean()  # 200-day moving average

# Drop rows with missing values (NaN) due to moving averages calculation
data = data.dropna()

# 3. Define Features and Target Variable
# Features: 50-day and 200-day moving averages
# Target: Close price
features = data[['50_MA', '200_MA']]
target = data['Close']

# 4. Train-Test Split: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 5. Model Selection: Using Linear Regression to predict stock prices
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Model Prediction: Predicting stock prices on the test set
predictions = model.predict(X_test)

# 7. Model Evaluation: Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(predictions - y_test))
print(f'Mean Absolute Error: {mae}')

# 8. Visualization: Plot the actual vs predicted prices
plt.figure(figsize=(10,6))
plt.plot(data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')
plt.plot(data.index[-len(y_test):], predictions, label='Predicted Prices', color='red')
plt.legend()
plt.show()
