# -*- coding: utf-8 -*-
"""StockPrediction_ARIMA.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1p1fIMbf-qsYfELoixhIIam-GurLzelGA

# Stock Price Prediction using ARIMA

This project aims to predict the stock price of **Tesla (TSLA)** using the **AutoRegressive Integrated Moving Average (ARIMA)** model, a statistical model for time-series forecasting.

## Features
- **Data Download**: Downloads historical stock data for Tesla from Yahoo Finance (yfinance).
- **Data Preprocessing**: Cleans the data and prepares it for training.
- **Model Training**: Trains the ARIMA model using the cleaned data.
- **Prediction and Forecasting**: Predicts future Tesla stock prices based on the trained ARIMA model.
- **Visualization**: Visualizes the actual vs predicted stock prices and forecasts for the next week.

## Requirements

To run this project, you will need the following Python packages:

- `yfinance` - For downloading stock data from Yahoo Finance.
- `numpy` - For numerical operations.
- `pandas` - For data manipulation and analysis.
- `matplotlib` - For plotting graphs.
- `statsmodels` - For ARIMA model training.
- `seaborn` (optional) - For enhanced plotting styles.

You can install all dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage
1. **Clone the Repository:**
```bash
git clone https://github.com/Poulami-Nandi/Stock_Prediction_ARIMA.git
cd Stock_Prediction_ARIMA
```
2. **Download and Preprocess Data:**
In the StockPrediction_ARIMA.ipynb, we download the historical stock data of Tesla for the last 5 years and clean the data by handling missing values and formatting the columns.

3. **Train the ARIMA Model:**
In train_arima_model.py, we train the ARIMA model on the stock data. The script downloads the data, processes it, and trains the ARIMA model to make predictions.
```bash
python train_arima_model.py
```

4. **Make Predictions:**
Once the ARIMA model is trained, you can use predict_stock_price.py to generate predictions for Tesla stock prices and compare them with actual values.
```bash
python predict_stock_price.py
```

5. **Visualize the Results:**
The notebook visualization.ipynb provides various visualizations:
* **Actual vs Predicted:** Comparing the actual stock price with the predicted values from ARIMA.
* **Forecasting:** Plotting the forecasted stock prices for the next week.
```bash
jupyter notebook visualization.ipynb
```
## Example Output
* **Predicted vs Actual Stock Prices:** A plot showing how well the ARIMA model predicted Tesla's stock price.
* **Stock Price Forecast:** Forecast of Tesla's stock price for the upcoming week.

## Directory Structure
```bash
Stock_Prediction_ARIMA/
│
├── data/                         # Contains raw and processed data
│   ├── tesla_stock_data_5y.csv   # Raw data of Tesla stock (from Yahoo Finance) for last 5 years
│   ├── tesla_stock_data_6m.csv   # Raw data of Tesla stock (from Yahoo Finance) for last 6 months
|   ├── tesla_next_week_pred.csv  # Prediction for next week
│
├── notebooks/                       # Jupyter notebooks for analysis and visualization
│   ├── StockPrediction_ARIMA.ipynb  # Data preprocessing and cleaning
│   ├── visualization.ipynb          # Visualizations of predicted and actual stock prices
│
├── scripts/                          # Python scripts for model training and prediction
│   ├── StockPrediction_ARIMA.py      # Script to train ARIMA model
│
├── requirements.txt              # List of required Python packages
├── README.md                     # Project overview and instructions
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"""

#@title Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

#@title Download historical stock Data of Tesla for last 5 years
import yfinance as yf

# Download historical data for Tesla (TSLA) for the last 5 years
tesla_data = yf.download('TSLA', period="5y", interval="1d")
tesla_data.to_csv('tesla_stock_data_5y.csv')

tesla_data.head()

#@title Download historical stock Data of Tesla for last 6 months
import yfinance as yf

# Download historical data for Tesla (TSLA) for the last 6 months
tesla_data = yf.download('TSLA', period="6mo", interval="1d")
tesla_data.to_csv('tesla_stock_data_6m.csv')

#@title Plot Daily Stock Price and Volume for last 6 months
import yfinance as yf
import pandas as pd

tickerStrings = ['TSLA']
df_list = []
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='6mo', interval='1d')
    df_list.append(data)

# Combine all dataframes into a single dataframe
df = pd.concat(df_list)
df.head()
df['TSLA'].head()
#for i in df.columns:
#  print(i)
df.columns = df.columns.get_level_values(1)
#df.info()
fig, ax = plt.subplots(figsize=(14, 7))  # Get the figure and axes objects
ax.plot(df.index, df['Close'], label='Tesla Daily Stock Price for last 6 months')
fig, ax = plt.subplots(figsize=(14, 7))  # Get the figure and axes objects
ax.plot(df.index, df['Volume'], label='Tesla Daily Stock Price for last 6 months')

#@title Plot Daily Stock Price and Volume for last 3 years
import yfinance as yf
import pandas as pd

tickerStrings = ['TSLA']
df_list = []
for ticker in tickerStrings:
    data = yf.download(ticker, group_by="Ticker", period='36mo', interval='1d')
    df_list.append(data)

# Combine all dataframes into a single dataframe
df = pd.concat(df_list)
df.head()
df['TSLA'].head()
#for i in df.columns:
#  print(i)
df.columns = df.columns.get_level_values(1)
#df.info()
fig, ax = plt.subplots(figsize=(14, 7))  # Get the figure and axes objects
ax.plot(df.index, df['Close'], label='Tesla Daily Stock Price for last 6 months')
fig, ax = plt.subplots(figsize=(14, 7))  # Get the figure and axes objects
ax.plot(df.index, df['Volume'], label='Tesla Daily Stock Price for last 6 months')

#@title YoY Growth for Tesla vs NASDAQ100
import numpy as np

# Download NASDAQ100 data for comparison
nasdaq_data = yf.download('^NDX', period="5y", interval="1d")
tesla_data = yf.download('TSLA', period="5y", interval="1d")
tesla_data.columns = tesla_data.columns.get_level_values(0)
nasdaq_data.columns = nasdaq_data.columns.get_level_values(0)

# Calculate YoY growth
tesla_data['YoY Growth'] = tesla_data['Close'].pct_change(252) * 100  # 252 trading days in a year
nasdaq_data['YoY Growth'] = nasdaq_data['Close'].pct_change(252) * 100

# Plot YoY growth comparison
plt.figure(figsize=(14, 7))
plt.plot(tesla_data.index, tesla_data['YoY Growth'], label='Tesla YoY Growth', color='blue')
plt.plot(nasdaq_data.index, nasdaq_data['YoY Growth'], label='NASDAQ100 YoY Growth', color='red')
plt.xlabel('Date')
plt.ylabel('Growth (%)')
plt.title('Tesla vs NASDAQ100 YoY Growth (Last 5 Years)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

#@title QoQ Growth for Tesla vs NASDAQ100
import numpy as np

# Download NASDAQ100 data for comparison
nasdaq_data = yf.download('^NDX', period="5y", interval="1d")
tesla_data = yf.download('TSLA', period="5y", interval="1d")
tesla_data.columns = tesla_data.columns.get_level_values(0)
nasdaq_data.columns = nasdaq_data.columns.get_level_values(0)
# Calculate QoQ growth
tesla_data['QoQ Growth'] = tesla_data['Close'].pct_change(63) * 100  # ~63 trading days per quarter
nasdaq_data['QoQ Growth'] = nasdaq_data['Close'].pct_change(63) * 100

# Plot QoQ growth comparison
plt.figure(figsize=(14, 7))
plt.plot(tesla_data.index, tesla_data['QoQ Growth'], label='Tesla QoQ Growth', color='green')
plt.plot(nasdaq_data.index, nasdaq_data['QoQ Growth'], label='NASDAQ100 QoQ Growth', color='orange')
plt.xlabel('Date')
plt.ylabel('Growth (%)')
plt.title('Tesla vs NASDAQ100 QoQ Growth (Last 5 Years)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

tesla_data.info()

# Download data
nasdaq_data = yf.download('^NDX', period="5y", interval="1d")
tesla_data = yf.download('TSLA', period="5y", interval="1d")
tesla_data.columns = tesla_data.columns.get_level_values(0)
nasdaq_data.columns = nasdaq_data.columns.get_level_values(0)

# --- Handle Missing Values ---
# Drop rows with any missing values in 'Close' column:
tesla_data.dropna(subset=['Close'], inplace=True)
# ARIMA model
arima_model = ARIMA(tesla_data['Close'], order=(5,1,0))  # p=5, d=1, q=0 (adjust parameters as needed)
arima_fitted = arima_model.fit()

#@title verification of ARIMA model
from sklearn.metrics import mean_squared_error
import numpy as np

arima_predictions = arima_fitted.predict(start=train_size, end=len(tesla_data)-1)
arima_rmse = np.sqrt(mean_squared_error(tesla_data['Close'][train_size:], arima_predictions))

print(f"ARIMA RMSE: {arima_rmse}")
print(f"valid_data:  {valid_data}")
print(f"arima_predictions:  {arima_predictions}")

#@title Predicted vs Actual in ARIMA model
import matplotlib.pyplot as plt

# Plotting predicted vs actual
plt.figure(figsize=(14, 7))

# Plot actual Tesla stock price
plt.plot(valid_data.index, valid_data['Close'], label='Actual Tesla Stock Price', color='blue')

# Plot ARIMA predictions (ensure the dates align)
plt.plot(valid_data.index, arima_predictions, label='ARIMA Predictions', color='green')

# If MCMC predictions are used, ensure alignment and uncomment the following line
# plt.plot(valid_data.index, mcmc_predictions, label='MCMC Predictions', color='orange')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Predicted vs Actual Stock Price for Tesla')

# Add legend
plt.legend()

# Display the plot
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()

model

#@title Forcast for coming week Feb3-7 using ARIMA
# Forecasting for next week using the models
forecast_dates = pd.date_range(start="2025-02-03", end="2025-02-07")

# ARIMA forecast
arima_forecast = arima_fitted.forecast(steps=5)

# MCMC forecast
#mcmc_forecast = np.mean(trace['mu'])

# Plot the forecasts
plt.figure(figsize=(14, 7))
#plt.plot(forecast_dates, lstm_forecast, label='LSTM Forecast')
plt.plot(forecast_dates, arima_forecast, label='ARIMA Forecast')
#plt.plot(forecast_dates, mcmc_forecast, label='MCMC Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price ($)')
plt.title('Tesla Stock Price Forecast for Feb 3–Feb 7, 2025')
plt.legend()
plt.show()
print(f"Forcast dates: \n{forecast_dates}")
print(f"ARIMA forcast for Feb2-7: \n{arima_forecast}")

#@title Store the next week prediction data
prediction_next_week = pd.DataFrame({'Date': forecast_dates, 'ARIMA_Forecast': arima_forecast})
prediction_next_week.to_csv('tesla_next_week_pred.csv', index=False)
prediction_next_week