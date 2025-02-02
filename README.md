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
