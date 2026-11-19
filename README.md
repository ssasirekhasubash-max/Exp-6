# Experiment 6: ARIMA Model for Forecasting

## Aim

To evaluate predictive models using performance metrics.

## Algorithm

1.  **START**
2.  Prepare stationary time series.
3.  Identify ARIMA -- (p, d, q) parameters using ACF/PACF.
4.  Fit the ARIMA model on the data.
5.  Forecast future periods and plot the results.
6.  Evaluate forecast accuracy.
7.  **STOP**



## Program

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Create sample time series data
np.random.seed(42)

n = 100
noise = np.random.normal(0, 5, n)

sales = [100]

for i in range(1, n):
    sales.append(0.6 * sales[i-1] + noise[i])

data = pd.Series(sales)

# Check stationarity
result = adfuller(data)

print("ADF Statistic:", result[0])
print("P-value:", result[1])

# Plot ACF and PACF
plot_acf(data, lags=20)
plt.title("ACF Plot")
plt.show()

plot_pacf(data, lags=20)
plt.title("PACF Plot")
plt.show()

# Split data into training and testing
train = data[:80]
test = data[80:]

# Fit ARIMA model
model = ARIMA(train, order=(1, 0, 0))
model_fit = model.fit()

print(model_fit.summary())

# Forecast future values
forecast = model_fit.forecast(steps=len(test))

# Evaluate model performance
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Plot actual and forecast values
plt.figure(figsize=(10, 5))

plt.plot(test.index, test, label="Actual Values")
plt.plot(test.index, forecast, label="Forecast Values")

plt.title("ARIMA Time Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.show()

## Output
 <img width="857" height="602" alt="image" src="https://github.com/user-attachments/assets/803ec206-601a-499d-8dde-8c0334d96189" />
<img width="708" height="546" alt="image" src="https://github.com/user-attachments/assets/20973343-382b-422b-8ed4-a6477017b7f8" />
<img width="853" height="547" alt="image" src="https://github.com/user-attachments/assets/c8054246-b289-41ca-9420-e535eb290948" />
<img width="1066" height="582" alt="image" src="https://github.com/user-attachments/assets/8b5a3559-0722-4795-8105-d11eedafde87" />

 
## Result
Thus, the ARIMA model was successfully implemented to forecast future time series values. Its prediction performance was evaluated using MAE, MSE and RMSE metrics. Lower error values indicate better forecasting accuracy.
