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
 /tmp/ipykernel_508/1894848338.py:27: FutureWarning: adfuller currently returns a plain tuple whose length depends on the store and autolag arguments. In release 0.16 or after July 2027, whichever is later, the default behavior will switch to always returning an ADFullerResult. Set result_object=True to switch now, or result_object=False to keep the current behavior and silence this warning.
  result = adfuller(data)
ADF Statistic: -11.494310031913571
P-value: 4.654276370360886e-21
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   80
Model:                 ARIMA(1, 0, 0)   Log Likelihood                -277.211
Date:                Wed, 18 Nov 2026   AIC                            560.423
Time:                        13:15:00   BIC                            567.569
Sample:                             0   HQIC                           563.288
                                 - 80                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         29.2987     38.403      0.763      0.446     -45.970     104.567
ar.L1          0.9860      0.023     42.329      0.000       0.940       1.032
sigma2        57.2603      5.221     10.968      0.000      47.028      67.493
===================================================================================
Ljung-Box (L1) (Q):                   0.00   Jarque-Bera (JB):               223.99
Prob(Q):                              0.95   Prob(JB):                         0.00
Heteroskedasticity (H):               0.31   Skew:                            -1.80
Prob(H) (two-sided):                  0.00   Kurtosis:                        10.36
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
MAE: 4.914708682516378
MSE: 38.16296859775095
RMSE: 6.177618359671545



## Result
Thus, the ARIMA model was successfully implemented to forecast future time series values. Its prediction performance was evaluated using MAE, MSE and RMSE metrics. Lower error values indicate better forecasting accuracy.
