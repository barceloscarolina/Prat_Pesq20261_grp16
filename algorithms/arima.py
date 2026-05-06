import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import numpy as np
import math
import sys, os
import datetime

#define hyperparameters
filename, p, d, q = 'trace60.csv', 2, 1, 2

path = 'http_requests_nasa/'
filename = path + filename

print(p, d, q)

# Read the csv file
dataset = pd.read_csv(filename, header=0, parse_dates=[0], index_col=0)

# Split into train and test sets
X = dataset.values
X = X.astype('float32')
size = int(len(X) * 0.67)
train, test = X[0:size], X[size:len(X)]
series = [x for x in train]
predictions = list()

training_size = len(train)

timer = start = datetime.datetime.now()

# Walk-forward validation
for t in range(len(test)):
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    output = model_fit.forecast()

    yhat = output[0]
    predictions.append(yhat)
    current = test[t]
    series.append(current)

# Evaluate forecasts
print("Timer: ", datetime.datetime.now() - timer)
score = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print('R2: %.2f, Testscore: %.2f MSE (%.2f RMSE)' % (r2, score, math.sqrt(score)))

# Plot the execution
pyplot.plot(test)
pyplot.plot(predictions, color='green')
pyplot.show()