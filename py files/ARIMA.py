#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.datasets import load_files
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import csv





data_1 = pd.read_excel('time_stamp_mean_data.xlsx')




data_2 = data_1.set_index(data_1['Time Stamp'])





data_2.drop(['Time Stamp'], axis=1, inplace=True)





print(data_2.head())




# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([.5, .5, 2, 1]) # left, bottom, width, height (range 0 to 1)


plt.plot(data_1['PM_US Post']) # 'r' is the color red
plt.xlabel('Period')
plt.ylabel('PM_US Post')
#plt.title('String Title Here')
plt.show()





X = data_2['PM_US Post']




train , test = X.values[:40000], X.values[40000:]


# ## Model




from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error





model = ARIMA(train, order=(1,1,1))
model_fit = model.fit(max_iter=1000)
window = model_fit.k_ar
coef = model_fit.params





history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]
	yhat = coef[0]
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]
	obs = test[t]
	predictions.append(yhat)
	history.append(obs)





for i in range(10):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))





error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)



# Create Figure (empty canvas)
fig = plt.figure()

# Add set of axes to figure
axes = fig.add_axes([.5, .5, 2, 1]) # left, bottom, width, height (range 0 to 1)


plt.plot(test, color='red')
plt.plot(predictions, color='green')
plt.show()

























