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





data_1 = pd.read_csv('BeijingPM20100101_20151231.csv')




print(data_1.head())




data_1.drop(['No','season', 'PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan',
            'DEWP', 'HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws', 'precipitation', 'Iprec'], axis=1, inplace=True)





print(data_1.head())





data_1['Time Stamp'] = data_1['hour'].astype(str) + ' ' + data_1['day'].astype(str) + ' ' + data_1['month'].astype(str)  + ' ' + data_1['year'].astype(str)





data_1.drop(['hour', 'month', 'day','year'], axis=1, inplace=True)




print(data_1.head())




from datetime import datetime

for i in range(49579):
    date_string = data_1['Time Stamp'][i]
    data_1['Time Stamp'][i] = datetime.strptime(date_string, '%H %d %m %Y')





data_2 = data_1



data_2.set_index(data_2['Time Stamp'], inplace=True)





data_2.drop(['Time Stamp'], axis=1, inplace=True)





data_2.to_excel('time_stamp_data.xlsx')





print(data_2.head())





data_1 = pd.read_csv('BeijingPM20100101_20151231.csv')





data_1['PM_US Post'].mean()




data_3 = data_2.fillna(data_1['PM_US Post'].mean())
print(data_3.head())





data_3.to_excel('time_stamp_mean_data.xlsx')

