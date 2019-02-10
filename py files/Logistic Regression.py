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


# ## Loading Dataset 




dataset_beijing  = pd.read_csv('FiveCitiePMData\BeijingPM20100101_20151231.csv')





print(dataset_beijing.head())


# ## Data Cleaning 




dataset_beijing.drop(['PM_Dongsi', 'PM_Dongsihuan', 'PM_Nongzhanguan'],
                    axis=1,
                    inplace=True)


# Note for parameter:
# 
# axis : {0 or ‘index’, 1 or ‘columns’}, default 0
# Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
# 
# inplace : bool, default False
# If True, do operation inplace and return None.




dataset_beijing.dropna(axis=0, how='any', inplace=True)


# Note for parameter:
# 
# how : {‘any’, ‘all’}, default ‘any’
# Determine if row or column is removed from DataFrame, when we have at least one NA or all NA.
# 
# ‘any’ : If any NA values are present, drop that row or column.
# ‘all’ : If all values are NA, drop that row or column.




g = []





for i in range(49579):
    g.append(dataset_beijing.index.values[i])





labelEncoder = LabelEncoder()
dataset_beijing['cbwd'] = labelEncoder.fit_transform(dataset_beijing['cbwd'])


# LabelEncoder:
# Encode labels with value between 0 and n_classes-1.
# 
# Label Encoder vs. One Hot Encoder in Machine Learning:
# https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621

# ## Labels for pollution




dataset_beijing["Result"] = 0





dataset_beijing.head()





for i in range(49579):
    if dataset_beijing["PM_US Post"][g[i]] > 75:
        dataset_beijing["Result"][g[i]] = 1        





dataset_beijing2 = dataset_beijing





dataset_beijing2.to_excel('bejing_pollutionlabels.xlsx')



dataset_beijing2 = pd.read_excel('bejing_pollutionlabels.xlsx')


# ## Scaling and Splitting Data




y = dataset_beijing2['Result']
X = dataset_beijing2.drop('Result', axis=1)





standardScaler=StandardScaler()
X_scaled=standardScaler.fit_transform(X)


# Note:
# 
# Q)
#   Why do data scientists use Sklearn’s StandardScaler and what does it do?
# 
# A)  
#  It turns out that standardizing your data is much more than having the number of row items in your CSV equal the number of labels (headers) the CSV has. Standardization is a bit more than that, and as the documentation states, “it is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data…”.
# 
# This means that before you start training or predicting on your dataset, you first need to eliminate the “oddballs”. You need to remove values that aren’t centered around 0, because they might throw off the learning your algorithm is doing.
# 
# https://medium.com/@oprearocks/why-do-data-scientists-use-sklearns-standardscaler-and-what-does-it-do-9d93e248eb4




print("X_scaled.shape:",X_scaled.shape)
print("y.shape:",y.shape)





from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.20, random_state = 0)





print("X_train.shape:",X_train.shape)
print("y_train.shape:",y_train.shape)
print("X_test.shape:",X_test.shape)
print("y_test.shape",y_test.shape)


# ## Training 

# ### Logistic Regression




from sklearn.linear_model import LogisticRegression
model = LogisticRegression(verbose=10, max_iter=100)





model.fit(X_train, y_train)





print(model.score(X_test, y_test)_





