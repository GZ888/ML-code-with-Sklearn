# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:44:43 2020

@author: gzhan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.api as sm
#X = np.append(arr = np.ones((50, 1)).astype(int), values=X, axis=1)

X_opt = sm.add_constant(X[:, [0,1,2,3,4]])
model = sm.OLS(y, X_opt)
regressor_OLS = model.fit()
print(regressor_OLS.summary())

X_opt = X_opt[:, [0,3,4,5]]
model = sm.OLS(y, X_opt)
regressor_OLS = model.fit()
print(regressor_OLS.summary())

X_opt = X_opt[:, [0,1,3]]
model = sm.OLS(y, X_opt)
regressor_OLS = model.fit()
print(regressor_OLS.summary())

X_opt = X_opt[:, [0,1]]
model = sm.OLS(y, X_opt)
regressor_OLS = model.fit()
print(regressor_OLS.summary())
