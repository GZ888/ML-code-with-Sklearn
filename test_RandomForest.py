# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:38:20 2020

@author: gzhan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X, y)

#Predicting a new result
y_pred = regressor.predict(np.array(6.5).reshape(1,-1))
y_hat = regressor.predict(X)

#Visualizing the Decision Tree Regression results
X_grid = np.arange(min(X), max(X), 0.01)
#transform a vector to two dimensional array
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Regression Estimation of Salary on Experience Level")
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

