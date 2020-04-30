# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:03:20 2020

@author: gzhan
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1:].values

#fitting linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
res_poly = poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing the linear Regression line
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Linear Estimation of Salary on Experience Level")
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression line
plt.scatter(X, y, color = 'red')
#plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title("Polynomial Estimation of Salary on Experience Level")
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
#transform a vector to two dimensional array
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
#plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title("Polynomial Estimation of Salary on Experience Level")
plt.xlabel('Experience Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
#X_test = np.array(6.5).reshape(1,-1)
lin_reg.predict(np.array(6.5).reshape(1,-1))

lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1)))
