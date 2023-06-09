# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

import matplotlib.pyplot as plt
import numpy as np
from dataPrep import *
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = loadData()
df = prepareData(df)

inputColumns = ['width']
output = 'reflectionScore'
X = df[inputColumns]
Y = df[output]

print(X.describe())
print()
print(Y.describe())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42)
plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, label="Training data")
plt.xlabel("width")
plt.ylabel("reflectionScore")
plt.title("Scatter plot of training data")

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

deg = 6
X_train_pol = X_train.values.reshape(1,-1)[0]
model_pol = np.poly1d(np.polyfit(X_train_pol,y_train, deg))

y_pred = model.predict(X_test)
y_pred_pol = model_pol(X_test)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

err = mean_absolute_error(y_test, np.full(y_test.shape[0],y_test.mean())) 
print('baseline err', err)

mae_pol = mean_absolute_error(y_test, y_pred_pol)
print('Mean Absolute Error for poly:', mae_pol)

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test)
plt.xlabel("width")
plt.ylabel("reflectionScore")
plt.title("Scatter plot of testing data")

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_pred)
plt.xlabel("width")
plt.ylabel("predicted reflectionScore")
plt.title("Scatter plot of LinearRegression model")

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_pred_pol)
plt.xlabel("width")
plt.ylabel("predicted reflectionScore")
plt.title("Scatter plot of Poly model")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual y-values")
plt.ylabel("Predicted y-values")
plt.title("Scatter plot of predicted vs actual y-values for LinearReg model")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_pol)
plt.xlabel("Actual y-values")
plt.ylabel("Predicted y-values")
plt.title("Scatter plot of predicted vs actual y-values for Poly Model")

plt.show()
