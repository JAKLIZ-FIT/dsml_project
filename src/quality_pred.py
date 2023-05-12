# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:34:26 2023

@author: Zalán
"""
import matplotlib.pyplot as plt
import numpy as np
from dataPrep import *
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = loadData()
df = prepareData(df)

inputColumns = ['pressure']
output = 'Quality'
X = df[inputColumns]
Y = df[output]

print(X.describe())
print()
print(Y.describe())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5,
                                                    random_state=42)
plt.figure()
plt.scatter(X_train, y_train, label="Training data")
plt.xlabel("x")
plt.ylabel("y")
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

err = mean_absolute_error(y_test, np.full(y_test.shape[0],y_test.mean())) #y_test.mean()
print('baseline err', err)

mae_pol = mean_absolute_error(y_test, y_pred_pol)
print('Mean Absolute Error for poly:', mae_pol)

#regression line
plt.figure()
plt.scatter(X_test, y_test, color="green", label="Test data")
plt.plot(X_test, y_pred, color="red", label="Regression line")

#Plot the residuals for the test data
residuals = y_test - y_pred
plt.figure()
plt.hist(residuals, bins=20, edgecolor='black')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of residuals for test data")

#QQ plot 
from statsmodels.graphics.gofplots import qqplot
plt.figure()
qqplot(residuals, line='s')
plt.title("QQ plot of residuals for test data")

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual y-values")
plt.ylabel("Predicted y-values")
plt.title("Scatter plot of predicted vs actual y-values for test data")

plt.figure()
plt.scatter(X_test, y_pred_pol)
plt.xlabel("X test values")
plt.ylabel("Predicted y-values from poly")
plt.title("Scatter plot for the poly")

plt.show()

