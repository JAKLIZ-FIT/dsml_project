# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:34:26 2023

@author: Zalán
"""
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from dataPrep import *

df = loadData()
df = prepareData(df)

df = df[df['width'] != 10000000000]

inputColumns = ['karma','pressure','modulation']
output = ['distortion']

X = df[inputColumns]
Y = df[output]
"""
plt.figure()
dff = df[inputColumns + output]
pd.plotting.scatter_matrix(dff, alpha=0.2)
plt.show()
"""
print(X.describe())
print()
print(Y.describe())

X = normalizeData(X)
Y = normalizeData(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

model = xgb.XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

err = mean_absolute_error(y_test, np.full(y_test.shape[0],y_test.mean())) #y_test.mean()
print('baseline err', err)

print(mae/err)

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.show()

xgb.plot_importance(model)
plt.show()

plt.figure()
plt.scatter(X_train['karma'], y_train, label="Training data")
plt.xlabel("karma")
plt.ylabel("distortion")
plt.title("Scatter plot of karma and distortion")

plt.figure()
plt.scatter(X_train['pressure'], y_train, label="Training data")
plt.xlabel("pressure")
plt.ylabel("distortion")
plt.title("Scatter plot of pressure and distortion")




