# -*- coding: utf-8 -*-
"""
Created on Fri May 12 12:41:59 2023

@author: jiri zilka

Model for task Q1
"""

from sklearn import metrics
from xgboost import XGBRegressor
import xgboost as xgb

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataPrep import loadData,prepareData, normalizeData
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error

df = loadData()
#print(df.dtypes)

df = prepareData(df)
#print(df.dtypes)
#sns.boxplot(df['width'])
#sns.boxplot(df['weight_in_kg'])
#print(df['weight_in_kg'].describe())
#print(df.describe())

allColumns = ['id','width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation','weight_in_kg','weight_in_g','error','error_type',
 'Quality','reflectionScore','distortion','nicesness','multideminsionality']

inputColumns = ['width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation']
output = 'weight_in_kg'

usedColumns = inputColumns.append(output)

X = df[inputColumns]
Y = df[output]
U = df[['width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation','reflectionScore']]

from sklearn.preprocessing import normalize
X = pd.DataFrame(normalize(X,axis=0),columns=X.columns)
#print(X.describe())

pd.plotting.scatter_matrix(U,figsize=(8,8),grid=True, marker='o')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42)

model = XGBRegressor() # Choose model

model.fit(X_train,y_train) # Train the model

xgb.plot_importance(model) # show feature importance

pred = model.predict(X_test) # make a prediction

err = mean_absolute_error(y_test, pred)
print('XGB model error:', err)
err = mean_absolute_error(y_test, np.full(245,y_test.mean()))
print('Baseline error: ', err)

plt.figure(figsize=(8,5))
plt.scatter(y_test, pred)
plt.xlabel("Weight - Ground truth")
plt.ylabel("Weight - Prediction")
plt.title("Prediction vs reality")
plt.show()

"""
# plot baseline
plt.figure()
plt.scatter(y_test, np.full(245,y_test.mean()))
plt.show()
"""
