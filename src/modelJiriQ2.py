# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:33:13 2023

@author: jiri zilka

Model for task Q2
"""

from sklearn import metrics
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataPrep import loadData,prepareData
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error

df = loadData()

df = prepareData(df)

#print(df['error_type'].unique())
#print(df['error_type'].describe())
#print(df.describe())

allColumns = ['id','width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation','weight_in_kg','weight_in_g','error','error_type',
 'Quality','reflectionScore','distortion','nicesness','multideminsionality']

inputColumns = ['width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation']
output = 'error_type'

X = df[inputColumns]
Y = df[output]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42,shuffle=True, stratify=Y)

model = XGBClassifier() # Choose model

model.fit(X_train,y_train) # Train the model

pred = model.predict(X_test) # make prediction

err = mean_absolute_error(y_test, pred)
print('model err', err)
err = mean_absolute_error(y_test, np.full(245,y_test.mean()))
print('baseline err', err)

# plot Prediction to Ground truth
plt.figure(figsize=(8,5))
plt.scatter(y_test, pred)
plt.xlabel("Weight - Ground truth")
plt.ylabel("Weight - Prediction")
plt.title("Prediction vs reality")
plt.show()

print(y_test.value_counts())

plt.figure(figsize=(8,5))
plt.scatter(y_test, np.full(245,y_test.mean()))
plt.title("Baseline model")
plt.show()

acc=sum(pred==y_test)/len(y_test)
print(f'accuracy = {acc}')

# compute accuracy for detecting any error
predSimple = np.copy(pred)
y_testSimple = y_test.copy()
y_testSimple.loc[y_testSimple != 0] = 1
predSimple[predSimple != 0] = 1
accSimple=sum(predSimple==y_testSimple)/len(y_test)
print(f'simple accuracy = {accSimple}')

confusion_matrix_simple = pd.crosstab(y_testSimple, predSimple, rownames=['Actual'], \
                               colnames=['Pred'])
print(confusion_matrix_simple)

confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], \
                               colnames=['Pred'])
print(confusion_matrix)

print("""
Second model for deciding the level of the error
""")

data_error = df[df['error'] == 1]

X = data_error[inputColumns]
Y = data_error[output] - 1 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42,shuffle=True, stratify=Y)

#print('ytest counts\n',y_test.value_counts().head())
#print('ytrain counts\n',y_train.value_counts().head())

model = XGBClassifier() # Choose model

model.fit(X_train,y_train) # Train the model

pred = model.predict(X_test)

err = mean_absolute_error(y_test, pred)
print('model err', err)
err = mean_absolute_error(y_test, np.full(y_test.shape[0],2)) #y_test.mean()
print('baseline err', err)
print('-------------')

plt.figure()
plt.scatter(y_test, pred+np.random.rand(y_test.shape[0]))
plt.show()

plt.figure()
plt.scatter(y_test, np.full(y_test.shape[0],y_test.mean()))
plt.show()



acc=sum(pred==y_test)/len(y_test)
print(f'accuracy = {acc}')

confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], \
                               colnames=['Predicted'])
print(confusion_matrix)


