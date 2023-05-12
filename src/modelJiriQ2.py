# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:33:13 2023

@author: jiri zilka

Model for task Q1
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
#print(df.dtypes)
#print("ionization classes:",df.ionizationclass.unique())
#print("Flux Compensation:", df.FluxCompensation.unique())
df = prepareData(df)
#print(df.dtypes)
print(df['error_type'].unique())

#print(np.where(df['width'] == 10000000000))
#df.loc[df['width'] == 10000000000,'width'] = df.width.median()
#sns.boxplot(df['width'])

allColumns = ['id','width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation','weight_in_kg','weight_in_g','error','error_type',
 'Quality','reflectionScore','distortion','nicesness','multideminsionality']

inputColumns = ['width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation']
output = 'error_type'

df = df[df['width'] != 10000000000]
print(df['error_type'].describe())
#sns.boxplot(df['weight_in_kg'])
#print(df.describe())

X = df[inputColumns]
Y = df[output]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42,shuffle=True, stratify=Y)
#alternative:
#X_train, X_test, y_train, y_test = train_test_split(data[inputColumns], 
#                                   data.niceness,test_size=0.25,random_state=42)

print()

model = XGBClassifier() # Choose model

model.fit(X_train,y_train) # Train the model

pred = model.predict(X_test)

err = mean_absolute_error(y_test, pred)
print('model err', err)
err = mean_absolute_error(y_test, np.full(245,y_test.mean()))
print('baseline err', err)

plt.figure()
plt.scatter(y_test, pred)
plt.show()

print(y_test.value_counts())

plt.figure()
plt.scatter(y_test, np.full(245,y_test.mean()))
plt.show()

acc=sum(pred==y_test)/len(y_test)
print(f'accuracy = {acc}')
confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], \
                               colnames=['Predicted'])
print(confusion_matrix)

#fig = plt.figure(figsize=(30,10))
#tree.plot_tree(model,feature_names=X_train.columns,class_names=['no_churn', 'churn'],\
#               filled=True,)#fontsize=10
#plt.show()
#res = model.predict(X)

print("""
Second model for deciding the level of the error
""")

data_error = df[df['error'] == 1]

X = data_error[inputColumns]
Y = data_error[output] - 1 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42,shuffle=True, stratify=Y)
#alternative:
#X_train, X_test, y_train, y_test = train_test_split(data[inputColumns], 
#                                   data.niceness,test_size=0.25,random_state=42)

print('ytest counts\n',y_test.value_counts().head())
print('ytrain counts\n',y_train.value_counts().head())

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
