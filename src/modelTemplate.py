# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 16:21:26 2023

@author: jiri zilka

Template for model creation code
"""
from sklearn import metrics
from xgboost import XGBRegressor
from xgboost import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataPrep import loadData,prepareData

df = loadData()
df = prepareData(df)

inputColumns = ['']
output = 'niceness'

X = df[inputColumns]
Y = df[output]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42)
#alternative:
#X_train, X_test, y_train, y_test = train_test_split(data[inputColumns], 
#                                   data.niceness,test_size=0.25,random_state=42)

model = DecisionTreeClassifier(criterion='entropy') # Choose model

model.fit(X_train,y_train) # Train the model

pred = model.predict(X_test)

acc=sum(pred==y_test)/len(y_test)
print(f'accuracy = {acc}')
confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], \
                               colnames=['Predicted'])

print(confusion_matrix)

fig = plt.figure(figsize=(30,10))
tree.plot_tree(model,feature_names=X_train.columns,class_names=['no_churn', 'churn'],\
               filled=True,)#fontsize=10
plt.show()
res = model.predict(X)
