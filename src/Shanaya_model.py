# -*- coding: utf-8 -*-
from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from dataPrep import loadData,prepareData
from sklearn.metrics import mean_absolute_error

#data
data = loadData()
data = prepareData(data)

#correMatrix
le = LabelEncoder()
data['ionizationclas'] = le.fit_transform(data['ionizationclass'])
data['FluxCompensation'] = le.fit_transform(data['FluxCompensation'])
data['multideminsionality']= le.fit_transform(data['multideminsionality'])
data['error'] = le.fit_transform(data['error'])
data['error_type'] = le.fit_transform(data['error_type'])
corrMatrix = data.corr()
#print(corrMatrix)


import seaborn as sns
data = data[data['width']!= 10000000000]
#data.loc[data['width'] == 10000000000,'width'] = data.width.median()
sns.boxplot(data['width']) 
sns.boxplot(data['nicesness'])

# X are inputs and Y are Outputs
X =data[['width', "height"]]
Y = data[["nicesness"]]



X_train, X_test, y_train, y_test =  train_test_split(X,Y,test_size=0.25,random_state=42)

# Normalize input values
from sklearn.preprocessing import normalize
X = pd.DataFrame(normalize(X,axis=0),columns=X.columns)
Y = pd.DataFrame(normalize(Y,axis=0),columns=Y.columns)
#tipo de datos
print(data.dtypes)
 
#Train the model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#Make a prediction with model fit
nicesness_Predict = model.predict(X_test)

#Plot the model test
plt.figure()
plt.scatter(X_test.width,y_test, label = "Test data")
plt.title('Test data')
plt.xlabel("width")
plt.ylabel("nicesness")
plt.show()

#Plot predicted model with the regression line
plt.plot(X_test.width,nicesness_Predict, color = "green", label = "Regression Line")
plt.scatter(y_test, nicesness_Predict, color = "orange")
plt.title('Predict nicesness data with regression Line')
plt.xlabel("Width")
plt.ylabel("nicesness Predicted")
plt.show()

#Plot the actual vs the predicted
plt.figure()
plt.scatter(X_test.width,y_test, label = "Test data")
plt.scatter(y_test, nicesness_Predict)
plt.title('Actual niceness VS Predicted niceness')
plt.xlabel("width")
plt.ylabel("nicesness")
plt.show()


avg = Y.mean()
#np.full(980,avg)

print(mean_absolute_error(y_test,nicesness_Predict), "average error on all data")
print(mean_absolute_error(Y,np.full(980,avg)), "average error on base line data")
