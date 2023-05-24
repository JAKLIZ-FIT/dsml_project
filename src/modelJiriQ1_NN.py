# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:09:24 2023

@author: jzilk
"""

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split
from dataPrep import loadData, prepareData
import pandas as pd

df = loadData()
#print(df.dtypes)
#print("ionization classes:",df.ionizationclass.unique())
#print("Flux Compensation:", df.FluxCompensation.unique())
df = prepareData(df)
#print(df.dtypes)




allColumns = ['id','width','height','ionizationclass','FluxCompensation','pressure',
 'karma','modulation','weight_in_kg','weight_in_g','error','error_type',
 'Quality','reflectionScore','distortion','nicesness','multideminsionality']

inputColumns = ['width','height','ionizationclass','FluxCompensation','pressure','karma','modulation']
output = 'weight_in_kg'

inCnt = len(inputColumns)

""" exploratory analysis"""
#import seaborn as sns
#sns.boxplot(df['width'])
#import numpy as np
#print("locs with high vals:", np.where(df['width'] == 10000000000))
#df = df[df['width'] != 10000000000]
#df.loc[df['width'] == 10000000000,'width'] = df.width.median()
#sns.boxplot(df['width'])

X = df[inputColumns]
Y = df[[output]]

print(X.describe())
print(Y.describe())

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

for c in inputColumns: 
    print(X[c].describe())

#from sklearn.preprocessing import normalize
#X = pd.DataFrame(normalize(X,axis=1),columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42)



#x_train = x_train.reshape(60000, 784)
#x_test = x_test.reshape(10000, 784)
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

X_train = X_train
X_test = X_test
y_train = y_train
y_test = y_test

model = Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(inCnt,)))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='mean_absolute_error',
              optimizer='sgd',
              metrics=['accuracy'])

batch_size = 32
epochs = 100

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])