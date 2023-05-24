# -*- coding: utf-8 -*-
"""
Created on Thu May 25 01:35:57 2023

@author: nguye
"""
import matplotlib.pyplot as plt
import numpy as np
from dataPrep import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
from keras.optimizers import Adam
import keras
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = loadData()
df = prepareData(df)

inputColumns = ['pressure']
output = 'Quality'
X = df[inputColumns]
Y = df[output]

# Examine data & scatter
print(X.describe())
print()
print(Y.describe())

plt.figure()
plt.scatter(X, Y, label="Scatter data")
plt.xlabel("pressure")
plt.ylabel("quality")
plt.title("Scatter plot of data")

# Prepare data for training
X_train, X_rest, y_train, y_rest = train_test_split(X, Y, test_size=0.25,
                                                    random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5,
                                                    random_state=42)

# Create model
model = Sequential()
model.add(Dense(32, activation="relu", input_dim=1))
model.add(Dense(8, activation="relu"))
# Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
model.add(Dense(1, activation="linear"))

# Compile model: The model is initialized with the Adam optimizer and then it is compiled.
model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10000, 
                    batch_size=100, verbose=2, callbacks=[es])

# Calculate predictions
y_pred = model.predict(X_test)

### Evaluation
from sklearn.metrics import mean_squared_error
# Assuming y_test is the actual target values for the test data
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

plt.figure()
# Plotting the actual values
plt.scatter(X_test, y_test, color='blue', label='Actual Quality')
# Plotting the predicted values
plt.scatter(X_test, y_pred, color='red', label='Predicted Quality')
plt.xlabel('Pressure')
plt.ylabel('Quality')
plt.title('Actual vs Predicted Quality')
plt.legend()

# Show the plot
plt.show()