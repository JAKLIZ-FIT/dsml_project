# -*- coding: utf-8 -*-
"""
Created on Thu May 25 03:36:17 2023

@author: nguye
"""
import matplotlib.pyplot as plt
import numpy as np
from dataPrep import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


df = loadData()
df = prepareData(df)

# Examine data & scatter
examineFeatures = [['ionizationclass'], [
    'FluxCompensation'], ['pressure'], ['karma'], ['modulation']]
output = 'multideminsionality'
# X = df[examineFeatures]
Y = df[output]

multi_0 = df[df['multideminsionality'] == 0]
multi_1 = df[df['multideminsionality'] == 1]

xyz = [
    'width',
    'height',
    'ionizationclass',
    'FluxCompensation',
    'pressure',
    'karma',
    'modulation'
]

for xi in range(0, len(xyz) - 3):
    for yi in range(xi+1, len(xyz) - 2):
        for zi in range(yi+1, len(xyz) - 1):
            fig = plt.figure(figsize=(8, 15))
            ax = fig.add_subplot(111, projection='3d')
            x0 = multi_0[xyz[xi]]
            y0 = multi_0[xyz[yi]]
            z0 = multi_0[xyz[zi]]
            ax.scatter(x0, y0, z0, c=['red'])
            x1 = multi_1[xyz[xi]]
            y1 = multi_1[xyz[yi]]
            z1 = multi_1[xyz[zi]]
            ax.scatter(x1, y1, z1, c=['green'])
            ax.set_xlabel(xyz[xi])
            ax.set_ylabel(xyz[yi])
            ax.set_zlabel(xyz[zi])
            ax.set_title('Scatter Plot  ' +
                         xyz[xi] + '-' +  xyz[yi] + '-' +  xyz[zi] + '  ' +
                         str(xi) + '-' + str(yi) + '-' + str(zi)
                         )
            plt.show()

# for feat in examineFeatures:
#     plt.figure()
#     plt.scatter(multi_0[feat[0]], multi_0['multideminsionality'],
#                 color='blue', label='Multi 0')
#     plt.scatter(multi_1[feat[0]], multi_1['multideminsionality'],
#                 color='red', label='Multi 1')
#     plt.xlabel(feat[0])
#     plt.ylabel('multideminsionality')
#     plt.legend()

# print(X.describe())
# print()
# print(Y.describe())

# plt.figure()
# plt.scatter(X, Y, label="Scatter data")
# plt.xlabel("pressure")
# plt.ylabel("quality")
# plt.title("Scatter plot of data")

# # Prepare data for training
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25,
#                                                     random_state=56)

# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5,
#                                                 random_state=56)

# # Create model
# model = Sequential()
# model.add(Dense(128, activation="relu", input_dim=1))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(8, activation="relu"))
# # Since the regression is performed, a Dense layer containing a single neuron with a linear activation function.
# # Typically ReLu-based activation are used but since it is performed regression, it is needed a linear activation.
# model.add(Dense(1, activation="linear"))

# # Compile model: The model is initialized with the Adam optimizer and then it is compiled.
# model.compile(loss='mean_squared_error',
#               optimizer=Adam(lr=1e-3, decay=1e-3 / 200))

# # Patient early stopping
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# # Fit the model
# history = model.fit(X_train, y_train, validation_data=(
#     X_val, y_val), epochs=10000000, batch_size=100, verbose=2, callbacks=[es])

# # Calculate predictions
# y_pred = model.predict(X_test)
# y_val_pred = model.predict(X_val)

# # Evaluation
# # Assuming y_test is the actual target values for the test data
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error (MSE):", mse)

# plt.figure()
# # Plotting the actual values
# plt.scatter(X_test, y_test, color='blue', label='Actual Quality')
# # Plotting the predicted values
# plt.scatter(X_test, y_pred, color='red', label='Predicted Quality')
# plt.xlabel('Pressure')
# plt.ylabel('Quality')
# plt.title('Actual vs Predicted Quality')
# plt.legend()

# Show the plot
plt.show()
