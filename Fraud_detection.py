# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 23:35:21 2018

@author: Dhaval
"""

# Mega Case Study - Make a Hybrid Deep Learning Model

# Part -1 Identify the Self Organizing Map 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[: , -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the Self Orgainzing Map
from minisom import MiniSom
som = MiniSom(x = 10 , y = 10 , input_len = 15 , sigma = 1.0 , learning_rate = 0.5 )
som.random_weights_init(X)
som.train_random(X,num_iteration = 100 )

# Visualising the Result 
from pylab import bone , pcolor , show , plot , colorbar
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'b']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[Y[i]],
         markeredgecolor = colors[Y[i]],
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2)
show()

# Finding the Frauds 
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,6)], mappings[(2,1)],mappings[(5,2)]), axis = 0)
frauds = sc.inverse_transform(frauds)


# Part -2 Going From Unsupervised Learning to Supervised Learning

# Creating the Matrix of Features 
customers = dataset.iloc[: , 1:].values

# Creating the dependent variables 
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
 
# Creating ANN
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Part 3 - Making predictions and evaluating the model

# Predicting the Probability of Frauds
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:,0:1].values , y_pred) , axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]
