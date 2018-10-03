# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
dist = []
for i in range(len(som.distance_map().T) -1, -1, -1):
    dist.append(som.distance_map().T[i])
dist = np.reshape(dist,(len(som.distance_map().T),len(som.distance_map().T)))

from pylab import bone, pcolor, colorbar, show #, plot
bone()
pcolor(som.distance_map().T)
colorbar()
# markers = ['o', 's']
# colors = ['r', 'g']
"""
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2
         )
"""
show()
dist

# Finding the frauds
mappings = som.win_map(X)
#frauds = np.concatenate((mappings[(2,2)], mappings[(6,8)]), axis = 0)
frauds = mappings[(2,2)]
frauds = sc.inverse_transform(frauds)
