import numpy as np
import pandas as pd
from collections import Counter
import math
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


class Parsen:
    def __init__(self, X_test, kernel='rectangular', n_neighbours=5):
        """
        PARAMETERS:
        n_neighbours - number of nearest neighbors
        X_test - np.array of shape (m, d)
        kernel - kernel of parsen window
        """

        self.kernel = kernel
        self.n_neighbours = n_neighbours

        self.X_train = None
        self.y_train = None
        self.X_test = X_test
        self.y_count = []
        self.l_y = None
        self.density = []

    def get_weight(self, r):
        """
        INPUT:
        r - kernel parameter

        OUTPUT:
        w - weight for neighbour
        """

        w = 0
        if self.kernel == 'rectangular':
            w = (r <= 1) / 2
        if self.kernel == 'quartic':
            w = (r <= 1) * (1 - r ** 2) ** 2
            w *= 15 / 16
        if self.kernel == 'triangular':
            w = (r <= 1) * (1 - r)
        if self.kernel == 'epanechnikov':
            w = (r <= 1) * (1 - r ** 2) * 3 / 4
        if self.kernel == 'gaussian':
            w = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * r ** 2)
        return w

    def fit(self, X_train, y_train):
        """
        INPUT:
        X_train - np.array of shape (l, d)
        y_train - np.array of shape (l,)

        OUTPUT:
        y_count - list of dictionaries with classes in keys
        and sum by class in values
        """
        self.X_train = X_train
        self.y_train = y_train
        self.l_y = Counter(y_train)

        for i in self.X_test:
            neighbours = []
            sum_by_class = {}
            for j in self.X_train:
                neighbours.append(math.sqrt(np.sum((j - i) ** 2)))
            index = np.argsort(neighbours)[:self.n_neighbours + 1]
            h = neighbours[index[-1]]
            for t in index[:-1]:
                if y_train[t] in sum_by_class:
                    sum_by_class[y_train[t]] += self.get_weight(neighbours[t] / h)
                else:
                    sum_by_class[y_train[t]] = self.get_weight(neighbours[t] / h)
            self.y_count.append(sum_by_class)

        return self.y_count

    def get_density(self):
        for obj in self.y_count:
            self.density.append(sum(obj.values()) / self.n_neighbours)

    def predict(self):
        """
        INPUT:
        none

        OUTPUT:
        y_pred - np.array of shape (m,)
        """
        y_pred = []
        for dct in self.y_count:
            for key in dct:
                p_y = self.l_y[key] / self.y_train.shape[0]
                dct[key] = dct[key] * p_y / self.l_y[key]
            result = sorted(dct, key=lambda x: dct.get(x), reverse=True)[0]
            y_pred.append(result)
        return np.array(y_pred)



'''
data = pd.read_csv("weight-height.csv")
heights = data["Height"].to_numpy()
weights = data["Weight"].to_numpy()
'''

data = pd.read_csv("height_weight2.csv")
heights = data.iloc[:, 1].to_numpy() * 1.
weights = data.iloc[:, 2].to_numpy() * 1.


pred = {}
for test in range(120, 210):
    parsen = Parsen(np.array([float(test)]), kernel='epanechnikov', n_neighbours=500)
    parsen.fit(heights, weights)
    parsen.get_density()
    pred[test] = parsen.density
print(pred)
plt.plot(pred.keys(), pred.values())
plt.show()
