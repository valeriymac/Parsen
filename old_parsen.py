import numpy as np
import pandas as pd
from collections import Counter
import math


class Parsen:
    def __init__(self, kernel='rectangular', h=None, n_neighbours=5):
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
        self.y_count = []
        self.l_y = None
        self.density = []
        self.h = h
        self.length = []
        if h is None:
            self.change_h = True
        else:
            self.change_h = False

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

    def fit(self, X_train, y_train, X_test):
        """
        INPUT:
        X_train - np.array of shape (l, d)
        y_train - np.array of shape (l,)

        OUTPUT:
        y_count - list of dictionaries with classes in keys
        and sum by class in values
        """
        min_X = min(X_train)
        max_X = max(X_train)
        self.X_train = (X_train - min_X)/(max_X - min_X)
        self.X_test = (X_test - min_X)/(max_X - min_X)
        self.y_train = y_train
        self.l_y = Counter(y_train)

        for i in self.X_test:
            neighbours = []
            sum_by_class = {'len': 0}
            for j in self.X_train:
                neighbours.append(math.sqrt(np.sum((j - i) ** 2)))
            if self.change_h:
                index = np.argsort(neighbours)[:self.n_neighbours + 1]
                self.h = neighbours[index[-1]] + 0.0000001
            else:
                index = np.argsort(neighbours)
            for t in index[:-1]:
                if neighbours[t] <= self.h:
                    sum_by_class['len'] += 1
                    if y_train[t] in sum_by_class:
                        sum_by_class[y_train[t]] += self.get_weight(neighbours[t] / self.h)
                    else:
                        sum_by_class[y_train[t]] = self.get_weight(neighbours[t] / self.h)
            self.y_count.append(sum_by_class)
            for obj in self.y_count:
                self.length.append(obj.pop('len', self.n_neighbours))
        return self.y_count

    def get_density(self):
        s = 0
        for n, obj in enumerate(self.y_count):
            s += sum(obj.values())
        for n, obj in enumerate(self.y_count):
            self.density.append(sum(obj.values()) / s ) #/ self.length[n])


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

    def clear(self):
        self.y_count = []
        self.density = []
        self.length = []
