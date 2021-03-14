import numpy as np

class Parsen:
    def __init__(self, kernel='rectangular', h=None, n_neighbours=3):
        """
        PARAMETERS:
        n_neighbours - number of nearest neighbors
        X_test - np.array of shape (m, d)
        kernel - kernel of parsen window
        """

        self.kernel = kernel
        self.n_neighbours = n_neighbours

        self.y_train = None
        self.y_count = []
        self.l_y = None
        self.density = []
        self.h = h
        self.length = []

    def kern(self, r):
        """
            INPUT:
            r - kernel parameter

            OUTPUT:
            w - kernel output
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

    def fit(self, X_train, y_train=None, X_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test

        self.l = len(X_train)

    def get_estimation(self, x_inp):
        """
            INPUT:
            x_inp - x axis

            OUTPUT:
            densities - estimation
        """
        densities = []
        if self.h is not None:
            v = 0
            for x in x_inp:
                v += self.kern(np.linalg.norm(x - self.X_train[0]) / self.h)
            for x in x_inp:
                p = 0
                for x_i in self.X_train:
                    p += self.kern(np.linalg.norm(x - x_i) / self.h)
                densities.append(p / (self.l * v))
        else:
            v = 0
            for x in x_inp:
                v += self.kern(np.linalg.norm(x - self.X_train[0]) / self.h)
            for x in x_inp:
                p = 0
                distances = [np.linalg.norm(x - x_i) for x_i in self.X_train]
                h = sorted(distances)[self.n_neighbours + 1]
                for x_i in self.X_train:
                    p += self.kern(np.linalg.norm(x - x_i) / h)
                densities.append(p / (self.l * v))
        return densities

    def get_estimation_by_class(self, x_inp, y):
        """
            INPUT:
            x_inp - x axis
            y - class
            OUTPUT:
            densities - estimation by class
        """
        densities = []
        if self.h is not None:
            for x in x_inp:
                p = 0
                n = 0
                for i in range(self.l):
                    if self.y_train[i] == y:
                        n += 1
                        p += self.kern(np.linalg.norm(x - self.X_train[i]) / self.h)
                densities.append(p)
        else:
            for x in x_inp:
                p = 0
                distances = [np.linalg.norm(x - x_i) for x_i in self.X_train]
                h = sorted(distances)[self.n_neighbours + 1]
                n = 0
                for i in range(self.l):
                    if self.y_train[i] == y:
                        n += 1
                        p += self.kern(np.linalg.norm(x - self.X_train[i]) / h)
                densities.append(p / n)
        return densities

    def get_estimation_by_classes(self):
        pre_predictions = []
        self.y_set = sorted(list(set(self.y_train)))
        for y in self.y_set:
            pre_predictions.append(self.get_estimation_by_class(self.X_test, y))
        return pre_predictions

    def predict(self):
        """
            INPUT:
            none

            OUTPUT:
            y_pred - np.array of shape (m,)
        """
        y_predictions = []
        pre_predictions = self.get_estimation_by_classes()
        for i in range(len(self.X_test)):
            density = [item[i] for item in pre_predictions]
            y_predictions.append(self.y_set[density.index(max(density))])
        return np.array(y_predictions)

    def clear(self):
        self.l = []
