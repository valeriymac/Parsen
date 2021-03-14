import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from parsen import Parsen


class ParameterSelector:
    def __init__(self):
        self.neighbours = None
        self.rng = None
        self.kernel = None
        self.method = None

    def get_parameter(self, X, y, start, stop, step):
        sums = []
        self.rng = np.arange(start, stop, step)
        for k in self.rng:
            if self.neighbours:
                par = Parsen(self.kernel, n_neighbours=k)
            else:
                par = Parsen(self.kernel, h=k)
            sum = 0
            for train_index, test_index in self.method.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                par.fit(X_train, y_train, X_test)
                pre = par.predict()
                for n in range(len(pre)):
                    if pre[n] != y_test[n]:
                        sum += pre[n]
                par.clear()
            print(sum)
            sums.append(sum)
        return sums

    def get_parameter_ker(self, X, y, start, stop, step):
        sums = []
        kers = ['rectangular', 'triangular', 'epanechnikov', 'quartic', 'gaussian']
        self.rng = np.arange(start, stop, step)
        for k in kers:
            par = Parsen(kernel=k)
            sum = 0
            for train_index, test_index in self.method.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                par.fit(X_train, y_train, X_test)
                pre = par.predict()
                for n in range(len(pre)):
                    if pre[n] != y_test[n]:
                        sum += pre[n]
                par.clear()
            sums.append(sum)
        return sums


class LOO_search(ParameterSelector):
    def __init__(self, neighbours=False, kernel='epanechnikov'):
        self.method = LeaveOneOut()
        self.neighbours = neighbours
        self.kernel = kernel


class KFold_search(ParameterSelector):
    def __init__(self, n, neighbours=False, kernel='epanechnikov'):
        self.method = KFold(n_splits=n)
        self.neighbours = neighbours
        self.kernel = kernel
