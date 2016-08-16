#!/usr/bin/python
import numpy as np

class LinearLeastSquaresRegressor:
    def __init__(self):
        return

    def fit(self, X_train, y_train):
        n = len(X_train)
        X_train = np.column_stack([X_train, np.ones(n)])
        self.a = np.dot(np.linalg.pinv(X_train), y_train)
        return

    def predict_single(self, x_test, k):
        dist = [ np.linalg.norm(x_train - x_test) for x_train in self.X_train ]
        knn = np.argsort(dist)[0:k-1]
        y_test = np.argmax(np.bincount(self.y_train[knn]))
        return y_test

    def predict(self, X_test):
        n = len(X_test.shape)
        X_test = np.hstack(X_test, np.ones(n).T)

        y_test = np.dot(X_test, self.a.T)
        return y_test

    def print_params(self):
        print self.a
