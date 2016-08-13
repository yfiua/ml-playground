#!/usr/bin/python
import numpy as np

class KnnClassifier:
    def __init__(self, k=3):
        self.k = k
        return

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return

    def predict_single(self, x_test, k):
        dist = [ np.linalg.norm(x_train - x_test) for x_train in self.X_train ]
        knn = np.argsort(dist)[0:k-1]
        y_test = np.argmax(np.bincount(self.y_train[knn]))
        return y_test

    def predict(self, X_test, k=None):
        if k is None:
            k = self.k

        X_test = np.array(X_test)

        if len(X_test.shape) == 1:
            return self.predict_single(X_test, k)
        else:
            y_test = [ self.predict_single(x_test, k) for x_test in X_test ]
            return np.array(y_test)
