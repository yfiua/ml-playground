#!/usr/bin/python
import numpy as np

class LogsiticRegressionClassifier:
    def __init__(self):
        return

    def fit(self, X_train, y_train):
        n = len(X_train)
        X_train = np.column_stack([X_train, np.ones(n)])
        self.a = np.dot(np.linalg.pinv(X_train), y_train)
        return

    def print_params(self):
        print self.a
