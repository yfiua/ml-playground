#!/usr/bin/python
from __future__ import division

import numpy as np

class LogisticRegressionClassifier:
    def __init__(self):
        return

    def fit(self, X_train, y_train):
        n, m = X_train.shape        # size, dimension
        X_train = np.column_stack([X_train, np.ones(n)])

        # init params
        self.epsilon = 0.02      # learning rate
        self.beta = np.ones(m + 1)

        # iteration
        for i in range(100):
            # probability of y == 1
            p = 1 / (1 + np.exp(-np.dot(self.beta, X_train.T)))
            # log likelihood
            log_L = -np.sum(np.log1p(np.exp(-np.dot(self.beta, X_train.T)))) - np.dot(np.dot(self.beta, X_train.T), (1 - y_train))

            # update params
            self.beta += np.dot((y_train - p), X_train) * self.epsilon
            print log_L
        return

    def predict_proba(self, X_test):
        n, m = X_test.shape        # size, dimension
        X_test = np.column_stack([X_test, np.ones(n)])

        # probability of y == 1
        p = 1 / (1 + np.exp(-np.dot(self.beta, X_test.T)))
        return p

    def print_params(self):
        print self.beta
