#!/usr/bin/python
import numpy as np
from lr import LogisticRegressionClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *

data = load_iris()
X, y = data.data, data.target
X, y = X[y < 2], y[y < 2]       # remove class 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = LogisticRegressionClassifier()
clf.fit(X_train, y_train)

print 'Predicting a list of samples ...'
y_predict = clf.predict_proba(X_test)
print 'Result :', y_predict
print 'AUC-ROC :', roc_auc_score(y_test, y_predict)

print 'Estimated parameters:'
clf.print_params()
