#!/usr/bin/python
from knn import KnnClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = KnnClassifier()
clf.fit(X_train, y_train)

print 'Predicting a single sample ...'
y_predict = clf.predict(X_test[0])
print 'Result :', y_predict

print 'Predicting a list of samples ...'
y_predict = clf.predict(X_test)
print 'Result :', y_predict
print 'Precision =', precision_score(y_test, y_predict, average='weighted')
print '   Recall =', recall_score(y_test, y_predict, average='weighted')
print '       F1 =', f1_score(y_test, y_predict, average='weighted')
