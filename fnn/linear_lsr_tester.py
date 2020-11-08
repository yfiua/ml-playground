#!/usr/bin/python
import numpy as np
from linear_lsr import LinearLeastSquaresRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *

# generate test data
a = np.array([ 3, 4, 8, 10 ])
n = 1000
x1 = np.random.rand(n) * 10
x2 = np.random.rand(n) * 10
x3 = np.random.rand(n) * 10
x0 = np.ones(n)
X = np.vstack([x1, x2, x3]).T
y = np.dot(np.column_stack([X, x0.T]), a.T)

rgs = LinearLeastSquaresRegressor()
rgs.fit(X, y)

print 'Original parameters:'
print a
# estimating (data without error)
print 'Estimated parameters (y without error):'
rgs.print_params()

# estimating (data with normally distributed error)
y += np.random.normal(0, 1, n)
rgs.fit(X, y)
print 'Estimated parameters (y with normally distributed error):'
rgs.print_params()
