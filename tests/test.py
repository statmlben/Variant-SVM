import sys
sys.path.append('..')

import numpy as np
from sklearn.datasets import make_classification
from VarSVM import noneg_driftsvm
from sklearn.utils.validation import check_X_y, check_array

np.random.seed(0)
## generate sample
X, y = make_classification(n_features=4, random_state=0)
y = y * 2 - 1
n = len(X)
drift = .28*np.ones(n)

# test - fit function
clf = noneg_driftsvm()
clf.fit(X=X, y=y, drift=drift)

# test - decision function
clf.decision_function(X=X, drift=drift) * y

# test - predict function
clf.predict(X=X, drift=drift) * y

# test - nonnegative coefficients
clf.beta

# test - gridsearchCV
from sklearn.model_selection import GridSearchCV
clf = noneg_driftsvm()
C_grid = { 'C': [.1, .2, .3, .5] }
search = GridSearchCV(clf, C_grid, cv=5, scoring='accuracy')
search.fit(X, y, drift=0.)
search.cv_results_
