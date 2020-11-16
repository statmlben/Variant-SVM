.. -*- mode: rst -*-

|PyPi|_ |License|_ |Python3| |downloads|_ |downloads_month|_ |DOI|_

.. |PyPi| image:: https://badge.fury.io/py/varsvm.svg
.. _PyPi: https://badge.fury.io/py/varsvm
.. |License| image:: https://img.shields.io/pypi/l/varsvm.svg
.. _License: https://img.shields.io/pypi/l/varsvm.svg

.. |Python3| image:: https://img.shields.io/badge/python-3-green.svg
.. |downloads| image:: https://pepy.tech/badge/varsvm
.. _downloads: https://pepy.tech/project/varsvm
.. |downloads_month| image:: https://pepy.tech/badge/varsvm/month
.. _downloads_month: https://pepy.tech/project/varsvm
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3830281.svg
.. _DOI: https://doi.org/10.5281/zenodo.3830281

Variant-SVMs
============

.. image:: ./logo/logo_header.png
   :align: center
   :width: 100

VarSVM is a Python scikit-learn estimators module for solving variants Support Vector Machines (SVM).

Website: https://variant-svm.readthedocs.io

This project was created by `Ben Dai <https://www.bendai.org/>`_. If there is any problem and suggestion please contact me via <bdai@umn.edu>.

Installation
------------

Dependencies
~~~~~~~~~~~~

Tab-Data requires:

- Python
- NumPy
- Pandas
- Sklearn

User installation
~~~~~~~~~~~~~~~~~

Install Variant-SVMs using ``pip`` ::

	pip install varsvm

or ::

	pip install git+https://github.com/statmlben/varsvm.git

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/statmlben/varsvm.git


Documentation
-------------

weightsvm
~~~~~~~~~
Classical weighted SVMs.

- class VarSVM.weightsvm(alpha=[], beta=[], C=1., max_iter = 1000, eps = 1e-4, print_step = 1)

	- Parameters:
		- **alpha**: Dual variable.
		- **beta**: Primal variable, or coefficients of the support vector in the decision function.
		- **C**: Penalty parameter C of the error term.
		- **max_iter**: Hard limit on iterations for coordinate descent.
		- **eps**: Tolerance for stopping criterion based on the relative l1 norm for difference of beta and beta_old.
		- **print_step**: If print the interations for coordinate descent, 1 indicates YES, 0 indicates NO.
	- Methods:
		- **decision_function(X)**: Evaluates the decision function for the samples in X.
			- X : array-like, shape (n_samples, n_features)
		- **fit(X, y, sample_weight=1.)**: Fit the SVM model.
			- X : {array-like, sparse matrix}, shape (n_samples, n_features)
			- y : array-like, shape (n_samples,) **NOTE: y must be +1 or -1!**
			- sample_weight : array-like, shape (n_samples,), weight for each sample.

Drift SVM
~~~~~~~~~
SVM with dift or fixed intercept for each instance.

- class VarSVM.driftsvm(alpha=[], beta=[], C=1., max_iter = 1000, eps = 1e-4, print_step = 1)

	- Parameters:
		- **alpha**: Dual variable.
		- **beta**: Primal variable, or coefficients of the support vector in the decision function.
		- **C**: Penalty parameter C of the error term.
		- **max_iter**: Hard limit on iterations for coordinate descent.
		- **eps**: Tolerance for stopping criterion based on the relative l1 norm for difference of beta and beta_old.
		- **print_step**: If print the interations for coordinate descent, 1 indicates YES, 0 indicates NO.
	- Methods:
		- **decision_function(X)**: Evaluates the decision function for the samples in X.
			- X : array-like, shape (n_samples, n_features)
		- **fit(X, y, drift, sample_weight=1.)**: Fit the SVM model.
			- X : {array-like, sparse matrix}, shape (n_samples, n_features)
			- y : array-like, shape (n_samples,). **NOTE: y must be +1 or -1!**
			- drift: array-like, shape (n_samples,), drift or fixed intercept for each instance, see `doc <./Variant-SVMs.pdf>`_.
			- sample_weight : array-like, shape (n_samples,), weight for each instance.

Non-negative Drift SVM
~~~~~~~~~~~~~~~~~~~~~~
SVM with non-negative constrains for coefficients.

- class VarSVM.noneg_driftsvm(alpha=[], beta=[], C=1., max_iter = 1000, eps = 1e-4, print_step = 1)

	- Parameters:
		- **alpha**: Dual variable.
		- **beta**: Primal variable, or coefficients of the support vector in the decision function.
		- **C**: Penalty parameter C of the error term.
		- **max_iter**: Hard limit on iterations for coordinate descent.
		- **eps**: Tolerance for stopping criterion based on the relative l1 norm for difference of beta and beta_old.
		- **print_step**: If print the interations for coordinate descent, 1 indicates YES, 0 indicates NO.
	- Methods:
		- **decision_function(X)**: Evaluates the decision function for the samples in X.
			- X : array-like, shape (n_samples, n_features)
		- **fit(X, y, drift, sample_weight=1.)**: Fit the SVM model.
			- X : {array-like, sparse matrix}, shape (n_samples, n_features)
			- y : array-like, shape (n_samples,). **NOTE: y must be +1 or -1!**
			- drift: array-like, shape (n_samples,), drift or fixed intercept for each instance, see `doc <./Variant-SVMs.pdf>`_.
			- sample_weight : array-like, shape (n_samples,), weight for each instance.

Example
~~~~~~~

.. code-block:: Python

   import numpy as np
   from sklearn.datasets import make_classification
   from VarSVM import noneg_driftsvm

   X, y = make_classification(n_features=4, random_state=0)
   y = y * 2 - 1

   n = len(X)
   drift = .28*np.ones(n)

   clf = noneg_driftsvm()
   clf.fit(X=X, y=y, drift=drift)
   y_pred = clf.decision_function(X=X, drift=drift)

