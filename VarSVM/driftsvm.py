"""
Linear SVM with fixed sample-adaptive intercepts
"""

# Author: Ben Dai <bdai@umn.edu>

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from VarSVM import CD_drift
from sklearn import preprocessing

class driftsvm(BaseEstimator, ClassifierMixin):
	"""
	Drifted Linear Support Vector Binary Classification. 

	$$ \sum_{i=1}^n w_i V(y_i(\beta^T x_i + drift_i)) + 1/2 \beta^T \beta $$

	Similar to linear SVM but with fixed intercepts. This class can be a sub-problem
	of many machine learning methods, such as semi-parametric classification,
	recommender systems, and multi-view classification.

    Parameters
    ----------
	def __init__(self, C=1., max_iter=1000, max_iter_dca=1000, verbose=0, tol=1e-4, loss='hinge'):
	self.dual_coef_ = []
	self.coef_ = []

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.

	loss : {'hinge', 't-hinge'}, default = 'hinge'
		Specifies the surrogate loss function in the objective function. The 'hinge'
		is the hinge loss used in SVM. The 't-hinge' is the truncated hinge loss.

    max_iter : int, default=1000
        The maximum number of iterations to be run for coordinate descent.

	max_iter_dca : int, default=1000
		The maximum number of iterations to be run for differenced convex algorithm (DCA).

	tol : float, default=1e-4
		Tolerance for stopping criteria.

    verbose : int, default=0
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in liblinear that, if enabled, may not work
        properly in a multithreaded context.

    Attributes
    ----------
    coef_ : ndarray of shape (1, n_features): Weights assigned to the features (coefficients in the primal
        problem).

	dual_coef_: ndarray of shape (1, n_sample): Weights assigned to the samples (coefficients in the dual
        problem).
    
	Examples
    --------
    >>> import numpy as np
	>>> from sklearn.datasets import make_classification
	>>> from varsvm import driftsvm
	>>> X, y = make_classification(n_features=4, random_state=0)
	>>> y = y * 2 - 1
	>>> drift = np.random.randn(len(X))
	>>> clf = noneg_driftsvm()
	>>> clf.fit(X=X, y=y, drift=drift)
	>>> y_pred = clf.decision_function(X=X, drift=drift)
    """

	def __init__(self, C=1., max_iter=1000, max_iter_dca=1000, verbose=1, tol=1e-4, loss='hinge'):
		self.loss = loss
		self.dual_coef_ = []
		self.coef_ = []
		self.C = C
		self.max_iter = max_iter
		self.max_iter_dca = max_iter_dca
		self.tol = tol
		self.verbose = verbose

	def get_params(self, deep=True):
		return {"C": self.C, "loss": self.loss, 'verbose': self.verbose}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def fit(self, X, y, drift=0., sample_weight=1.):
		"""Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X. y must be +1 or -1!
		drift : array-like of shape (n_samples,), default=0.0
            Array of drifts that are assigned to the decision function for each sample.
        sample_weight : array-like of shape (n_samples,), default=None
            Array of weights that are assigned to individual
            samples. If not provided,
            then each sample is given unit weight.
            .. versionadded:: 0.18
        Returns
        -------
        self : object
            An instance of the estimator.
        """
		X, y = check_X_y(X, y)
		if set(y) != {-1, 1}:
			raise NameError('y must be +1 or -1!')
		# le = preprocessing.LabelEncoder()
		# y = 2*le.fit_transform(y) - 1 
		n, d = X.shape
		self.dual_coef_ = np.zeros(n)
		diff = 1.
		sample_weight = self.C*np.array(sample_weight)
		sample_weight = sample_weight * np.ones(n)
		drift = y * drift
		## compute Xy matrix
		if sparse.issparse(X):
			Xy = sparse.csr_matrix(X.multiply(y.reshape(-1, 1)))
		else:
			Xy = X * y[:, np.newaxis]
		## compute diag vector
		if sparse.issparse(X):
			diag = np.array([Xy[i].dot(Xy[i].T).toarray()[0][0] for i in range(n)])
		else:
			diag = np.array([Xy[i].dot(Xy[i]) for i in range(n)])

		self.coef_ = Xy.T.dot(self.dual_coef_)
		# coordinate descent
		if sparse.issparse(Xy):
			if d > n:
				Q = Xy.dot(Xy.T)
				for ite in range(self.max_iter):
					if diff < self.tol:
						break
					alpha_old = np.copy(self.dual_coef_)
					for i in range(n):
						if diag[i] != 0:
							Grad_tmp = Q[i].dot(self.dual_coef_)[0]
							delta_tmp = (1. - drift[i] - Grad_tmp) / diag[i]
							delta_tmp = max(-self.dual_coef_[i], min(sample_weight[i] - self.dual_coef_[i], delta_tmp))
						if diag[i] == 0:
							if Grad_tmp < 1 - drift[i]:
								delta_tmp = sample_weight[i] - self.dual_coef_[i]
							else:
								delta_tmp = -self.dual_coef_[i]
						self.dual_coef_[i] = self.dual_coef_[i] + delta_tmp
					diff = np.sum(np.abs(alpha_old - self.dual_coef_))/np.sum(np.abs(alpha_old+1e-10))
					if self.verbose == 1:
						if ite > 0:
							print("ite %s coordinate descent with diff: %.3f;" %(ite, diff))
				self.coef_ = Xy.T.dot(self.dual_coef_)
			else:
				for ite in range(self.max_iter):
					if diff < self.eps:
						break
					beta_old = np.copy(self.coef_)
					for i in range(n):
						if diag[i] != 0:
							Grad_tmp = Xy[i].dot(self.coef_)[0]
							delta_tmp = (1. - drift[i] - Grad_tmp) / diag[i]
							delta_tmp = max(-self.dual_coef_[i], min(sample_weight[i] - self.dual_coef_[i], delta_tmp))
						if diag[i] == 0:
							if Grad_tmp < 1 - drift[i]:
								delta_tmp = sample_weight[i] - self.dual_coef_[i]
							else:
								delta_tmp = -self.dual_coef_[i]
						self.dual_coef_[i] = self.dual_coef_[i] + delta_tmp
						self.coef_ = np.array(self.coef_ + delta_tmp*Xy[i])[0]
					diff = np.sum(np.abs(beta_old - self.coef_))/np.sum(np.abs(beta_old+1e-10))
					if self.verbose == 1:
						if ite > 0:
							print("ite %s coordinate descent with diff: %.3f;" %(ite, diff))
		else:
			alpha_C, beta_C = CD_drift(Xy, diag, drift, self.dual_coef_, self.coef_, sample_weight, self.max_iter, self.eps, self.verbose)
			self.dual_coef_, self.coef_ = np.array(alpha_C), np.array(beta_C)
		
		if self.loss == 't-hinge':
			diff_dca = 1. 
			for ite_dca in range(self.max_iter_dca):
				if diff_dca < self.eps:
					break
				beta_old = np.copy(self.coef_)
				G = 1.*( (Xy.dot(self.coef_) + drift) < 0)
				self.coef_ = Xy.T.dot(self.dual_coef_ - G)
				if sparse.issparse(Xy):
					if d > n:
						Q = Xy.dot(Xy.T)
						for ite in range(self.max_iter):
							if diff < self.eps:
								break
							alpha_old = np.copy(self.dual_coef_)
							for i in range(n):
								if diag[i] != 0:
									Grad_tmp = Q[i].dot(self.dual_coef_)[0]
									delta_tmp = (1. - drift[i] - Grad_tmp) / diag[i]
									delta_tmp = max(-self.dual_coef_[i], min(sample_weight[i] - self.dual_coef_[i], delta_tmp))
								if diag[i] == 0:
									if Grad_tmp < 1 - drift[i]:
										delta_tmp = sample_weight[i] - self.dual_coef_[i]
									else:
										delta_tmp = -self.dual_coef_[i]
								self.dual_coef_[i] = self.dual_coef_[i] + delta_tmp
							diff = np.sum(np.abs(alpha_old - self.dual_coef_))/np.sum(np.abs(alpha_old+1e-10))
							if self.verbose == 1:
								if ite > 0:
									print("ite %s coordinate descent with diff: %.3f;" %(ite, diff))
						self.coef_ = Xy.T.dot(self.dual_coef_ - G)
					else:
						for ite in range(self.max_iter):
							if diff < self.eps:
								break
							beta_old = np.copy(self.coef_)
							for i in range(n):
								if diag[i] != 0:
									Grad_tmp = Xy[i].dot(self.coef_)[0]
									delta_tmp = (1. - drift[i] - Grad_tmp) / diag[i]
									delta_tmp = max(-self.dual_coef_[i], min(sample_weight[i] - self.dual_coef_[i], delta_tmp))
								if diag[i] == 0:
									if Grad_tmp < 1 - drift[i]:
										delta_tmp = sample_weight[i] - self.dual_coef_[i]
									else:
										delta_tmp = -self.dual_coef_[i]
								self.dual_coef_[i] = self.dual_coef_[i] + delta_tmp
								self.coef_ = np.array(self.coef_ + delta_tmp*Xy[i])[0]
							diff = np.sum(np.abs(beta_old - self.coef_))/np.sum(np.abs(beta_old+1e-10))
							if self.verbose == 1:
								if ite > 0:
									print("ite %s coordinate descent with diff: %.3f;" %(ite, diff))
				else:
					alpha_C, beta_C = CD_drift(Xy, diag, drift, self.dual_coef_, self.coef_, sample_weight, self.max_iter, self.eps, self.verbose)
					self.dual_coef_, self.coef_ = np.array(alpha_C), np.array(beta_C)
				diff_dca = np.sum(np.abs(self.coef_ - beta_old)) / (np.sum(np.abs(beta_old))+1e-10)
				obj_t_hinge = np.sum(np.minimum(np.maximum(1 - self.decision_function(X,drift)*y, 0),1)) + .5*self.coef_.dot(self.coef_)
				if self.verbose == 1:
					print("DCA fits t-hinge-loss with diff: %.3f; primal obj: %.3f" %(diff_dca, obj_t_hinge))

		# for ite in range(self.max_iter):
		# 	if diff < self.eps:
		# 		break
		# 	beta_old = np.copy(self.coef_)
		# 	for i in range(n):
		# 		if diag[i] != 0:
		# 			delta_tmp = (1. - drift[i] - np.dot(self.coef_, Xy[i])) / diag[i]
		# 			delta_tmp = max(-self.dual_coef_[i], min(sample_weight[i] - self.dual_coef_[i], delta_tmp))
		# 		if diag[i] == 0:
		# 			if np.dot(self.coef_, Xy[i]) < 1 - drift[i]:
		# 				delta_tmp = sample_weight[i] - self.dual_coef_[i]
		# 			else:
		# 				delta_tmp = -self.dual_coef_[i]
		# 		self.dual_coef_[i] = self.dual_coef_[i] + delta_tmp
		# 		self.coef_ = self.coef_ + delta_tmp*Xy[i]
		# 	obj = self.dual_obj(Xy=Xy, drift=drift)
		# 	diff = np.sum(np.abs(beta_old - self.coef_))/np.sum(np.abs(beta_old+1e-10))
		# 	if self.verbose:
		# 		if ite > 0:
		# 			print("ite %s coordinate descent with diff: %.3f; obj: %.3f" %(ite, diff, obj))

	def dual_obj(self, Xy, drift):
		## compute the dual objective function
		sum_tmp = np.dot(self.dual_coef_, Xy)
		return np.dot(1. - drift, self.dual_coef_) - .5 * np.dot(sum_tmp, sum_tmp)

	def decision_function(self, X, drift=0.0):
		"""Signed score to the samples.
		
		Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Testing vector, where n_samples in the number of samples and
            n_features is the number of features.
		drift : array-like of shape (n_samples,), default=0.0
            Array of drifts that are assigned to the decision function for each testing sample.

		Return
		------
		score : ndarray of shape (n_samples,)
			Returns the decision function of the samples.
		"""
		return np.dot(X, self.coef_) + drift

	def predict(self, X, drift=0.0):
		"""Signed label to the samples.
		
		Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Testing vector, where n_samples in the number of samples and
            n_features is the number of features.
		drift : array-like of shape (n_samples,), default=0.0
            Array of drifts that are assigned to the decision function for each testing sample.

		Return
		------
		label : ndarray of shape (n_samples,)
			Returns the labels of the samples.
		"""

		X = check_array(X)
		return np.sign(self.decision_function(X, drift))
