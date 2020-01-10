import numpy as np
cimport cython
from libc.stdio cimport printf
cimport scipy.linalg.cython_blas as blas

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void run_scopy(double[::1] x, double[::1] y, int dim):
	cdef int inc = 1
	blas.dcopy(&dim, &x[0], &inc, &y[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void run_daxpy(double a, double[::1] x, double[::1] y, int dim):
	cdef int inc = 1
	blas.daxpy(&dim, &a, &x[0], &inc, &y[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double[::1] run_vec_minus(double[::1] x, double[::1] y, int dim):
	cdef int inc = 1
	cdef double[::1] tmp = np.zeros(dim)
	cdef double pos = 1
	cdef double neg = -1
	blas.daxpy(&dim, &pos, &x[0], &inc, &tmp[0], &inc)
	blas.daxpy(&dim, &neg, &y[0], &inc, &tmp[0], &inc)
	return tmp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double l1_norm(double[::1] x, int dim):
	cdef int inc = 1
	return blas.dasum(&dim, &x[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double run_blas_dot(double[::1] x, double[::1] y, int dim):
	cdef int inc = 1
	return blas.ddot(&dim, &x[0], &inc, &y[0], &inc)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def CD(double[:,::1] Xy, double[::1] diag, double[::1] alpha, double[::1] beta, double[::1] sample_weight, int max_iter, double eps, int print_step):
	cdef int n = Xy.shape[0]
	cdef int d = Xy.shape[1]
	cdef int i
	cdef double diff = 1
	cdef double grad_tmp, alpha_tmp, delta_tmp
	cdef double[::1] beta_old = np.ones(d)

	for ite in xrange(max_iter):
		if diff < eps:
			break
		run_scopy(beta, beta_old, d)
		for i in xrange(n):
			grad_tmp = run_blas_dot(beta, Xy[i], d)
			if diag[i] != 0:
				delta_tmp = (1 - grad_tmp) / diag[i]
				delta_tmp = max(-alpha[i], min(sample_weight[i] - alpha[i], delta_tmp))
			if diag[i] == 0:
				if grad_tmp < 1.:
					delta_tmp = sample_weight[i] - alpha[i]
				else:
					delta_tmp = -alpha[i]
			alpha[i] += delta_tmp
			# beta = beta + delta_tmp*Xy[i]
			run_daxpy(delta_tmp, Xy[i], beta, d)
		diff = l1_norm(run_vec_minus(beta, beta_old, d), d) / (l1_norm(beta_old, d) + 1e-10)
		if print_step==1:
			printf('ite %d: coordinate descent with diff: %10.3f. \n', ite, diff)
		# printf('ite %d', ite,' coordinate descent with diff: %10.3f.', diff)
	if ite == (max_iter-1):
		print('The algo did not convergence, pls increase max_iter')
	return alpha, beta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def CD_drift(double[:,::1] Xy, double[::1] diag, double[::1] drift, double[::1] alpha, double[::1] beta, double[::1] sample_weight, int max_iter, double eps, int print_step):
	cdef int n = Xy.shape[0]
	cdef int d = Xy.shape[1]
	cdef int i
	cdef double diff = 1
	cdef double grad_tmp, alpha_tmp, delta_tmp
	cdef double[::1] beta_old = np.ones(d)

	for ite in xrange(max_iter):
		if diff < eps:
			break
		run_scopy(beta, beta_old, d)
		for i in xrange(n):
			grad_tmp = run_blas_dot(beta, Xy[i], d)
			if diag[i] != 0:
				delta_tmp = (1. - drift[i] - grad_tmp) / diag[i]
				delta_tmp = max(-alpha[i], min(sample_weight[i] - alpha[i], delta_tmp))
			if diag[i] == 0:
				if grad_tmp < 1 - drift[i]:
					delta_tmp = sample_weight[i] - alpha[i]
				else:
					delta_tmp = -alpha[i]
			alpha[i] += delta_tmp
			# beta = beta + delta_tmp*Xy[i]
			run_daxpy(delta_tmp, Xy[i], beta, d)
		diff = l1_norm(run_vec_minus(beta, beta_old, d), d) / (l1_norm(beta_old, d) + 1e-10)
		if print_step==1:
			printf('ite %d: coordinate descent with diff: %10.3f. \n', ite, diff)
		# printf('ite %d', ite,' coordinate descent with diff: %10.3f.', diff)
	if ite == (max_iter-1):
		print('The algo did not convergence, pls increase max_iter')
	return alpha, beta

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def noneg_CD_drift(double[:,::1] Xy, double[::1] diag, double[::1] drift, double[::1] alpha, double[::1] beta, double[::1] rho, double[::1] sample_weight, int max_iter, double eps, int print_step):
	cdef int n = Xy.shape[0]
	cdef int d = Xy.shape[1]
	cdef int i
	cdef double diff = 1
	cdef double grad_tmp, alpha_tmp, delta_tmp
	cdef double[::1] beta_old = np.ones(d)

	for ite in xrange(max_iter):
		if diff < eps:
			break
		run_scopy(beta, beta_old, d)
		for i in xrange(n):
			grad_tmp = run_blas_dot(beta, Xy[i], d)
			if diag[i] != 0:
				delta_tmp = (1. - drift[i] - grad_tmp) / diag[i]
				delta_tmp = max(-alpha[i], min(sample_weight[i] - alpha[i], delta_tmp))
			if diag[i] == 0:
				if grad_tmp < 1 - drift[i]:
					delta_tmp = sample_weight[i] - alpha[i]
				else:
					delta_tmp = -alpha[i]
			alpha[i] += delta_tmp
			# beta = beta + delta_tmp*Xy[i]
			run_daxpy(delta_tmp, Xy[i], beta, d)
		for j in xrange(d):
			delta_tmp = max(-rho[j], -beta[j])
			rho[j] += delta_tmp
			beta[j] += delta_tmp
		diff = l1_norm(run_vec_minus(beta, beta_old, d), d) / (l1_norm(beta_old, d) + 1e-10)
		if print_step==1:
			printf('ite %d: coordinate descent with diff: %10.5f. \n', ite, diff)
		# printf('ite %d', ite,' coordinate descent with diff: %10.3f.', diff)
	if ite == (max_iter-1):
		print('The algo did not convergence, pls increase max_iter')
	return alpha, beta, rho
	
