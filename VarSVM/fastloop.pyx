import numpy as np
from libc.stdio cimport printf
cimport scipy.linalg.cython_blas as blas

cpdef run_scopy(double[:] x, double[:] y, int dim):
	cdef int inc = 1
	blas.dcopy(&dim, &x[0], &inc, &y[0], &inc)

cpdef run_daxpy(double a, double[:] x, double[:] y, int dim):
	cdef int inc = 1
	blas.daxpy(&dim, &a, &x[0], &inc, &y[0], &inc)

cpdef double[:] run_vec_minus(double[:] x, double[:] y, int dim):
	cdef double[:] tmp = np.zeros(dim)
	cdef int inc = 1
	cdef double pos = 1
	cdef double neg = -1
	blas.daxpy(&dim, &pos, &x[0], &inc, &tmp[0], &inc)
	blas.daxpy(&dim, &neg, &y[0], &inc, &tmp[0], &inc)
	return tmp

cpdef double l1_norm(double[:] x, int dim):
	cdef int inc = 1
	return blas.dasum(&dim, &x[0], &inc)

cpdef double run_blas_dot(double[:] x, double[:] y, int dim):
	cdef int inc = 1
	return blas.ddot(&dim, &x[0], &inc, &y[0], &inc)

def CD(double[:,:] Xy, double[:] diag, double[:] alpha, double[:] beta, double[:] sample_weight, int max_iter, double eps, int print_step):
	cdef int n = Xy.shape[0]
	cdef int d = Xy.shape[1]
	cdef int i
	cdef double diff = 1
	cdef double grad_tmp, alpha_tmp, delta_tmp
	cdef double[:] beta_old = np.ones(d)

	for ite in xrange(max_iter):
		if diff < eps:
			break
		run_scopy(beta, beta_old, d)
		for i in xrange(n):
			grad_tmp = run_blas_dot(beta, Xy[i], d)
			if diag[i] != 0:
				delta_tmp = (1. - grad_tmp) / diag[i]
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
	return alpha, beta



def CD_drift(double[:,:] Xy, double[:] diag, double[:] drift, double[:] alpha, double[:] beta, double[:] sample_weight, int max_iter, double eps, int print_step):
	cdef int n = Xy.shape[0]
	cdef int d = Xy.shape[1]
	cdef int i
	cdef double diff = 1
	cdef double grad_tmp, alpha_tmp, delta_tmp
	cdef double[:] beta_old = np.ones(d)

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
	return alpha, beta

def noneg_CD_drift(double[:,:] Xy, double[:] diag, double[:] drift, double[:] alpha, double[:] beta, double[:] rho, double[:] sample_weight, int max_iter, double eps, int print_step):
	cdef int n = Xy.shape[0]
	cdef int d = Xy.shape[1]
	cdef int i
	cdef double diff = 1
	cdef double grad_tmp, alpha_tmp, delta_tmp
	cdef double[:] beta_old = np.ones(d)

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
			printf('ite %d: coordinate descent with diff: %10.3f. \n', ite, diff)
		# printf('ite %d', ite,' coordinate descent with diff: %10.3f.', diff)
	return alpha, beta, rho
	
