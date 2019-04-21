import numpy as np
from libc.stdio cimport printf

cdef extern from 'cblas.h':
	ctypedef enum CBLAS_ORDER:
		CblasRowMajor
		CblasColMajor
	ctypedef enum CBLAS_TRANSPOSE:
		CblasNoTrans
		CblasTrans
		CblasConjTrans
	void dgemv 'cblas_dgemv'(CBLAS_ORDER order,
							CBLAS_TRANSPOSE transpose,
							int M, int N,
							double alpha, double* A, int lda,
							double* X, int incX,
							double beta, double* Y, int incY) nogil

cdef extern from 'cblas.h':
	double ddot 'cblas_ddot'(int N, double* X, int incX, double* Y, int incY) nogil
	void dscal 'cblas_dscal'(int N, double alpha, double* X, int incX) nogil
	void daxpy 'cblas_daxpy'(int N, double a, double* X, int incX, double* Y, int incY) nogil
	double dasum 'cblas_dasum'(int N, double* X, int incX) nogil
	void dcopy 'cblas_dcopy' (int N, double* X, int incX, double* Y, int incY) nogil

cpdef run_scopy(double[:] x, double[:] y, int dim):
	cdef double* x_ptr = &x[0]
	cdef double* y_ptr = &y[0]
	dcopy(dim, x_ptr, 1, y_ptr, 1)

cpdef run_daxpy(double a, double[:] x, double[:] y, int dim):
	cdef double* x_ptr = &x[0]
	cdef double* y_ptr = &y[0]
	daxpy(dim, a, x_ptr, 1, y_ptr, 1)

cpdef double[:] run_vec_minus(double[:] x, double[:] y, int dim):
	cdef double[:] tmp = np.zeros(dim)
	cdef double* x_ptr = &x[0]
	cdef double* y_ptr = &y[0]
	cdef double* tmp_ptr = &tmp[0]
	daxpy(dim, 1, x_ptr, 1, tmp_ptr, 1)
	daxpy(dim, -1, y_ptr, 1, tmp_ptr, 1)
	return tmp

cpdef double l1_norm(double[:] x, int dim):
	cdef double* x_ptr = &x[0]
	return dasum(dim, x_ptr, 1)

cpdef double run_blas_dot(double[:] x, double[:] y, int dim):

	# Get the pointers.
	cdef double* x_ptr = &x[0]
	cdef double* y_ptr = &y[0]

	return ddot(dim, x_ptr, 1, y_ptr, 1)

cpdef run_blas_dgemv(double[:,:] A,
					 double[:] x,
					 double[:] y,
					 int M,
					 int N):

	cdef double* A_ptr = &A[0,0]
	cdef double* x_ptr = &x[0]
	cdef double* y_ptr = &y[0]

	dgemv(CblasRowMajor,
		  CblasNoTrans,
		  M,
		  N,
		  1,
		  A_ptr,
		  N,
		  x_ptr,
		  1,
		  1,
		  y_ptr,
		  1)

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
	