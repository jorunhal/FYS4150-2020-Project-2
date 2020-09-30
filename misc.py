from numpy import zeros, array, matrix
from scipy.linalg import toeplitz

def minabs(x, y):
	# Returns the argument with the lowest norm
	if abs(x) > abs(y):
		return y
	else:
		return x

def max_offdiag(M):
	# Loops through the off-diagonal elements of a matrix M and returns 
	# the indices of the element with the largest norm
	m, n = M.shape
	k, l = 0, 1
	for i in range(m):
		for j in range(n):
			if not i == j:		# Rule out diagonal elements
				if abs(M[i, j]) > abs(M[k, l]):
					k, l = i, j
	return k, l

def min_diag(M):
	# Loops through the diagonal elements of a matrix M and returns the index 
	# of the element with the smallest norm
	k = 0
	for i in range(1, min(M.shape)):
		if abs(M[i, i]) < abs(M[k, k]):
			k = i
	return k

def norm_Frob(M):
	# Returns the Frobenius norm ||M||_F of an (m x n) matrix M
	total = 0.0
	m, n = M.shape
	# Loop through elements of M
	for i in range(m):
		for j in range(n):
			total = total + M[i, j]**2
	return sqrt(total)

def norm_offdiag(M):
	# Returns the off-diagonal norm off(M) of an (m x n) matrix M
	total = 0.0
	m, n = M.shape
	# Loop through elements of M
	for i in range(m):
		for j in range(n):
			if not i == j:		# Rule out diagonal elements
				total = total + M[i, j]**2
	return sqrt(total)

def get_diags(M):
	# Returns the diagonal elements of a matrix M
	m, n = M.shape
	elements = zeros(min(m, n))
	for i in range(min(m, n)):
		elements[i] = M[i, i]
	return elements

def tridiag_toeplitz(n, entries):
	# Returns a tridiagonal Toeplitz matrix with entries from argument array
	u, v = zeros((2, n))
	u[0] = entries[1]
	u[1] = entries[2]
	v[0] = entries[1]
	v[1] = entries[0]
	return toeplitz(u, v)
