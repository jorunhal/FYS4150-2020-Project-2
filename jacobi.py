from numpy import zeros, linspace, array, copy, sqrt, matrix, identity
from misc import minabs, max_offdiag
from scipy.linalg import toeplitz



def jacobi_rotalg(M, err=10**(-5), max_iter=500):
	# Finds a rotation matrix to remove off-diagonal elements of a matrix M
	# until the off-diagonal norm is acceptably small, and returns a 
	# diagonalized matrix, below an error limit. M is assumed real and 
	# symmetric.

	n = M.shape[0]				# Assume M is a square matrix
	A = copy(M)				# Make a copy of argument matrix M
	count = 0				# Iteration counter
	while abs(A[max_offdiag(A)]) > err and count < max_iter:
		k, l, = max_offdiag(A)		# Find largest-norm off-diagonal 
						# element
		B = copy(A)			# Copy altered matrix

		# Compute trigonometric quantities
		tau = (A[l, l] - A[k, k])/(2.0*A[k, l])
		t = minabs(-tau + sqrt(1.0 + tau**2), -tau - sqrt(1.0 + tau**2))
		c = 1.0/sqrt(1.0 + t**2)
		s = t*c

		# Rotate matrix elements
		B[:, k] = c*A[:, k] - s*A[:, l]
		B[:, l] = s*A[:, k] + c*A[:, l]
		B[k, :] = B[:, k]
		B[l, :] = B[:, l]
		B[k, k] = A[k, k] - t*A[k, l]
		B[l, l] = A[l, l] + t*A[k, l]
		B[k, l] = 0.0
		B[l, k] = 0.0

		A = B				# Set equal to rotated matrix
		count = count + 1		# Increment iteration counter
	print("Step count: " + str(count))
	return matrix(A)

def jacobi_rotalg_eig(M, err=10**(-5), max_iter=500):
	# Finds a rotation matrix to remove off-diagonal elements of a matrix M
	# until the off-diagonal norm is acceptably small, and returns a 
	# diagonalized matrix, below an error limit, and a matrix V whose 
	# column vectors are the eigenvalues of M. M is assumed real and 
	# symmetric.

	n = M.shape[0]				# Assume M is a square matrix
	A = copy(M)				# Make a copy of argument matrix M
	U = identity(n)				# Matrix to hold the eigenvectors
						# of M

	# Run algorithm
	count = 0				# Iteration counter
	while abs(A[max_offdiag(A)]) > err and count < max_iter:
		k, l, = max_offdiag(A)		# Find largest-norm off-diagonal 
						# element
		B = copy(A)			# Copy altered matrix
		V = copy(U)			# Copy eigenvector matrix

		# Compute trigonometric quantities
		tau = (A[l, l] - A[k, k])/(2.0*A[k, l])
		t = minabs(-tau + sqrt(1.0 + tau**2), -tau - sqrt(1.0 + tau**2))
		c = 1.0/sqrt(1.0 + t**2)
		s = t*c

		# Rotate matrix elements
		B[:, k] = c*A[:, k] - s*A[:, l]
		B[:, l] = s*A[:, k] + c*A[:, l]
		B[k, :] = B[:, k]
		B[l, :] = B[:, l]
		B[k, k] = A[k, k] - t*A[k, l]
		B[l, l] = A[l, l] + t*A[k, l]
		B[k, l] = 0.0
		B[l, k] = 0.0

		# Rotate eigenvector matrix
		V[:, k] = c*U[:, k] - s*U[:, l]
		V[:, l] = c*U[:, l] + s*U[:, k]
		V[k, k] = c*U[k, k] - s*U[k, l]
		V[l, l] = c*U[l, l] + s*U[l, k]
		V[k, l] = c*U[k, l] + s*U[k, k]
		V[l, k] = c*U[l, k] - s*U[l, l] 

		A = B				# Set equal to rotated matrix
		U = V
		count = count + 1		# Increment iteration counter
	print("Step count: " + str(count))
	return matrix(A), matrix(U)
