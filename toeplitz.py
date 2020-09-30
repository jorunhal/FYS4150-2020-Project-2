from numpy import array, linspace, matrix, sort, sin, cos, pi, transpose
from numpy import linalg as LA
from misc import tridiag_toeplitz, get_diags, min_diag
from jacobi import jacobi_rotalg, jacobi_rotalg_eig
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex=True)


# Define the relevant Toeplitz matrix with (-1, 2, -1) along the 
# tridiagonal
N = 30
r = linspace(0.0, 1.0, N + 2)[1:-1]
h = 1.0/(N + 1.0)
entries = array([-1.0, 2.0, -1.0])
A = matrix(tridiag_toeplitz(N, entries))/h**2
indices = linspace(1.0, 1.0*N, N)			# Array of indices

# Run algorithm, store returned matrix and put eigenvalues in an array
D, U = jacobi_rotalg_eig(A, err=10**(-5), max_iter=N**3)
eigs = get_diags(D)

# Find eigenvalues with library function eig(), sort in descending order
w, v = LA.eig(A)

# Compute array of analytic solutions
w_a = sort((2.0 + 2*cos(indices*pi/(N + 1)))/h**2)

# Plot the two result spectra agianst each other in ascending order
plt.plot(indices, sort(eigs))
plt.plot(indices, sort(w))
plt.plot(indices, sort(w_a))
plt.title("Computed and ordered eigenvalues, $N = " + str(N) + "$")
plt.legend(("Jacobi", "NumPy", "Analytic"), loc=2)
plt.axis([indices[0] - 0.05*max(indices), indices[-1] + 0.05*max(indices), 
		min(w) - 0.05*max(w), 1.05*max(w)])
plt.xlabel("$n$")
plt.ylabel("$\lambda_n$")
#plt.show()

# Plot eigenvector of lowest eigenvalue
m = min_diag(D)
u_1 = U[:, m]
plt.plot(r, u_1)
plt.title("Eigenfunction $u_1(\\rho)$ of the lowest eigenvalue ($N = " 
		+ str(N) + "$)")
plt.xlabel("$\\rho$")
plt.ylabel("$u_1(\\rho)$")
plt.axis((r[0], r[-1], min(u_1) - 0.1*max(u_1), 1.1*max(u_1)))
plt.show()
