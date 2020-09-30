from numpy import linspace, array, zeros, matrix, sort, diag, linalg as LA
from misc import tridiag_toeplitz, get_diags, min_diag
from jacobi import jacobi_rotalg_eig
import matplotlib.pyplot as plt

N = 50
r_0, r_max = 0.0, 5.0
r = linspace(r_0, r_max, N + 2)[1:-1]
h = (r_max - r_0)/(N + 1)
indices = linspace(1.0, 1.0*N, N)

# Define Hamiltonian operator with no oscilattor term
H_0 = matrix(tridiag_toeplitz(N, array([-1.0, 2.0, -1.0]))/h**2 + diag(1.0/r))

w = array([0.01, 0.5, 1.0, 5.0])	# Array of different frequency values
P = zeros((len(w), N))			# Array of ground state eigenfunctions

# Loop through frequencies and find eigenvalues for each
for i in range(len(w)):
	w_r = w[i]

	# Add oscillator term to Hamiltonian
	H = H_0 + w_r**2*diag(r**2)

	# Find eigenvalues and eigenvectors
	D, U = jacobi_rotalg_eig(H, err=10**(-5), max_iter=N**3)
	eigs = get_diags(D)

	# Find the ground state and plot probability distribution
	m = min_diag(D)
	plt.plot(r, P[i, :])

plt.title("Ground state probability distributions$")
plt.xlabel("$\rho$")
plt.ylabel("$|\\psi|^2$")
plt.legend(("$\\omega\_r = " + str(w) + "$" for w_r in w))
plt.show()

