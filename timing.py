# This script constructs the Toeplitz matrix and computes its eigenvalues 
# using the Jacobi rotation algorithm, and the Numpy library function, 
# times them, and prints and plots their times for each step count

from numpy import array, linspace, zeros, matrix, sort, sin, cos, pi, transpose
from numpy import linalg as LA
from misc import tridiag_toeplitz, get_diags, min_diag
from jacobi import jacobi_rotalg, jacobi_rotalg_eig
import matplotlib.pyplot as plt
from matplotlib import rc
from time import perf_counter_ns

rc("text", usetex=True)

steps = array([10, 20, 50, 100])		# Array of step counts
times = zeros((2, len(steps)))			# Array of clocked times

for i in range(len(steps)):
	N = steps[i]
	print("Step count: " + str(N))	

	# Define the relevant Toeplitz matrix with (-1, 2, -1) along the 
	# tridiagonal
	r = linspace(0.0, 1.0, N + 2)[1:-1]
	h = 1.0/(N + 1.0)
	entries = array([-1.0, 2.0, -1.0])
	A = matrix(tridiag_toeplitz(N, entries))/h**2
	indices = linspace(1.0, 1.0*N, N)		# Array of indices

	# Run the two different diagonalization functions with 
	# a timer

	# Jacobi
	start = perf_counter_ns()/10**9
	D = jacobi_rotalg(A, err=10**(-5), max_iter=N**3)
	end = perf_counter_ns()/10**9
	times[0, i] = end - start
	print("	Jacobi: " + str(times[0, i]))

	# NumPy
	start = perf_counter_ns()/10**9
	w, v = LA.eig(A)
	end = perf_counter_ns()/10**9
	times[1, i] = end - start
	print("	NumPy: " + str(times[1, i]))

# Plot clocked times as dependent on step count
plt.plot(steps, times[0,:])
plt.plot(steps, times[1,:])
plt.title("Step count vs. Running time")
plt.legend(("Jacobi", "NumPy"), loc=2)
plt.xlabel("$N$")
plt.ylabel("$T$ (s)")
plt.show()