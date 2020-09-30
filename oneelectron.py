from numpy import linspace, array, matrix, sort, diag, linalg as LA
from misc import tridiag_toeplitz, get_diags
from jacobi import jacobi_rotalg
import matplotlib.pyplot as plt

N = 100
r_0, r_max = 0.0, 10.0
r = linspace(r_0, r_max, N + 2)[1:-1]
h = (r_max - r_0)/(N + 1)
indices = linspace(1.0, 1.0*N, N)

H = matrix(tridiag_toeplitz(N, array([-1.0, 2.0, -1.0]))/h**2 + diag(r**2))

D, U = jacobi_rotalg(H, err=10**(-5), max_iter=N**3)
eigs = get_diags(D)

plt.plot(indices, eigs)
plt.show()

