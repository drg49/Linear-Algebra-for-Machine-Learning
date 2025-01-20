import numpy as np
from matplot.plot_functions import plot_vectors

# Av = λv

# A is a square matrix we want to find eigenvectors and eigenvalues for this
A = np.array([[-1,  4],
              [ 2, -2]])

print('\nMatrix A:')
print(A)

lambdas, V = np.linalg.eig(A) 

# V has 2 columns, since A has two columns. Each column is an eigenvector.
print('\nV (Eigenvectors):')
print(V)

print('\nLamdas (Eigenvalues):')
print(lambdas)

# First eigenvector of V (lightblue)
v = V[:,0] # v

print('\nv:', v)

# Get the eigenvalue for the first eigenvector
lam = lambdas[0] # λ

print("\nλ:", lam)

# Av (blue)
Av = np.dot(A, v)
print('\nAv:', Av)

# λv (blue)
lam_v = np.dot(lam, v)
print('\nλv:', lam_v)

# Av = λv

# Plot the vector with a custom color
plot_vectors([Av, v], ['blue', 'lightblue'], (-3, 3), (-3, 3))