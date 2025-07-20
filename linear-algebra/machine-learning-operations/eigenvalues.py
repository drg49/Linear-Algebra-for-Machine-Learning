import numpy as np
import matplotlib.pyplot as plt

def plot_vectors(vectors, colors, xlim, ylim):
    plt.figure()
    plt.axvline(x=0, color='grey', lw=1)
    plt.axhline(y=0, color='grey', lw=1)
    for i in range(len(vectors)):
        plt.quiver(0, 0, vectors[i][0], vectors[i][1], angles='xy', scale_units='xy', scale=1, color=colors[i])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.show()


# Av = λv

# A is a square matrix we want to find eigenvectors and eigenvalues for this
A = np.array([[-1,  4],
              [ 2, -2]])

print('\nMatrix A:')
print(A)

lambdas, V = np.linalg.eig(A) 

# V has 2 columns, since A has two columns. Each column is an eigenvector of A.
print('\nV (Eigenvectors):')
print(V)

print('\nLamdas (Eigenvalues):')
print(lambdas)

# First eigenvector of V (lightblue)
v = V[:,0]

print('\n1st Eigenvector (v):')
print(v)

# Get the eigenvalue (λ) for the first eigenvector
lam1 = lambdas[0]

print("\n1st Eigenvalue (λ):")
print(lam1)

# Av (blue)
Av = np.dot(A, v)
print('\nAv:', Av)

# λv (blue)
lam_v = np.dot(lam1, v)
print('\nλv:', lam_v)

# Av = λv

# Plot the vector with a custom color
plot_vectors([Av, v], ['blue', 'lightblue'], (-3, 3), (-3, 3))