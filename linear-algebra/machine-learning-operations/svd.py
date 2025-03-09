import numpy as np

# Singular Value Decomposition

# Matrix A
A = np.array([
  [-1, 2],
  [3, -2],
  [5, 7]
])

U, d, VT = np.linalg.svd(A) # V is already transposed

# The columns in U are the left singular vectors of A (eigenvectors of A * A.T)
print("\nU")
print(U)

print("\nD")
print(np.diag(d))

# The columns in VT are the right singular vectors of A (eigenvectors of A.T * A)
print("\nVT")
print(VT)