import numpy as np

# Singular Value Decomposition

# Matrix A
A = np.array([
  [-1, 2],
  [3, -2],
  [5, 7]
])

U, d, VT = np.linalg.svd(A) # V is already transposed

print("\nU")
print(U)

print("\nD")
print(np.diag(d))

print("\nVT")
print(VT)