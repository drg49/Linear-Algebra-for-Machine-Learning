import numpy as np

# Calculate the Moore-Penrose Pseudoinverse of a matrix (the long way)

A = np.array([[-1,  2],
              [ 3, -2],
              [ 5,  7]])

U, d, VT = np.linalg.svd(A)

D = np.diag(d)
Dinv = np.linalg.inv(D)
# Dplus must have the same dimensions as A.T
Dplus = np.concatenate((Dinv, np.array([[0, 0]]).T), axis=1)

result = np.dot(VT.T, np.dot(Dplus, U.T))
print("\nResult")
print(result)

# Calculate the Moore-Penrose Pseudoinverse of a matrix (the short way)
Apinv = np.linalg.pinv(A)
print("\nResult (using numpy)")
print(Apinv)