import numpy as np

# Calculate the Moore-Penrose Pseudoinverse of a matrix

A = np.array([[-1,  2],
              [ 3, -2],
              [ 5,  7]])

U, d, VT = np.linalg.svd(A)

