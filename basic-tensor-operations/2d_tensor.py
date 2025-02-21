import numpy as np

# Matrices in NumPy
# 2-Dimensional Tensors
# Denoted in uppercase (for example: "X")

X = np.array([
  [25 , 2], 
  [5, 26], 
  [3, 7]
])

print(X)

# print("Transpose the matrix:")
# print(X.T)

print("Shape:", X.shape) # (3, 2) or (3 rows, 2 columns)

# The size represents the total number of elements in the matrix
print("Size:", X.size) # 6 or (3 * 2)

# # Get the first row in the matrix
print("First row in the matrix:", X[0,:])

# # Get the second column in the matrix
# print("Second column in the matrix:", X[:,1])
