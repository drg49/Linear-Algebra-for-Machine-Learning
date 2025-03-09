import torch

N = torch.tensor([
  [1, 2, 4],
  [2, -1, 3],
  [0, 5, 1]
], dtype=torch.float32) # must use float not int

print("\nN", N)

print("det(N):", torch.det(N))

# Compute eigenvalues
eigenvalues = torch.linalg.eigvals(N)

# Compute the product of all eigenvalues
product = torch.prod(eigenvalues)

print("Eigenvalues:", eigenvalues)

print("Product of eigenvalues:", product)

## Conclusion: The determinant of a matrix N is equal to the product of all eigenvalues of N
