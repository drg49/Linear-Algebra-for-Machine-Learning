import torch

# Create a 3x3 identity matrix
identity_matrix = torch.eye(3, dtype=torch.int64)

# Convert A to a 2D tensor (3x1 matrix)
A = torch.tensor([25, 2, 5])

print("\nIdentity Matrix:")
print(identity_matrix)

print("\nA:")
print(A)

# Perform matrix multiplication
product = torch.matmul(identity_matrix, A)

print("\nProduct:")
print(product)