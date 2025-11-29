import torch

# X: features
X = torch.tensor([
    [3, 4, 9],
    [2, 3, 7],
    [4, 4, 5]
], dtype=torch.float32)

# y: outputs, let's make one value wrong
y = torch.tensor([
    [51],
    [61],
    [80]  # wrong value
], dtype=torch.float32)

# Compute weights using normal equation: w = (X^T X)^-1 X^T y
Xt = X.T
w = torch.inverse(Xt @ X) @ Xt @ y

print("Weights:")
print(w)

# Check predictions
y_pred = X @ w
print("\nPredicted y:")
print(y_pred)

# Compare to original y
print("\nOriginal y:")
print(y)
