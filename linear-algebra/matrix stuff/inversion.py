import torch

# Solving the equation: y = Xw

# `X` is a 3×3 matrix
# Each row represents a sample (ex: a single house)
# Each column represents a feature (ex: lot size, # of bedrooms, distance from school)
X = torch.tensor([
    [3, 4, 9], 
    [2, 3, 7], 
    [4, 4, 5]
], dtype=torch.float32) 

# This is the known output (ex: house prices)
y = torch.tensor([
  [51], 
  [61], 
  [80]
], dtype=torch.float32)

# Get inverse of `X`
Xinv = torch.inverse(X)

print("X Inverse:")
print(Xinv)
# `w` is the unknown weights and parameters we are solving for
#  Weights are numbers that tell the model how important each input feature is when making a prediction.
print("\nWeights:")
w = torch.matmul(Xinv, y)
print(w)

# Check work (features * weights = output)
y_check = torch.matmul(X, w)
print("\nOutput:")
print(y_check)