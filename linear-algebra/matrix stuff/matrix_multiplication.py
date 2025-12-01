import torch

# `X` is a 3×3 matrix
# Each row represents a sample
# Each column represents a feature
X = torch.tensor([
    [3, 4, 7], 
    [5, 6, 8], 
    [7, 8, 9]
], dtype=torch.float32) 

# `w` is a 3×1 column vector, where each row corresponds to the weight for one feature
# Weights represent the importance of input features (`X`) to the output (`y`)
w = torch.tensor([
    [1], 
    [2],
    [4],
], dtype=torch.float32)

# Remember, for matrix multiplication to work, the number of columns in `X` must match the number of rows in `w`.

# The result is a 3×1 tensor representing the predicted values `y` for each of the 3 samples
y = torch.matmul(X, w)
print(y)
