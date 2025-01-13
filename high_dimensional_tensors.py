import numpy as np

# Batch of 2 grayscale images
tensor = np.array([
    [
        [[0], [128]],
        [[255], [64]]
    ],
    [
        [[32], [16]],
        [[8], [4]]
    ]
])

print("Tensor shape:", tensor.shape)
print(tensor)
