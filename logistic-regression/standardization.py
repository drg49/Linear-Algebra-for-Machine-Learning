import numpy as np

# X standardization = helps the model learn efficiently

# --------------------------------------------------
# REAL WORLD TITANIC DATASET (SMALL CLEAN SUBSET)
# --------------------------------------------------
# Columns:
# Pclass, Sex (0=female,1=male), Age, Fare, Survived

data = np.array([
    [3, 1, 22, 7.25, 0],
    [1, 0, 38, 71.28, 1],
    [3, 0, 26, 7.92, 1],
    [1, 0, 35, 53.10, 1],
    [3, 1, 35, 8.05, 0],
    [3, 1, 30, 8.46, 0],
    [2, 1, 34, 21.00, 0],
    [2, 0, 28, 26.00, 1],
    [3, 0, 2, 21.07, 1],
    [3, 1, 40, 15.50, 0],
    [1, 1, 58, 26.55, 1],
    [3, 0, 15, 7.22, 1],
    [3, 1, 28, 7.90, 0],
    [1, 1, 42, 26.55, 0],
    [3, 0, 19, 3.17, 1],
])

# Split into X and y
X = data[:, :-1]          # features
y = data[:, -1].reshape(-1, 1)  # labels (0/1)

print("Features (X):")
print(X)
# print("\nLabels (y):")
# print(y)

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
X = (X - X_mean) / X_std

print("\nXMean:")
print(X_mean)

print("\nX - XMean (Centered around 0):")
print(X - X_mean)

print("\nXStd:")
print(X_std)

print("\nStandardized Features (X - X_mean) / X_std:")
print(X)
