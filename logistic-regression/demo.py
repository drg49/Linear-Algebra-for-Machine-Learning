import numpy as np

# --------------------------------------------------
# REAL WORLD TITANIC DATASET (SMALL CLEAN SUBSET)
# --------------------------------------------------
# Columns:
# Pclass, Sex (0=female,1=male), Age, Fare, Survived (0=died, 1=survived)

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

# --------------------------------------------------
# Standardize features (important for gradient descent)
# --------------------------------------------------
# If we don't standardize, features with larger scales (e.g., Fare)
# can dominate the learning process, leading to poor convergence.
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0)
# When we do (X - X_mean), we are centering the data around 0
# Dividing by X_std scales each feature column to have std = 1
X = (X - X_mean) / X_std

# --------------------------------------------------
# Sigmoid function
# --------------------------------------------------
# Converts a number to a probability between 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --------------------------------------------------
# Binary cross entropy loss
# --------------------------------------------------
def loss_fn(y_true, y_pred):
    eps = 1e-9
    return -np.mean(
        y_true * np.log(y_pred + eps) +
        (1 - y_true) * np.log(1 - y_pred + eps)
    )

# --------------------------------------------------
# Print survival probabilities
# --------------------------------------------------
def print_survival_probs(probs):
    probs = probs.ravel()
    for index, p in enumerate(probs):
        status = "Survived" if p >= 0.5 else "Died"
        print(f"({index:02d}) {p:.3f} -> {status}")

# --------------------------------------------------
# Initialize weights
# --------------------------------------------------
n_features = X.shape[1]
w = np.zeros((n_features, 1))
b = 0.0

learning_rate = 0.1
epochs = 2000

print("Y:")
print(y)

# --------------------------------------------------
# Training loop (pure NumPy)
# --------------------------------------------------
for epoch in range(epochs):
    # forward pass: the process of making predictions
    z = X @ w + b
        
    y_pred = sigmoid(z)

    if (epoch % 200) == 0:
        print(f"\n--- Epoch {epoch} ---")
        print(f"\nWeights: {w.ravel()} Bias: {b}")
        print(f"Linear combination (z):")
        print(z)
        print(f"Predictions:")
        print_survival_probs(y_pred)

    # loss
    loss = loss_fn(y, y_pred)

    # gradients
    dz = y_pred - y
    dw = (X.T @ dz) / len(X)
    db = np.mean(dz)

    w -= learning_rate * dw   # update weights
    b -= learning_rate * db   # update bias

    if epoch % 200 == 0:
        pred_labels = (y_pred >= 0.5).astype(int)
        acc = (pred_labels == y).mean()
        print(f"Loss={loss:.4f} Accuracy={acc:.3f}")

# --------------------------------------------------
# Final evaluation
# --------------------------------------------------
final_preds = (sigmoid(X @ w + b) >= 0.5).astype(int)
final_acc = (final_preds == y).mean()

print("\nTraining Complete.")
print("Final Accuracy:", final_acc)
print("Weights:", w.ravel())
print("Bias:", b)
