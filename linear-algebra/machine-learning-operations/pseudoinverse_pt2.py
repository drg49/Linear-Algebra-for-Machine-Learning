import numpy as np
import matplotlib.pyplot as plt

# 🏡 Sample dataset: (Size in sq ft, Number of Bedrooms)
X = np.array([
    [1400, 3], 
    [1600, 3], 
    [1700, 3], 
    [1875, 4], 
    [1100, 2], 
    [1550, 3], 
    [2350, 4], 
    [2450, 4], 
    [1425, 3], 
    [1700, 3]
])

# House Prices ($1000s)
y = np.array([245, 312, 279, 308, 199, 219, 405, 324, 319, 255])

# Add a bias term (column of 1s) for the intercept
X_b = np.column_stack((np.ones(X.shape[0]), X))  # Shape (10,3)
print("X_b:\n", X_b)

# Compute the Moore-Penrose Pseudoinverse of X
X_pseudo = np.linalg.pinv(X_b)

# Compute optimal weights (w) using Normal Equation: w = (X^T X)^+ X^T y
w = X_pseudo @ y

print("Optimal weights (w):", w)  # Intercept, coefficient for size, coefficient for bedrooms

# 🎯 Predict the price of a new house (1800 sq ft, 3 bedrooms)
new_house = np.array([1, 1800, 3])  # Include bias term
predicted_price = new_house @ w
print(f"Predicted price for 1800 sq ft, 3-bedroom house: ${predicted_price * 1000:.2f}")

# 🔍 Plot: Relationship between House Size and Price
plt.scatter(X[:, 0], y, color='red', label='Actual Prices')
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price ($1000s)")

# Generate a range of house sizes for the best-fit line
size_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
predicted_prices = np.column_stack((np.ones(size_range.shape[0]), size_range, np.full(size_range.shape[0], 3))) @ w  # Assume 3 bedrooms

# Plot regression line for 3-bedroom houses
plt.plot(size_range, predicted_prices, color='blue', label="Regression Line (3-Bedroom Homes)")

plt.title("House Price Prediction using Moore-Penrose Pseudoinverse")
plt.legend()
plt.grid(True)
plt.show()
