from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
iris = datasets.load_iris()

print(iris.feature_names)
print("\n")
# Show the first 5 rows (start from 0 to 5)
print(iris.data[0:5])

# Create a PCA object, specifying the number of components.
pca = PCA(n_components=2)

# Fit the PCA object to the iris data
X = pca.fit_transform(iris.data)

print("\n")
print('X shape:', X.shape) # (150, 2)

# Plot the first two principal components
_ = plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Iris Dataset: First Two Principal Components')
plt.show()