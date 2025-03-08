from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Image Compression with Singular Value Decomposition

# Load the image and convert it to grayscale (to prevent complexity with colors)
img = Image.open("./images/dog.JPG").convert('LA')

imgmat = np.array(list(img.getdata(band=0)), float)

# img.size[1]: Height (number of rows).
# img.size[0]: Width (number of columns).
imgmat.shape = (img.size[1], img.size[0])

# Convert image data to a matrix
imgmat = np.matrix(imgmat)

print(imgmat)

# Get the SVD of the image
U, sigma, V = np.linalg.svd(imgmat)

# Reconstruct the image
new_img = np.matrix(U[:, :5]) * np.diag(sigma[:5]) * np.matrix(V[:5, :])

plt.imshow(new_img, cmap='gray')
plt.show()