from plot_functions import plot_vectors
import matplotlib.pyplot as plt
import numpy as np

# [x, y]
v = np.array([3, 1])

# Flip v on the y-axis
F = np.array([[-1, 0], [0, 1]])
Fv = np.dot(F, v)


# Plot the vector with a custom color
plot_vectors([v, Fv], ['lightblue', 'blue'], (-5, 5), (0, 5))

