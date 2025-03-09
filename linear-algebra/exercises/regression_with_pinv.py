import numpy as np
import matplotlib.pyplot as plt

# Regression with the pseudo-inverse
# Psuedo-inverse is a generalization of the matrix inverse for non-square matrices

# Dosage of drug for treating Alzheimer's disease
x1 = [0, 1, 2, 3, 4, 5, 6, 7.]

# Create a y-intercept
x0 = np.ones(len(x1))

# The input matrix
X = np.matrix([x0, x1]).T

# Patient's "forgetfulness score" (the output)
y = [1.86, 1.31, .62, .33, .09, -.67, -1.23, -1.37]

# We can calculate the weights using the pseudo-inverse
w = np.dot(np.linalg.pinv(X), y)

# First weight is the y-intercept
b = np.asarray(w).reshape(-1)[0]

# Second weight is the slope
m = np.asarray(w).reshape(-1)[1]


title = 'Clinical Trial'
xlabel = 'Drug dosage (mL)'
ylabel = 'Forgetfulness'

fig, ax = plt.subplots()

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

ax.scatter(x1, y)

x_min, x_max = ax.get_xlim()

y_at_xmin = m * x_min + b
y_at_xmax = m * x_max + b

ax.set_xlim([x_min, x_max])
ax.plot([x_min, x_max], [y_at_xmin, y_at_xmax], c='C01')

plt.show()