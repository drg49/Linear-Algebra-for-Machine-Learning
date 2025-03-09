import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 10000) # start, finish, n points
y = x**2 + 2*x + 2
fig, ax = plt.subplots()
ax.plot(x,y)
plt.show()